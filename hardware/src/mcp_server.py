"""The MCP endpoint (``POST /mcp`` on the UI server): lets an AI agent watch and drive the rig.

It runs inside the UI process because that process owns the Pico's serial port: the agent and the browser
share one HardwareService, so an agent's E-stop stops a jog from the page and vice versa.

- Auth: ``Authorization: Bearer cpmcp_...``, one of the tokens made in the UI's *AI agents* dialog
  (mcp_tokens.py). Each token has a scope: "read" (watch + emergency_stop) or "operate" (also move the
  cart). A tool outside the caller's scope is never offered and does not exist for it.
- Confirmation: with ``[mcp] require_confirmation = true``, a tool that moves the cart does not run on the
  first call. The reply says so in its text (all the model reads) and carries a short-lived token in the
  result's ``_meta``, which the model never sees. The agent's host asks the person and repeats the call
  with ``_meta: {"cartpole/confirm": token}``. The token is an HMAC over the caller, the tool and its exact
  arguments, so it confirms that one action. A 2026-07-28 client that can show forms gets a native
  elicitation instead. (Same scheme as I-Vision's ``confirm.py``.)
- Audit: every tool call is appended to ``[mcp] audit_file`` as one JSON line; the UI's Activity tab reads
  it back. Entries older than AUDIT_RETENTION_DAYS are dropped at start-up.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp_protocol as protocol
import mcp_tools
from hardware_service import CommandError, HardwareService
from mcp_tokens import TOKEN_PREFIX, Token, TokenStore

CONFIRM_META_KEY = "cartpole/confirm"
CONFIRM_TTL_S = 300
CONFIRMATION_REQUIRED = "confirmation_required"
DECLINED = "declined_by_user"
NATIVE_REQUEST = "confirm"  # the key of the one input request a confirmation asks the client for
AUDIT_RETENTION_DAYS = 90
_MAX_LOGGED_ARGS = 2000

# What an audit entry's ``result`` says (the Activity tab's filter): ran, waiting on the person, the person
# said no, or the tool refused (bad arguments, wrong mode, Pico not connected).
DONE, ASKED, DECLINED_RESULT, REFUSED = "done", "asked", "declined", "refused"

__all__ = ["McpServer", "TOKEN_PREFIX", "CONFIRM_META_KEY"]


class McpServer:
    def __init__(self, service: HardwareService, tokens: TokenStore):
        self.service = service
        self.cfg = service.cfg.mcp
        self.tokens = tokens
        # Confirmation tokens only need to outlive one exchange, so a per-process key is enough.
        self._confirm_key = secrets.token_bytes(32)
        self._audit_path = Path(self.cfg.audit_file)
        self._audit_lock = threading.Lock()
        self._last_call: dict[str, Any] | None = None
        self._prune_audit()

    # ---- HTTP ----------------------------------------------------------------------------------------

    def handle_post(self, headers: dict[str, str], raw_header_names: list[str], body: bytes
                    ) -> tuple[int, dict[str, Any] | list[Any] | None, dict[str, str]]:
        """One POST -> (HTTP status, JSON body or None, extra headers). ``headers`` keyed lowercase."""
        origin = headers.get("origin")
        if origin is not None and origin not in self.cfg.allowed_origins:
            # MCP clients are programs and send no Origin; a browser always does. The spec's guard
            # against a web page on the LAN talking to the motor.
            return 403, {"detail": "Origin not allowed."}, {}
        caller = self.tokens.authenticate(headers.get("authorization"))
        if caller is None:
            return 401, {"detail": f"Send a cart-pole MCP access token ({TOKEN_PREFIX}...) as a Bearer token. "
                                   "Create one in the rig UI under AI agents."}, {"WWW-Authenticate": "Bearer"}
        try:
            message = json.loads(body or b"null")
        except ValueError:
            return 400, protocol.error_reply(None, protocol.PARSE_ERROR, "Body is not valid JSON."), {}

        version = headers.get("mcp-protocol-version")
        request_id = message.get("id") if isinstance(message, dict) else None
        if version and version not in protocol.ALL_VERSIONS:
            return 400, protocol.unsupported_version_reply(request_id, version), {}

        def list_tools() -> list[dict[str, Any]]:
            return self.list_tools(caller)

        def call_tool(name: str, arguments: dict[str, Any], meta: dict[str, Any]) -> protocol.ToolResult:
            return self.call_tool(caller, name, arguments, meta)

        if version in protocol.MODERN_VERSIONS:
            repeated = next((h for h in protocol.ROUTING_HEADERS if raw_header_names.count(h) > 1), None)
            if repeated:
                return 400, protocol.error_reply(request_id, protocol.HEADER_MISMATCH,
                                                 f"{repeated} header appears twice."), {}
            reply, code = protocol.handle_modern(message, headers, list_tools, call_tool)
            return code, reply, {}

        # 2025-03-26 clients may send a batch; later versions never do.
        messages = message if isinstance(message, list) else [message]
        replies = [r for m in messages if (r := protocol.handle(m, list_tools, call_tool))]
        if not replies:
            return 202, None, {}
        return 200, replies if isinstance(message, list) else replies[0], {}

    # ---- tools ---------------------------------------------------------------------------------------

    def list_tools(self, caller: Token) -> list[dict[str, Any]]:
        return [t.schema() for t in mcp_tools.offered(self.service, caller.scope)]

    def call_tool(self, caller: Token, name: str, arguments: dict[str, Any], meta: dict[str, Any]
                  ) -> protocol.ToolResult:
        # A tool outside the scope (or the simulated rig) was never offered: it does not exist here.
        if name not in {t.name for t in mcp_tools.offered(self.service, caller.scope)}:
            raise protocol.RpcError(protocol.INVALID_PARAMS, f"Unknown tool: {name}")
        started = time.perf_counter()
        if (self.cfg.require_confirmation and name in mcp_tools.CONFIRM
                and not self._confirmed(meta.get(CONFIRM_META_KEY), caller, name, arguments)):
            answers = meta.get(protocol.ANSWERS_KEY) or {}
            if not self._confirmed(answers.get("state"), caller, name, arguments):
                return self._ask_first(caller, name, arguments, native=bool(answers.get("can_ask")))
            action = ((answers.get("responses") or {}).get(NATIVE_REQUEST) or {}).get("action")
            if action != "accept":
                self._record(caller, name, arguments, DECLINED_RESULT)
                return protocol.ToolResult(json.dumps({
                    "status": DECLINED, "done": False,
                    "next": "The user did not confirm, so nothing was done. Do not retry unless they ask again.",
                }))
        try:
            result = mcp_tools.run(self.service, name, arguments)
            text, error = json.dumps(result), ""
        except (CommandError, ValueError, FileNotFoundError, ImportError, OSError) as e:
            error = str(e) or type(e).__name__
            text = json.dumps({"error": error})
        self._record(caller, name, arguments, REFUSED if error else DONE, error=error,
                     duration_ms=round((time.perf_counter() - started) * 1000))
        return protocol.ToolResult(text, bool(error))

    def _ask_first(self, caller: Token, name: str, arguments: dict[str, Any], native: bool) -> protocol.ToolResult:
        """Do nothing yet: tell the model so, and hand the host what to ask."""
        try:
            summary = mcp_tools.summary(self.service, name, arguments)
        except ValueError as e:  # bad arguments: nothing to confirm
            self._record(caller, name, arguments, REFUSED, error=str(e))
            return protocol.ToolResult(json.dumps({"error": str(e)}), True)
        token, expires_at = self._issue(caller, name, arguments)
        self._record(caller, name, arguments, ASKED, error=summary)
        text = json.dumps({
            "status": CONFIRMATION_REQUIRED, "summary": summary, "done": False,
            "next": "Nothing has been done. The user is being asked to confirm; tell them what is waiting "
                    "for their answer and do not call this tool again yourself.",
        })
        for_host = {CONFIRM_META_KEY: {"token": token, "summary": summary, "expires_at": expires_at}}
        if not native:
            return protocol.ToolResult(text, meta=for_host)
        # 2026-07-28 with forms: the client itself asks the person (an empty form, just the question) and
        # repeats the call with the answer. The signed token rides as requestState.
        return protocol.ToolResult(text, meta=for_host, input_required={
            "inputRequests": {NATIVE_REQUEST: {"method": "elicitation/create", "params": {
                "mode": "form", "message": summary, "requestedSchema": {"type": "object", "properties": {}}}}},
            "requestState": token,
        })

    # ---- confirmation tokens -------------------------------------------------------------------------

    def _signature(self, caller: Token, tool: str, arguments: dict[str, Any], expires_at: int) -> str:
        call = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
        message = f"{caller.id}\n{tool}\n{call}\n{expires_at}".encode()
        return hmac.new(self._confirm_key, message, hashlib.sha256).hexdigest()

    def _issue(self, caller: Token, tool: str, arguments: dict[str, Any], now: float | None = None) -> tuple[str, int]:
        expires_at = int(now if now is not None else time.time()) + CONFIRM_TTL_S
        return f"{expires_at}.{self._signature(caller, tool, arguments, expires_at)}", expires_at

    def _confirmed(self, token: Any, caller: Token, tool: str, arguments: dict[str, Any],
                   now: float | None = None) -> bool:
        if not isinstance(token, str):
            return False
        expires, _, signature = token.partition(".")
        if not expires.isdigit() or int(expires) < (now if now is not None else time.time()):
            return False
        return hmac.compare_digest(signature, self._signature(caller, tool, arguments, int(expires)))

    # ---- audit ---------------------------------------------------------------------------------------

    def _record(self, caller: Token, tool: str, arguments: dict[str, Any], result: str, *, error: str = "",
                duration_ms: int = 0) -> None:
        args_text = json.dumps(arguments, default=str)
        entry = {
            "at": round(time.time(), 3),
            "token": caller.name,
            "token_id": caller.id,
            "tool": tool,
            "arguments": arguments if len(args_text) <= _MAX_LOGGED_ARGS else {"_truncated": args_text[:_MAX_LOGGED_ARGS]},
            "result": result,
            "ok": result == DONE,
            "error": error[:500],
            "duration_ms": duration_ms,
        }
        with self._audit_lock:
            self._last_call = entry
            try:
                self._audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self._audit_path.open("a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except OSError:
                pass  # never fail a call (least of all an E-stop) over the log

    def _prune_audit(self) -> None:
        if not self._audit_path.exists():
            return
        cutoff = time.time() - AUDIT_RETENTION_DAYS * 86400
        lines = self._audit_path.read_text().splitlines()
        kept = [line for line in lines if _entry_time(line) >= cutoff]
        if len(kept) != len(lines):
            self._audit_path.write_text("".join(line + "\n" for line in kept))

    def activity(self, limit: int = 500) -> list[dict[str, Any]]:
        """The newest ``limit`` audit entries, newest first."""
        with self._audit_lock:
            if not self._audit_path.exists():
                return []
            with self._audit_path.open() as f:
                tail = deque(f, maxlen=limit)
        entries = []
        for line in reversed(tail):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if "result" not in entry:  # written before tokens had names (one shared token, ISO times)
                entry["at"] = _entry_time(line)
                entry.setdefault("token", "default")
                entry["result"] = DONE if entry.get("ok") else REFUSED
            entries.append(entry)
        return entries

    def last_call(self) -> dict[str, Any] | None:
        with self._audit_lock:
            if self._last_call is None:
                return None
            c = self._last_call
            return {"tool": c["tool"], "token": c["token"], "result": c["result"], "at": c["at"]}

    # ---- the web UI ----------------------------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        """What the AI agents dialog shows: settings, tokens, and every tool with its scope."""
        return {
            "require_confirmation": self.cfg.require_confirmation,
            "max_jog_s": self.cfg.max_jog_s,
            "scopes": list(mcp_tools.SCOPES),
            "tokens": self.tokens.list(),
            "tools": [{"name": t.name, "title": t.title, "scope": t.tier, "moves": t.confirm is not None,
                       "read_only": t.read_only}
                      for t in mcp_tools.TOOLS.values() if self.service.fake or not t.fake_only],
        }


def _entry_time(line: str) -> float:
    try:
        at = json.loads(line).get("at")
        if isinstance(at, str):  # ISO timestamps, from the first version of this log
            return datetime.fromisoformat(at).timestamp()
        return float(at) if isinstance(at, (int, float)) else 0.0
    except (ValueError, AttributeError, TypeError):
        return 0.0

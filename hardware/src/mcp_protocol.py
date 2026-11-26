"""The MCP messages this server answers: JSON-RPC 2.0, tools only, no I/O.

Ported from I-Vision's MCP server (``src/modules/mcp/protocol.py``), made synchronous for the stdlib HTTP
server. Two eras, picked per request by the ``MCP-Protocol-Version`` header:

- **Handshake** (2024-11-05 ... 2025-11-25): ``initialize`` first, then plain requests (``handle``).
- **Per-request envelope** (2026-07-28): no handshake. Every request carries its protocol version and the
  client's capabilities in ``params._meta``, mirrored in the ``MCP-Protocol-Version`` / ``Mcp-Method`` /
  ``Mcp-Name`` headers; ``server/discover`` replaces ``initialize`` (``handle_modern``).

Written by hand rather than with the ``mcp`` SDK: a tools-only server is four methods, and the Pi keeps its
dependencies to numpy, pyserial and onnxruntime. The transport is Streamable HTTP in its stateless form:
every POST is answered with plain JSON, there is no session and no server-initiated stream.

``handle`` takes one decoded message and returns the reply, or None when the message needs none (a
notification, or a reply to something we never asked).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Newest first. A client asking for a version not listed is offered the newest. All are the handshake era
# (``initialize``). 2026-07-28 is a different transport (no handshake, a ``server/discover`` probe,
# per-request headers) and its clients fall back to these on the -32022 below.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MODERN_VERSIONS = ("2026-07-28",)
ALL_VERSIONS = MODERN_VERSIONS + PROTOCOL_VERSIONS

# Envelope keys of a 2026-07-28 request's ``params._meta``.
VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"

SERVER_INFO = {"name": "cartpole", "title": "Cart-pole rig", "version": "1.0.0"}

INSTRUCTIONS = (
    "This server drives a real cart-pole rig: a stepper motor moves a cart along a belt-driven rail, a "
    "free pendulum is mounted on the cart, and a Raspberry Pi runs a reinforcement-learning balance "
    "policy (trained in Isaac Lab) at 60 Hz. The motor is real: tools that move the cart (home, "
    "calibrate_track, jog, start_balancing, connect_pico) can hit the end stops or swing the pole, so "
    "only call them when the user asked for that motion. emergency_stop is always available and always "
    "safe. Angles are reported in degrees with 0 = pole upright and +-180 = hanging down; positions in "
    "metres with x = 0 at the rail centre once homed. Start with get_status. The policy only takes over "
    "while the pole is within engage_deg of upright, so a person must lift the pole by hand; tell the "
    "user that rather than expecting the rig to swing it up."
)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
UNSUPPORTED_PROTOCOL_VERSION = -32022
HEADER_MISMATCH = -32020

# The HTTP status a 2026-07-28 error goes out with; anything else is 200.
MODERN_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    INVALID_PARAMS: 400,
    HEADER_MISMATCH: 400,
    UNSUPPORTED_PROTOCOL_VERSION: 400,
    METHOD_NOT_FOUND: 404,
}
ROUTING_HEADERS = ("mcp-protocol-version", "mcp-method", "mcp-name")


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool = False
    # Result ``_meta``: for the agent's host application, never shown to the model.
    meta: dict[str, Any] | None = None
    # 2026-07-28 only: ask the client for input first. ``{"inputRequests": ..., "requestState": ...}``;
    # the reply is then an input-required result, and the client repeats the call with its answers.
    input_required: dict[str, Any] | None = None


# Set by the server on every tools/call (a client cannot supply it): the client's answers to an earlier
# input-required result, and whether it can show a form. ``{"state": str | None, "responses": dict,
# "can_ask": bool}``
ANSWERS_KEY = "cartpole/answers"


ListTools = Callable[[], list[dict[str, Any]]]
# (name, arguments, the request's _meta) -> result
CallTool = Callable[[str, dict[str, Any], dict[str, Any]], ToolResult]


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def error_reply(request_id: Any, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def unsupported_version_reply(request_id: Any, requested: str) -> dict[str, Any]:
    """Names the versions this server speaks, so a newer client falls back to one."""
    return error_reply(
        request_id,
        UNSUPPORTED_PROTOCOL_VERSION,
        f"Unsupported protocol version: {requested}",
        {"supported": list(ALL_VERSIONS), "requested": requested},
    )


def _capabilities() -> dict[str, Any]:
    return {"tools": {"listChanged": False}}


def _initialize(params: dict[str, Any]) -> dict[str, Any]:
    requested = params.get("protocolVersion")
    return {
        "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
        "capabilities": _capabilities(),
        "serverInfo": SERVER_INFO,
        "instructions": INSTRUCTIONS,
    }


def _call(params: dict[str, Any], call_tool: CallTool) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str) or not name:
        raise RpcError(INVALID_PARAMS, "tools/call needs the tool's name.")
    if not isinstance(arguments, dict):
        raise RpcError(INVALID_PARAMS, "tools/call arguments must be an object.")
    meta = dict(params["_meta"]) if isinstance(params.get("_meta"), dict) else {}
    capabilities = meta.get(CAPABILITIES_META_KEY)
    elicitation = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
    responses = params.get("inputResponses")
    meta[ANSWERS_KEY] = {
        "state": params.get("requestState") if isinstance(params.get("requestState"), str) else None,
        "responses": responses if isinstance(responses, dict) else {},
        # Forms only: no "url"-only clients. An empty object means form.
        "can_ask": isinstance(elicitation, dict) and ("form" in elicitation or not elicitation),
    }
    result = call_tool(name, arguments, meta)
    if result.input_required:
        reply: dict[str, Any] = {"resultType": "input_required", **result.input_required}
        if result.meta:
            reply["_meta"] = result.meta
        return reply
    reply = {"content": [{"type": "text", "text": result.text}], "isError": result.is_error}
    if result.meta:
        reply["_meta"] = result.meta
    return reply


def handle(message: Any, list_tools: ListTools, call_tool: CallTool) -> dict[str, Any] | None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return error_reply(None, INVALID_REQUEST, "Not a JSON-RPC 2.0 message.")
    method = message.get("method")
    if not isinstance(method, str):
        return (
            None
            if "result" in message or "error" in message
            else error_reply(message.get("id"), INVALID_REQUEST, "Missing method.")
        )
    if "id" not in message:
        return None
    request_id = message["id"]
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return error_reply(request_id, INVALID_PARAMS, "params must be an object.")
    try:
        if method == "initialize":
            result = _initialize(params)
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": list_tools()}
        elif method == "tools/call":
            result = _call(params, call_tool)
        else:
            raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
    except RpcError as exc:
        return error_reply(request_id, exc.code, exc.message)
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


# ---- 2026-07-28 ----------------------------------------------------------------------------------------


def decode_header_value(value: str | None) -> str | None:
    """A header value as sent: plain, or ``=?base64?...?=`` for anything that is not printable ASCII.
    A malformed wrapper decodes to None, matching nothing."""
    if value is None or not (value.startswith("=?base64?") and value.endswith("?=")):
        return value
    payload = value[len("=?base64?"):-2]
    try:
        raw = base64.b64decode(payload, validate=True)
        return raw.decode("utf-8") if base64.b64encode(raw).decode() == payload else None
    except (binascii.Error, UnicodeDecodeError):
        return None


def _envelope_error(message: dict[str, Any], headers: dict[str, str]) -> RpcError | None:
    """The first check a 2026-07-28 request fails, in the spec's order, or None."""
    params = message.get("params")
    meta = params.get("_meta") if isinstance(params, dict) else None
    if not isinstance(meta, dict):
        return RpcError(INVALID_PARAMS, "params._meta must carry the protocol version and client capabilities.")
    missing = [k for k in (VERSION_META_KEY, CAPABILITIES_META_KEY) if k not in meta]
    if missing:
        return RpcError(INVALID_PARAMS, f"params._meta is missing: {', '.join(missing)}")
    version = meta[VERSION_META_KEY]
    method = message.get("method")
    if headers.get("mcp-protocol-version") != version:
        return RpcError(HEADER_MISMATCH, "MCP-Protocol-Version header does not match params._meta.")
    if headers.get("mcp-method") != method:
        return RpcError(HEADER_MISMATCH, "Mcp-Method header does not match the request's method.")
    if method == "tools/call":
        name = params.get("name")
        if name is not None and decode_header_value(headers.get("mcp-name")) != name:
            return RpcError(HEADER_MISMATCH, "Mcp-Name header does not match the tool's name.")
    if not isinstance(version, str):
        return RpcError(INVALID_PARAMS, "The protocol version must be a string.")
    return None


def handle_modern(
    message: Any,
    headers: dict[str, str],
    list_tools: ListTools,
    call_tool: CallTool,
) -> tuple[dict[str, Any] | None, int]:
    """One 2026-07-28 request -> (reply, HTTP status); (None, 202) for a notification.

    ``headers`` are keyed by lowercase name. One request per POST: no batches, no ``initialize``, no
    ``ping``.
    """
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return error_reply(None, INVALID_REQUEST, "One JSON-RPC 2.0 request per POST."), 400
    if "id" not in message:
        return None, 202  # this revision defines no client notifications to act on
    request_id = message["id"]
    method = message.get("method")
    if not isinstance(method, str):
        return error_reply(request_id, INVALID_REQUEST, "Missing method."), 400

    def error(exc: RpcError) -> tuple[dict[str, Any], int]:
        return error_reply(request_id, exc.code, exc.message), MODERN_HTTP_STATUS.get(exc.code, 200)

    rejected = _envelope_error(message, headers)
    if rejected is not None:
        return error(rejected)
    version = message["params"]["_meta"][VERSION_META_KEY]
    if version not in MODERN_VERSIONS:
        return unsupported_version_reply(request_id, version), 400

    params = message["params"]
    try:
        # Caching hints are required on these two. The rig's state changes every tick, so nothing may
        # be kept.
        no_cache = {"ttlMs": 0, "cacheScope": "private"}
        if method == "server/discover":
            result: dict[str, Any] = {
                "supportedVersions": list(ALL_VERSIONS),
                "capabilities": _capabilities(),
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
                **no_cache,
            }
        elif method == "tools/list":
            result = {"tools": list_tools(), **no_cache}
        elif method == "tools/call":
            result = _call(params, call_tool)
        else:
            raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
    except RpcError as exc:
        return error(exc)
    return {"jsonrpc": "2.0", "id": request_id, "result": {"resultType": "complete", **result}}, 200

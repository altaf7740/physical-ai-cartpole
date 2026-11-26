"""Web UI server: the page, live telemetry over Server-Sent Events, commands over POST.

Standard library only. Routes:
    GET  /             ui.html
    GET  /api/info     static rig/config summary
    GET  /api/stream   SSE: {"status": {...}, "samples": [...]} at [ui] push_hz
    POST /api/command  {"cmd": "...", ...} -> {"ok": true} | {"ok": false, "error": "..."}
    GET  /api/mcp      AI agents dialog: MCP settings, tokens (never the tokens themselves), tools
    GET  /api/mcp/activity   the newest MCP tool calls, newest first
    POST /mcp          MCP (Streamable HTTP, stateless) for AI agents, bearer token; see mcp_server.py
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import Config
from hardware_service import CommandError, HardwareService
from mcp_server import McpServer
from mcp_tokens import TokenError, TokenStore

UI_HTML = Path(__file__).with_name("ui.html")

COMMANDS = {
    "connect": lambda svc, body: svc.connect(),
    "disconnect": lambda svc, body: svc.disconnect(),
    "reset": lambda svc, body: svc.reset_counters(),
    "calibrate": lambda svc, body: svc.start_calibration(float(body.get("speed", 0.1)), full=True),
    "home": lambda svc, body: svc.start_calibration(float(body.get("speed", 0.1)), full=False),
    "estop": lambda svc, body: svc.estop(),
    "jog": lambda svc, body: svc.jog(float(body.get("velocity", 0.0))),
    "policy_start": lambda svc, body: svc.start_policy(bool(body.get("live", False))),
    "policy_stop": lambda svc, body: svc.stop_policy(),
    "clear_fault": lambda svc, body: svc.clear_fault(),
    "record": lambda svc, body: {"recording": svc.set_recording(bool(body.get("on", False)))},
    "lift": lambda svc, body: svc.lift_pole(),
}


def _days(value) -> float | None:
    return None if value in (None, "", 0) else float(value)


def _create_token(mcp: McpServer, body: dict) -> dict:
    row, token = mcp.tokens.create(str(body.get("name", "")), str(body.get("scope", "read")),
                                   _days(body.get("expires_in_days")))
    return {"token": token, "row": row.public()}  # the only time the token itself is returned


# The AI agents dialog. Same trust as the rest of this page: no login, anyone who can open it can drive the
# motor anyway. The tokens keep agents (and other programs) out of /mcp unless someone here let them in.
MCP_COMMANDS = {
    "mcp_create_token": _create_token,
    "mcp_revoke_token": lambda mcp, body: {"revoked": mcp.tokens.revoke(str(body.get("id", ""))).public()},
}


def make_handler(service: HardwareService, push_hz: float, history: int, mcp: McpServer | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cartpole-ui"

        def log_message(self, fmt, *args):
            pass  # the SSE stream would flood the console

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            if self.path == "/mcp":
                self._no_stream()
            elif self.path == "/":
                self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(service.info())
            elif self.path == "/api/stream":
                self._stream()
            elif self.path == "/api/mcp":
                self._json({"enabled": False} if mcp is None else {"enabled": True, **mcp.overview()})
            elif self.path.startswith("/api/mcp/activity"):
                self._json({"entries": [] if mcp is None else mcp.activity()})
            else:
                self._send(404, b"not found", "text/plain")

        def _no_stream(self):
            # Stateless Streamable HTTP: no server-initiated stream (GET) and no session to end (DELETE).
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_DELETE = _no_stream

        def _mcp(self):
            if mcp is None:
                self._send(404, b"MCP is disabled ([mcp] enabled = false)", "text/plain")
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            headers = {k.lower(): v for k, v in self.headers.items()}
            raw_names = [k.lower() for k in self.headers.keys()]
            code, reply, extra = mcp.handle_post(headers, raw_names, body)
            payload = b"" if reply is None else json.dumps(reply).encode()
            self.send_response(code)
            if reply is not None:
                self.send_header("Content-Type", "application/json")
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path == "/mcp":
                self._mcp()
                return
            if self.path != "/api/command":
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                cmd = body.get("cmd")
                if cmd in MCP_COMMANDS:
                    if mcp is None:
                        raise CommandError("MCP is disabled ([mcp] enabled = false)")
                    self._json({"ok": True, **MCP_COMMANDS[cmd](mcp, body)})
                    return
                handler = COMMANDS.get(cmd)
                if handler is None:
                    raise CommandError(f"unknown command {cmd!r}")
                self._json({"ok": True, **(handler(service, body) or {})})
            except (CommandError, TokenError, FileNotFoundError, ValueError, ImportError, OSError) as e:
                self._json({"ok": False, "error": str(e)}, 400)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seq, max_count = 0, history  # first message backfills the chart window
            try:
                while True:
                    samples, seq = service.samples_since(seq, max_count)
                    max_count = 10**9
                    status = service.status()
                    if mcp is not None:
                        status["agent"] = mcp.last_call()
                    msg = json.dumps({"status": status, "samples": samples})
                    self.wfile.write(f"data: {msg}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1.0 / push_hz)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser tab closed

    return Handler


def serve(cfg: Config, service: HardwareService) -> None:
    history = int(cfg.policy.rate_hz * 10)
    mcp = None
    if cfg.mcp.enabled:
        mcp = McpServer(service, TokenStore(Path(cfg.mcp.tokens_file)))
    server = ThreadingHTTPServer((cfg.ui.host, cfg.ui.port), make_handler(service, cfg.ui.push_hz, history, mcp))
    server.daemon_threads = True
    print(f"UI: http://{cfg.ui.host}:{cfg.ui.port}  ({'simulated Pico' if service.fake else cfg.serial.port})")
    if mcp is not None:
        print(f"MCP: http://{cfg.ui.host}:{cfg.ui.port}/mcp  {len(mcp.tokens.list())} token(s)"
              f"{'  confirm=on' if cfg.mcp.require_confirmation else ''}  (tokens: UI -> AI agents)")
    print("Ctrl+C to stop.")
    service.connect()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()

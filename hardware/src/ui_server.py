"""Web UI server: the page, live telemetry over Server-Sent Events, commands over POST.

Standard library only. Routes:
    GET  /             ui.html
    GET  /api/info     static rig/config summary
    GET  /api/stream   SSE: {"status": {...}, "samples": [...]} at [ui] push_hz
    POST /api/command  {"cmd": "...", ...} -> {"ok": true} | {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import Config
from hardware_service import CommandError, HardwareService

UI_HTML = Path(__file__).with_name("ui.html")

COMMANDS = {
    "connect": lambda svc, body: svc.connect(),
    "disconnect": lambda svc, body: svc.disconnect(),
    "reset": lambda svc, body: svc.reset_counters(),
    "calibrate": lambda svc, body: svc.start_calibration(float(body.get("speed", 0.1))),
    "estop": lambda svc, body: svc.estop(),
    "jog": lambda svc, body: svc.jog(float(body.get("velocity", 0.0))),
    "policy_start": lambda svc, body: svc.start_policy(bool(body.get("live", False))),
    "policy_stop": lambda svc, body: svc.stop_policy(),
    "clear_fault": lambda svc, body: svc.clear_fault(),
    "record": lambda svc, body: {"recording": svc.set_recording(bool(body.get("on", False)))},
    "lift": lambda svc, body: svc.lift_pole(),
}


def make_handler(service: HardwareService, push_hz: float, history: int):
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
            if self.path == "/":
                self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(service.info())
            elif self.path == "/api/stream":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/command":
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                handler = COMMANDS.get(body.get("cmd"))
                if handler is None:
                    raise CommandError(f"unknown command {body.get('cmd')!r}")
                self._json({"ok": True, **(handler(service, body) or {})})
            except (CommandError, FileNotFoundError, ValueError, ImportError, OSError) as e:
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
                    msg = json.dumps({"status": service.status(), "samples": samples})
                    self.wfile.write(f"data: {msg}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1.0 / push_hz)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser tab closed

    return Handler


def serve(cfg: Config, service: HardwareService) -> None:
    history = int(cfg.policy.rate_hz * 10)
    server = ThreadingHTTPServer((cfg.ui.host, cfg.ui.port), make_handler(service, cfg.ui.push_hz, history))
    server.daemon_threads = True
    print(f"UI: http://{cfg.ui.host}:{cfg.ui.port}  ({'simulated Pico' if service.fake else cfg.serial.port})")
    print("Ctrl+C to stop.")
    service.connect()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()

"""Live status web page for the Pi controller (Python standard library only).

The control loop calls Telemetry.add() once per step; it only appends to a buffer under a short lock, so a slow
browser or a dropped Wi-Fi link can never delay the loop. The web server runs in its own threads:
    GET  /         the page (web_ui.html)
    GET  /events   Server-Sent Events: an "info" message, then batches of new samples every 50 ms
    POST /control  {"cmd": "start" | "stop" | "estop" | "reset"}: queued for the control loop to act on
The server never touches the motor itself; the loop decides what a command means.
"""

import collections
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui.html")
COMMANDS = ("start", "stop", "estop", "reset")


class Telemetry:
    def __init__(self, info: dict, history: int = 600):
        self.info = info  # static facts for the page (limits, rate, mode)
        self._samples = collections.deque(maxlen=history)  # (seq, dict)
        self._seq = 0
        self._commands = []
        self._lock = threading.Lock()

    def add(self, sample: dict):
        with self._lock:
            self._seq += 1
            self._samples.append((self._seq, sample))

    def since(self, seq: int) -> tuple[int, list]:
        with self._lock:
            new = [s for n, s in self._samples if n > seq]
            return self._seq, new

    def push_command(self, cmd: str):
        with self._lock:
            self._commands.append(cmd)

    def pop_commands(self) -> list:
        with self._lock:
            cmds, self._commands = self._commands, []
            return cmds


def serve(telemetry: Telemetry, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = open(PAGE, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self._send({"type": "info", **telemetry.info})
                    seq, history = telemetry.since(0)  # what is already buffered, so plots start full
                    self._send({"type": "samples", "samples": history})
                    while True:
                        time.sleep(0.05)
                        seq, new = telemetry.since(seq)
                        if new:
                            self._send({"type": "samples", "samples": new})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return  # browser went away
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/control":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                cmd = json.loads(self.rfile.read(length) or b"{}").get("cmd")
            except (ValueError, AttributeError):
                cmd = None
            if cmd not in COMMANDS:
                self.send_error(400, "cmd must be one of " + ", ".join(COMMANDS))
                return
            telemetry.push_command(cmd)
            self.send_response(204)
            self.end_headers()

        def _send(self, msg: dict):
            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *args):  # keep the console for the controller
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

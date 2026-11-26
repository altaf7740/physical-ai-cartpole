"""The MCP server: protocol replies, which tools it offers, confirmation, and the HTTP endpoint.

Runs against the simulated Pico, so what is pinned here is the wire behaviour an agent depends on.
"""

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

import mcp_protocol as protocol
import mcp_tools
from config import load_config
from fake_pico import FakePico
from hardware_service import HardwareService
from mcp_server import CONFIRM_META_KEY, McpServer
from mcp_tokens import LEGACY_FILE, TOKEN_PREFIX, TokenError, TokenStore
from ui_server import make_handler

TEMPLATE = Path(__file__).resolve().parents[2] / "config.toml.template"


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def request(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def no_call(name, arguments, meta):  # pragma: no cover - must not be reached
    raise AssertionError("tool called")


class TestProtocol(unittest.TestCase):
    def test_initialize_echoes_a_version_it_supports(self):
        reply = protocol.handle(request("initialize", {"protocolVersion": "2025-03-26"}), list, no_call)
        self.assertEqual(reply["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(reply["result"]["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(reply["result"]["serverInfo"]["name"], "cartpole")

    def test_initialize_offers_its_newest_version_for_an_unknown_one(self):
        reply = protocol.handle(request("initialize", {"protocolVersion": "2099-01-01"}), list, no_call)
        self.assertEqual(reply["result"]["protocolVersion"], protocol.PROTOCOL_VERSIONS[0])

    def test_notifications_get_no_reply(self):
        self.assertIsNone(protocol.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, list, no_call))

    def test_unknown_method_is_a_json_rpc_error(self):
        reply = protocol.handle(request("resources/list"), list, no_call)
        self.assertEqual(reply["error"]["code"], protocol.METHOD_NOT_FOUND)

    def test_tool_call_wraps_the_result_as_text_content(self):
        reply = protocol.handle(request("tools/call", {"name": "t", "arguments": {"a": 1}}), list,
                                lambda n, a, m: protocol.ToolResult(json.dumps({"name": n, **a})))
        self.assertFalse(reply["result"]["isError"])
        self.assertEqual(json.loads(reply["result"]["content"][0]["text"]), {"name": "t", "a": 1})

    def test_client_cannot_forge_answers(self):
        seen = {}

        def call(name, arguments, meta):
            seen.update(meta)
            return protocol.ToolResult("ok")

        forged = {"state": "x", "responses": {"confirm": {"action": "accept"}}, "can_ask": True}
        protocol.handle(request("tools/call", {"name": "jog", "_meta": {protocol.ANSWERS_KEY: forged}}), list, call)
        self.assertEqual(seen[protocol.ANSWERS_KEY], {"state": None, "responses": {}, "can_ask": False})

    def test_modern_discover_and_header_checks(self):
        meta = {protocol.VERSION_META_KEY: "2026-07-28", protocol.CAPABILITIES_META_KEY: {}}
        headers = {"mcp-protocol-version": "2026-07-28", "mcp-method": "server/discover"}
        reply, code = protocol.handle_modern(request("server/discover", {"_meta": meta}), headers, list, no_call)
        self.assertEqual(code, 200)
        self.assertEqual(reply["result"]["resultType"], "complete")
        self.assertIn("2026-07-28", reply["result"]["supportedVersions"])
        reply, code = protocol.handle_modern(request("tools/list", {"_meta": meta}), headers, list, no_call)
        self.assertEqual((code, reply["error"]["code"]), (400, protocol.HEADER_MISMATCH))

    def test_base64_header_values(self):
        self.assertEqual(protocol.decode_header_value("=?base64?am9n?="), "jog")
        self.assertIsNone(protocol.decode_header_value("=?base64?!!?="))


class ServiceCase(unittest.TestCase):
    mcp_overrides: dict = {}
    scope = "operate"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config(TEMPLATE)
        mcp = replace(cfg.mcp, audit_file=str(Path(self.tmp.name) / "audit.jsonl"), **self.mcp_overrides)
        self.cfg = replace(cfg, ui=replace(cfg.ui, home_on_connect=False), mcp=mcp)
        self.svc = HardwareService(self.cfg, lambda: FakePico(self.cfg), fake=True)
        self.tokens = TokenStore(Path(self.tmp.name) / "mcp_tokens.json")
        self.row, self.token = self.tokens.create("test", self.scope)
        self.server = McpServer(self.svc, self.tokens)

    def tearDown(self):
        self.svc.close()
        self.tmp.cleanup()

    def call(self, name, arguments=None, meta=None):
        return protocol.handle(request("tools/call", {"name": name, "arguments": arguments or {}, "_meta": meta or {}}),
                               self.list_tools, self.call_tool)

    def list_tools(self):
        return self.server.list_tools(self.row)

    def call_tool(self, name, arguments, meta):
        return self.server.call_tool(self.row, name, arguments, meta)

    def connect(self):
        self.svc.connect()
        self.assertTrue(wait_for(lambda: self.svc.status()["link"] == "connected"))


class TestTools(ServiceCase):
    def test_every_offered_tool_has_a_valid_schema(self):
        tools = self.list_tools()
        self.assertEqual({t["name"] for t in tools}, set(mcp_tools.TOOLS))
        for tool in tools:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])
        flagged = {t["name"] for t in tools if t.get("_meta", {}).get(mcp_tools.CONFIRM_META_KEY)}
        self.assertEqual(flagged, mcp_tools.CONFIRM)

    def test_status_and_telemetry(self):
        self.connect()
        self.assertTrue(wait_for(lambda: len(self.svc.samples_since(0, 100)[0]) > 30))
        status = json.loads(self.call("get_status")["result"]["content"][0]["text"])
        self.assertEqual((status["link"], status["control"]), ("connected", "manual"))
        self.assertAlmostEqual(abs(status["latest"]["theta_deg"]), 180, delta=2)  # hanging down
        tel = json.loads(self.call("get_telemetry", {"seconds": 0.5, "max_points": 5})["result"]["content"][0]["text"])
        self.assertLessEqual(len(tel["samples"]), 6)
        self.assertEqual(tel["summary"]["count"], 30)

    def test_jog_moves_for_its_duration(self):
        self.connect()
        reply = self.call("jog", {"velocity_mps": 0.1, "duration_s": 0.3})
        self.assertFalse(reply["result"]["isError"])
        self.assertTrue(wait_for(lambda: self.svc.samples_since(0, 1)[0][-1]["cmd"] > 0))
        self.assertTrue(wait_for(lambda: self.svc.samples_since(0, 1)[0][-1]["cmd"] == 0, timeout=1.5))
        self.assertAlmostEqual(self.svc.samples_since(0, 1)[0][-1]["x"], 0.03, delta=0.01)

    def test_errors_are_tool_errors_not_rpc_errors(self):
        reply = self.call("jog", {"velocity_mps": 0.1})  # not connected
        self.assertTrue(reply["result"]["isError"])
        self.assertIn("Pico is", json.loads(reply["result"]["content"][0]["text"])["error"])
        reply = self.call("jog", {"velocity_mps": 0.1, "duration_s": 99})
        self.assertIn("duration_s", reply["result"]["content"][0]["text"])

    def test_unknown_tool_is_an_rpc_error(self):
        self.assertEqual(self.call("format_disk")["error"]["code"], protocol.INVALID_PARAMS)

    def test_calls_are_audited(self):
        self.call("emergency_stop")
        self.call("jog", {"velocity_mps": 0.1})  # refused: not connected
        newest, older = self.server.activity()[:2]
        self.assertEqual((newest["tool"], newest["result"], newest["token"]), ("jog", "refused", "test"))
        self.assertEqual((older["tool"], older["result"]), ("emergency_stop", "done"))
        self.assertEqual(self.server.last_call()["tool"], "jog")


    def test_entries_from_the_first_log_format_still_read(self):
        Path(self.cfg.mcp.audit_file).write_text(
            '{"at": "2026-11-26T09:20:00.000+00:00", "tool": "get_status", "arguments": {}, "ok": true, "error": "", "duration_ms": 1}\n')
        entry = self.server.activity()[0]
        self.assertEqual((entry["token"], entry["result"]), ("default", "done"))
        self.assertGreater(entry["at"], 1.7e9)


class TestReadScope(ServiceCase):
    scope = "read"

    def test_read_scope_can_watch_and_stop_only(self):
        names = {t["name"] for t in self.list_tools()}
        self.assertEqual(names, {"get_status", "get_rig_info", "get_telemetry", "emergency_stop"})
        self.assertEqual(self.call("jog", {"velocity_mps": 0.1})["error"]["code"], protocol.INVALID_PARAMS)


class TestConfirmation(ServiceCase):
    mcp_overrides = {"require_confirmation": True}

    def test_motion_waits_for_a_token_bound_to_the_exact_call(self):
        self.connect()
        args = {"velocity_mps": 0.1, "duration_s": 0.2}
        first = self.call("jog", args)["result"]
        self.assertEqual(json.loads(first["content"][0]["text"])["status"], "confirmation_required")
        self.assertEqual(self.svc.samples_since(0, 1)[0][-1]["cmd"], 0)
        token = first["_meta"][CONFIRM_META_KEY]["token"]
        other = self.call("jog", {**args, "velocity_mps": -0.2}, {CONFIRM_META_KEY: token})["result"]
        self.assertIn("confirmation_required", other["content"][0]["text"])  # different arguments
        done = self.call("jog", args, {CONFIRM_META_KEY: token})["result"]
        self.assertTrue(json.loads(done["content"][0]["text"])["jogging"])
        self.assertEqual([e["result"] for e in self.server.activity()[:3]], ["done", "asked", "asked"])

    def test_stop_and_reads_never_ask(self):
        self.assertFalse(self.call("emergency_stop")["result"]["isError"])
        self.assertNotIn("_meta", self.call("get_status")["result"])

    def test_native_elicitation_for_modern_clients(self):
        self.connect()
        meta = {protocol.VERSION_META_KEY: "2026-07-28", protocol.CAPABILITIES_META_KEY: {"elicitation": {"form": {}}}}
        headers = {"mcp-protocol-version": "2026-07-28", "mcp-method": "tools/call", "mcp-name": "home"}
        params = {"name": "home", "arguments": {}, "_meta": meta}
        reply, _ = protocol.handle_modern(request("tools/call", params), headers, self.list_tools, self.call_tool)
        self.assertEqual(reply["result"]["resultType"], "input_required")
        state = reply["result"]["requestState"]
        declined = {**params, "requestState": state, "inputResponses": {"confirm": {"action": "decline"}}}
        reply, _ = protocol.handle_modern(request("tools/call", declined), headers, self.list_tools, self.call_tool)
        self.assertIn("declined_by_user", reply["result"]["content"][0]["text"])
        self.assertEqual(self.svc.status()["control"], "manual")
        accepted = {**declined, "inputResponses": {"confirm": {"action": "accept"}}}
        reply, _ = protocol.handle_modern(request("tools/call", accepted), headers, self.list_tools, self.call_tool)
        self.assertEqual(self.svc.status()["control"], "calibrating")
        self.svc.estop()


class TestEndpoint(ServiceCase):
    def setUp(self):
        super().setUp()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.svc, 20, 600, self.server))
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}/mcp"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        super().tearDown()

    def post(self, body, token="", **headers):
        token = self.token if token == "" else token
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            **({"Authorization": f"Bearer {token}"} if token else {}), **headers})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                raw = r.read()
                return r.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def test_handshake_list_and_call_over_http(self):
        code, reply = self.post(request("initialize", {"protocolVersion": "2025-06-18"}))
        self.assertEqual((code, reply["result"]["protocolVersion"]), (200, "2025-06-18"))
        code, reply = self.post({"jsonrpc": "2.0", "method": "notifications/initialized"},
                                **{"MCP-Protocol-Version": "2025-06-18"})
        self.assertEqual((code, reply), (202, None))
        code, reply = self.post(request("tools/list"), **{"MCP-Protocol-Version": "2025-06-18"})
        self.assertIn("emergency_stop", {t["name"] for t in reply["result"]["tools"]})
        code, reply = self.post(request("tools/call", {"name": "get_rig_info", "arguments": {}}))
        self.assertEqual(json.loads(reply["result"]["content"][0]["text"])["rate_hz"], 60)

    def test_auth_and_origin(self):
        self.assertEqual(self.post(request("tools/list"), token=None)[0], 401)
        self.assertEqual(self.post(request("tools/list"), token=TOKEN_PREFIX + "wrong")[0], 401)
        self.assertEqual(self.post(request("tools/list"), Origin="http://evil.example")[0], 403)

    def test_unknown_version_header_is_rejected(self):
        code, reply = self.post(request("tools/list"), **{"MCP-Protocol-Version": "1999-01-01"})
        self.assertEqual((code, reply["error"]["code"]), (400, protocol.UNSUPPORTED_PROTOCOL_VERSION))

    def get(self, path):
        with urllib.request.urlopen(self.url.replace("/mcp", path), timeout=5) as r:
            return json.loads(r.read())

    def command(self, **body):
        req = urllib.request.Request(self.url.replace("/mcp", "/api/command"), data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def test_ui_creates_lists_and_revokes_tokens(self):
        created = self.command(cmd="mcp_create_token", name="laptop", scope="read", expires_in_days=7)
        self.assertTrue(created["ok"] and created["token"].startswith(TOKEN_PREFIX))
        overview = self.get("/api/mcp")
        self.assertEqual({t["name"] for t in overview["tokens"]}, {"test", "laptop"})
        self.assertNotIn(created["token"], json.dumps(overview))  # never shown again
        self.assertNotIn("token_hash", json.dumps(overview))
        code, reply = self.post(request("tools/list"), token=created["token"])
        self.assertNotIn("jog", {t["name"] for t in reply["result"]["tools"]})
        self.post(request("tools/call", {"name": "get_status"}), token=created["token"])
        self.assertEqual(self.get("/api/mcp/activity")["entries"][0]["token"], "laptop")
        self.assertTrue(self.command(cmd="mcp_revoke_token", id=created["row"]["id"])["ok"])
        self.assertEqual(self.post(request("tools/list"), token=created["token"])[0], 401)
        self.assertFalse(self.command(cmd="mcp_create_token", name="", scope="read")["ok"])

    def test_get_is_405(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.url, timeout=5)
        self.assertEqual(ctx.exception.code, 405)


class TestTokenStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mcp_tokens.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_a_hash_is_stored_privately_and_reloaded(self):
        store = TokenStore(self.path)
        row, token = store.create("agent", "operate")
        self.assertNotIn(token, self.path.read_text())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        again = TokenStore(self.path)
        self.assertEqual(again.authenticate(f"Bearer {token}").id, row.id)
        self.assertIsNone(again.authenticate(f"Bearer {token}x"))
        self.assertIsNone(again.authenticate(token))  # no scheme

    def test_expired_and_revoked_tokens_are_refused(self):
        store = TokenStore(self.path)
        row, token = store.create("short", "read", expires_in_days=1)
        row.expires_at = time.time() - 1
        self.assertIsNone(store.authenticate(f"Bearer {token}"))
        self.assertTrue(store.list()[0]["expired"])
        store.revoke(row.id)
        self.assertEqual(store.list(), [])
        with self.assertRaises(TokenError):
            store.revoke(row.id)

    def test_names_and_scopes_are_checked(self):
        store = TokenStore(self.path)
        store.create("a", "read")
        for name, scope in (("a", "read"), ("", "read"), ("b", "full")):
            with self.assertRaises(TokenError):
                store.create(name, scope)

    def test_the_old_single_token_is_imported_once(self):
        legacy = self.path.with_name(LEGACY_FILE)
        legacy.write_text(TOKEN_PREFIX + "y" * 43 + "\n")
        store = TokenStore(self.path)
        self.assertFalse(legacy.exists())
        self.assertEqual(store.authenticate(f"Bearer {TOKEN_PREFIX}{'y' * 43}").name, "default")
        self.assertEqual(len(TokenStore(self.path).list()), 1)


if __name__ == "__main__":
    unittest.main()

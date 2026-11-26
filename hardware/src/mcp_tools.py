"""The tools an MCP agent is offered: thin wrappers over HardwareService, the same calls the web UI makes.

Each tool has a tier. A token's ``scope`` (``[mcp] scope``) reaches its own tier and the ones below it:

- ``read``: look at the rig (status, telemetry, config) and emergency_stop. Stopping is always allowed.
- ``operate``: everything that changes the rig, including moving the cart.

Tools in ``CONFIRM`` move the cart. With ``[mcp] require_confirmation = true`` they only run once the
person has agreed (see mcp_server.py); their ``confirm`` function says what will happen.

A handler returns a JSON-able dict or raises CommandError / ValueError, which the agent gets back as an
error result with the message.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hardware_service import CommandError, HardwareService

SCOPES: tuple[str, ...] = ("read", "operate")

# Tool _meta key flagging the tools that may ask the person first.
CONFIRM_META_KEY = "cartpole/requiresConfirmation"

Handler = Callable[[HardwareService, dict[str, Any]], dict[str, Any]]
Summary = Callable[[HardwareService, dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    handler: Handler
    tier: str = "read"
    properties: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    # Moves the cart: the human-readable summary of what will happen, for the confirmation.
    confirm: Summary | None = None
    read_only: bool = False
    fake_only: bool = False

    def schema(self) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.properties,
                **({"required": list(self.required)} if self.required else {}),
                "additionalProperties": False,
            },
            "annotations": {
                "title": self.title,
                "readOnlyHint": self.read_only,
                "destructiveHint": self.confirm is not None,
                "idempotentHint": self.read_only,
                "openWorldHint": False,
            },
        }
        if self.confirm is not None:
            tool["_meta"] = {CONFIRM_META_KEY: True}
        return tool


# ---- argument helpers ----------------------------------------------------------------------------------


def _number(args: dict[str, Any], key: str, default: float | None = None) -> float:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _flag(args: dict[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _deg(rad: float | None, digits: int = 2) -> float | None:
    return None if rad is None else round(math.degrees(rad), digits)


def _sample_view(s: dict[str, Any]) -> dict[str, Any]:
    """One telemetry sample in units an agent reads directly."""
    return {
        "t_s": s["t"],
        "x_m": s["x"],
        "x_dot_mps": s["dx"],
        "theta_deg": _deg(s["theta"]),
        "theta_dot_dps": _deg(s["dtheta"], 1),
        "step_rate_cmd": s["cmd"],
        "policy_action": s["action"],
        "safety": s["safety"] or None,
        "limit_switch": {0: None, 1: "left", 2: "right", 3: "both"}.get(s["limits"], s["limits"]),
        "reply_ok": s["ok"],
    }


def _latest(svc: HardwareService) -> dict[str, Any] | None:
    samples, _ = svc.samples_since(0, 1)
    return _sample_view(samples[-1]) if samples else None


# ---- read ----------------------------------------------------------------------------------------------


def get_status(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    status = svc.status()
    return {
        **status,
        "simulated": svc.fake,
        "latest": _latest(svc),
        "hint": _hint(status),
    }


def _hint(status: dict[str, Any]) -> str:
    if status["link"] != "connected":
        return f"The Pico is {status['link']}{': ' + status['error'] if status['error'] else ''}. connect_pico reconnects (it homes the cart if home_on_connect is set)."
    if status["control"] == "calibrating":
        return "Homing/calibrating: the cart is moving. Poll get_status until control is back to manual."
    if status["control"] in ("live", "dry-run"):
        return ("The policy is running" + (" and driving the motor" if status["control"] == "live" else " with the motor off")
                + ". latest.safety is idle until a person holds the pole within engage_deg of upright, balancing while it is in control, fault when latched (clear_fault).")
    track = status.get("track")
    if not track or not track.get("homed"):
        return "Manual mode, not homed: x is not centred. home (or calibrate_track) before balancing."
    return "Manual mode, homed. start_balancing runs the policy on the motor; start_dry_run runs it with the motor off."


def get_rig_info(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    return svc.info()


def get_telemetry(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    seconds = min(max(_number(args, "seconds", 2.0), 0.05), 60.0)
    max_points = int(min(max(_number(args, "max_points", 30), 1), 600))
    rate = svc.cfg.policy.rate_hz
    samples, _ = svc.samples_since(0, max(1, round(seconds * rate)))
    if not samples:
        return {"samples": [], "summary": None, "note": "no samples yet: the Pico is not connected"}
    stride = max(1, math.ceil(len(samples) / max_points))
    picked = samples[::stride]
    if picked[-1] is not samples[-1]:
        picked.append(samples[-1])
    thetas = [abs(s["theta"]) for s in samples if s["theta"] is not None]
    xs = [s["x"] for s in samples if s["x"] is not None]
    safety = [s["safety"] for s in samples if s["safety"]]
    summary = {
        "count": len(samples),
        "duration_s": round(samples[-1]["t"] - samples[0]["t"], 3),
        "max_abs_theta_deg": _deg(max(thetas)) if thetas else None,
        "mean_abs_theta_deg": _deg(sum(thetas) / len(thetas)) if thetas else None,
        "x_range_m": [min(xs), max(xs)] if xs else None,
        "balancing_fraction": round(safety.count("balancing") / len(samples), 3) if safety else 0.0,
        "missed_replies": sum(1 for s in samples if not s["ok"]),
        "limit_switch_hit": any(s["limit"] for s in samples),
    }
    return {"summary": summary, "every_nth_sample": stride, "samples": [_sample_view(s) for s in picked]}


def emergency_stop(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.estop()
    return {"stopped": True, "control": svc.status()["control"]}


# ---- operate -------------------------------------------------------------------------------------------


def _speed(svc: HardwareService, args: dict[str, Any]) -> float:
    return min(abs(_number(args, "speed_mps", svc.cfg.ui.home_speed_mps)), svc.max_velocity)


def home(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.start_calibration(_speed(svc, args), full=False)
    return {"started": True, "calibration": svc.status()["calibration"],
            "next": "The cart is moving. Poll get_status until control is manual and calibration says homed."}


def calibrate_track(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.start_calibration(_speed(svc, args), full=True)
    return {"started": True, "calibration": svc.status()["calibration"],
            "next": "The cart is moving between both limit switches (twice). Poll get_status until control is manual."}


def _jog_args(svc: HardwareService, args: dict[str, Any]) -> tuple[float, float]:
    velocity = _number(args, "velocity_mps")
    duration = _number(args, "duration_s", 0.5)
    cap = svc.cfg.mcp.max_jog_s
    if not 0 < duration <= cap:
        raise ValueError(f"duration_s must be in (0, {cap}]")
    return max(-svc.max_velocity, min(velocity, svc.max_velocity)), duration


def jog(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    velocity, duration = _jog_args(svc, args)
    svc.jog(velocity, duration_s=duration)
    return {"jogging": True, "velocity_mps": velocity, "duration_s": duration,
            "expected_distance_m": round(velocity * duration, 4),
            "note": "The jog stops by itself after duration_s; the Pico also stops it at a limit switch."}


def start_balancing(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.start_policy(live=True)
    return {"control": svc.status()["control"],
            "next": f"The policy is armed with the motor on. It engages once a person holds the pole within "
                    f"{svc.cfg.safety.engage_angle_deg} deg of upright and lets go above "
                    f"{svc.cfg.safety.disengage_angle_deg} deg. Watch it with get_telemetry."}


def start_dry_run(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.start_policy(live=False)
    return {"control": svc.status()["control"],
            "note": "The policy runs on live sensors; its actions are logged (policy_action) but the motor stays off."}


def reset_counters(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    homed = bool((svc.status().get("track") or {}).get("homed"))
    svc.reset_counters()
    return {"reset": "pole angle only (x stays centred)" if homed else "pole angle and cart position",
            "note": "theta = 0 is taken from where the pole is now, per angle_zero in config.toml "
                    f"({svc.cfg.hardware.angle_zero!r}), so the pole must be at rest there."}


def clear_fault(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.clear_fault()
    return {"cleared": True}


def set_recording(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    return {"recording": svc.set_recording(_flag(args, "on"))}


def connect_pico(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.connect()
    return {"link": svc.status()["link"],
            "note": "Homes the cart on the limit switches once connected" if svc.cfg.ui.home_on_connect else ""}


def disconnect_pico(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.disconnect()
    return {"link": svc.status()["link"]}


def lift_pole(svc: HardwareService, args: dict[str, Any]) -> dict[str, Any]:
    svc.lift_pole()
    return {"lifted": True, "note": "Simulated: the pole is placed just off upright and let go."}


# ---- confirmation summaries ----------------------------------------------------------------------------


def _speed_text(svc: HardwareService, args: dict[str, Any]) -> str:
    return f"{_speed(svc, args):.3f} m/s"


def _jog_summary(svc: HardwareService, args: dict[str, Any]) -> str:
    v, d = _jog_args(svc, args)
    side = "right (+x)" if v > 0 else "left (-x)"
    return f"Jog the cart {side} at {abs(v):.3f} m/s for {d:.2f} s (about {abs(v * d) * 100:.1f} cm)."


SPEED = {"speed_mps": {"type": "number", "exclusiveMinimum": 0,
                       "description": "Cart speed in m/s; default [ui] home_speed_mps, capped at the rig's max speed."}}

TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("get_status", "Rig status",
         "The rig right now: Pico link, control mode (manual / dry-run / live / calibrating), faults, loop "
         "health (rate_hz, late_ticks, missed_reads), track calibration and homing, recording, and the latest "
         "sample (cart x, pole theta in degrees, velocities, step rate, policy safety state), with a hint on "
         "what to do next.", get_status, read_only=True),
    Tool("get_rig_info", "Rig configuration",
         "Static configuration: control rate, engage / disengage angles, max cart speed, policy model path, "
         "observation order and scale, action type.", get_rig_info, read_only=True),
    Tool("get_telemetry", "Recent telemetry",
         "The last `seconds` of 60 Hz telemetry (at most 60 s), downsampled to about `max_points` samples, "
         "with a summary: max / mean |theta|, the x range, the fraction of time the policy was balancing, "
         "missed replies, and whether a limit switch was hit. Use it to judge how well the pole is balanced.",
         get_telemetry, read_only=True,
         properties={"seconds": {"type": "number", "minimum": 0.05, "maximum": 60, "default": 2},
                     "max_points": {"type": "integer", "minimum": 1, "maximum": 600, "default": 30}}),
    Tool("emergency_stop", "Emergency stop",
         "Stop the motor now: ends the policy, any jog and any homing / calibration, and returns to manual. "
         "Always allowed, always safe; use it whenever something looks wrong.", emergency_stop),
    Tool("home", "Home the cart",
         "MOVES THE CART. Drive to the left limit switch, then half the saved track travel, so x = 0 is the "
         "rail centre (a full calibration if no travel is saved). Needs manual mode. Poll get_status until "
         "control is manual again.", home, tier="operate", properties=SPEED,
         confirm=lambda svc, a: f"Home the cart: drive to the left limit switch at {_speed_text(svc, a)}, then to the rail centre."),
    Tool("calibrate_track", "Calibrate the track",
         "MOVES THE CART. Measure the travel between the limit switches: left, right, twice, then park in the "
         "middle as x = 0 and save the result. Run after changing the rail, pulley or microstepping.",
         calibrate_track, tier="operate", properties=SPEED,
         confirm=lambda svc, a: f"Calibrate the track: drive between both limit switches twice at {_speed_text(svc, a)}, then park at the centre."),
    Tool("jog", "Jog the cart",
         "MOVES THE CART at velocity_mps (+ = right / +x, - = left) for duration_s, then stops by itself. The "
         "speed is capped at the rig's max speed and the Pico stops at a limit switch. Needs manual mode.",
         jog, tier="operate", required=("velocity_mps",), confirm=_jog_summary,
         properties={"velocity_mps": {"type": "number", "description": "m/s, + = right (+x), - = left"},
                     "duration_s": {"type": "number", "exclusiveMinimum": 0, "default": 0.5,
                                    "description": "seconds, at most [mcp] max_jog_s"}}),
    Tool("start_balancing", "Start balancing (motor on)",
         "MOVES THE CART. Run the balance policy with the motor on. It does nothing until a person holds the "
         "pole near upright (within engage_deg), then balances it, and lets go (motor stop) past "
         "disengage_deg. Home first so the cart starts at the rail centre.", start_balancing, tier="operate",
         confirm=lambda svc, a: "Start the balance policy with the MOTOR ON: the cart moves on its own whenever the pole is held near upright."),
    Tool("start_dry_run", "Start a dry run (motor off)",
         "Run the balance policy on live sensors with the motor off, to check what it would do.",
         start_dry_run, tier="operate"),
    Tool("reset_counters", "Reset counters",
         "Zero the pole angle at its current position (and the cart position if not homed). The pole must be "
         "at rest where config.toml's angle_zero says (hanging down by default).", reset_counters, tier="operate"),
    Tool("clear_fault", "Clear fault",
         "Clear a latched policy fault (limit switch, lost link) so the policy can engage again.",
         clear_fault, tier="operate"),
    Tool("set_recording", "Record CSV",
         "Start (on = true) or stop recording every control tick to a CSV in logs/.", set_recording,
         tier="operate", required=("on",), properties={"on": {"type": "boolean"}}),
    Tool("connect_pico", "Connect to the Pico",
         "Open the serial link to the Pico (resets its counters). With [ui] home_on_connect the cart then "
         "homes by itself, so this MOVES THE CART.", connect_pico, tier="operate",
         confirm=lambda svc, a: "Connect to the Pico" + (" and home the cart on the limit switches." if svc.cfg.ui.home_on_connect else ".")),
    Tool("disconnect_pico", "Disconnect the Pico",
         "Stop the control loop and close the serial link (the motor is commanded to 0).", disconnect_pico,
         tier="operate"),
    Tool("lift_pole", "Lift the pole (simulated)",
         "Simulated rig only: place the pole just off upright and let go, as a hand would.", lift_pole,
         tier="operate", fake_only=True),
)}

CONFIRM = {name for name, t in TOOLS.items() if t.confirm is not None}


def in_scope(name: str, scope: str) -> bool:
    """A tool is in reach of a scope that is at least its tier."""
    tool = TOOLS.get(name)
    return tool is not None and scope in SCOPES and SCOPES.index(tool.tier) <= SCOPES.index(scope)


def offered(svc: HardwareService, scope: str) -> list[Tool]:
    return [t for t in TOOLS.values() if in_scope(t.name, scope) and (svc.fake or not t.fake_only)]


def summary(svc: HardwareService, name: str, arguments: dict[str, Any]) -> str:
    return TOOLS[name].confirm(svc, arguments)


def run(svc: HardwareService, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool. CommandError / ValueError propagate: the caller reports them as tool errors."""
    return TOOLS[name].handler(svc, arguments)


__all__ = ["CONFIRM", "SCOPES", "TOOLS", "CommandError", "in_scope", "offered", "run", "summary"]

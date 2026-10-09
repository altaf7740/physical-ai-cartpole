"""Raspberry Pi 4 controller: load policy.pt and run it at 60 Hz. No Isaac Lab dependency (torch + stdlib only).

    python run_policy.py --policy policy.pt --io policy_io.json              # starts at once, stops if it falls
    python run_policy.py --policy policy.pt --io policy_io.json --web        # + live page on http://<pi>:8000
    python run_policy.py --policy policy.pt --io policy_io.json --web --fake # same, with a simulated rig (no hardware)

The two hardware functions below are stubs. On the real rig, read_encoders() should get the latest motor and
pendulum encoder counts from the Pico 2 over serial, and write_pwm() should send the signed duty back. Everything
between them (unit conversion, velocity estimate, obs order and scaling, action clamp) is what the policy was
trained with.

With --web the controller starts IDLE (motor off) and is driven from the page:
    Start   run the policy (only if the rig is inside the envelope: |x| < cart limit, |theta| < max angle)
    Stop    motor off, back to idle
    E-STOP  motor off and latched until Reset
    Reset   clear E-STOP / a trip (with --fake it also stands the simulated pole up and holds it, like a hand,
            until Start)
Leaving the envelope while running trips the controller: motor off until Start is pressed again. The web page is
a convenience, not a safety device: keep a physical emergency stop on the rig.
"""

import argparse
import json
import math
import os
import time

import torch

# --- rig-specific conversion constants (fill in for your encoders) ---
# cart position comes from the motor encoder: the belt moves 40 mm per motor output revolution
MOTOR_COUNTS_PER_REV = 4096  # [counts] at the motor OUTPUT shaft (encoder CPR x gear ratio) (placeholder)
TRAVEL_PER_REV = 0.040  # [m] GT2 belt, 20T pulley
CART_CENTRE_COUNTS = 0  # motor count with the carriage at the rail centre (home it at start-up)
CART_SIGN = 1  # set to -1 if the motor count goes down when the cart moves toward +x (positive PWM)
# pendulum encoder on the carriage: it reads 0 with the pole HANGING DOWN, the policy wants 0 = upright
POLE_COUNTS_PER_REV = 2400  # [counts] (placeholder)
POLE_SIGN = 1  # set to -1 if the angle goes down when the pole leans toward +x


def read_encoders() -> tuple[int, int]:
    """STUB: return (motor_counts, pole_counts) from the Pico 2. Here: cart at centre, pole 2 deg past upright."""
    return CART_CENTRE_COUNTS, round((math.pi + math.radians(2.0)) / (2 * math.pi) * POLE_COUNTS_PER_REV)


def write_pwm(duty: float) -> None:
    """STUB: send signed duty in [-1, 1] to the Pico 2 (positive = cart toward +x)."""


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="policy.pt")
    parser.add_argument("--io", default="policy_io.json")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl+C)")
    parser.add_argument("--web", action="store_true", help="serve the live status page")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fake", action="store_true", help="simulated rig instead of read_encoders / write_pwm")
    args = parser.parse_args()

    torch.set_num_threads(1)
    policy = torch.jit.load(args.policy, map_location="cpu").eval()
    io = json.load(open(args.io))
    scale = torch.tensor(io["obs_scale"], dtype=torch.float32)
    hz = io["control_hz"]
    dt = 1.0 / hz
    x_lim, th_lim = io["cart_limit_m"], io["max_pole_angle_rad"]
    print("obs order:", io["obs_order"], "scale:", io["obs_scale"])
    with torch.inference_mode():  # TorchScript optimizes on the first calls; keep that out of the control loop
        for _ in range(5):
            policy(torch.zeros(1, 4))

    read, write, fake = read_encoders, write_pwm, None
    if args.fake:
        from fake_rig import FakeRig

        if "rig" not in io:
            raise SystemExit("--fake needs the 'rig' section in policy_io.json: re-export with `make play`")
        fake = FakeRig(io["rig"], MOTOR_COUNTS_PER_REV, TRAVEL_PER_REV, POLE_COUNTS_PER_REV, dt)
        read, write = fake.read_counts, fake.write_duty

    telemetry = None
    if args.web:
        import socket

        from web_server import Telemetry, serve

        info = {
            "mode": "fake" if fake else "hardware",
            "control_hz": hz,
            "cart_limit_m": x_lim,
            "max_pole_angle_rad": th_lim,
            "supply_voltage": io["supply_voltage"],
            "pole_length_m": io.get("rig", {}).get("pole_length", 0.5),
            "policy": os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.policy)))),
        }
        telemetry = Telemetry(info, history=int(10 * hz))
        serve(telemetry, args.port)
        print(f"web page: http://{socket.gethostname()}:{args.port}  (or http://localhost:{args.port})")

    state = "idle" if args.web else "running"
    msg, falls = "", 0
    prev = None
    next_t = time.perf_counter()
    last_t = next_t
    worst = 0.0
    k = 0
    try:
        while args.seconds <= 0 or k < args.seconds * hz:
            t0 = time.perf_counter()
            cart_c, pole_c = read()
            x = CART_SIGN * (cart_c - CART_CENTRE_COUNTS) / MOTOR_COUNTS_PER_REV * TRAVEL_PER_REV
            theta = wrap(POLE_SIGN * pole_c / POLE_COUNTS_PER_REV * 2 * math.pi - math.pi)  # 180 deg: 0 = upright
            # velocities by finite difference at the control rate (first step: 0)
            x_dot, theta_dot = (0.0, 0.0) if prev is None else ((x - prev[0]) / dt, wrap(theta - prev[1]) / dt)
            prev = (x, theta)
            inside = abs(x) < x_lim and abs(theta) < th_lim

            # commands from the page
            for cmd in telemetry.pop_commands() if telemetry else []:
                if cmd == "estop":
                    state, msg = "estop", "E-STOP pressed: motor off until Reset"
                elif cmd == "stop" and state != "estop":
                    state, msg = "idle", "stopped"
                elif cmd == "reset":
                    if fake:
                        fake.reset()
                        prev = None
                    state, msg = "idle", "reset" + (" (simulated pole held upright: press Start)" if fake else "")
                elif cmd == "start":
                    if state == "estop":
                        msg = "E-STOP is latched: press Reset first"
                    elif not inside:
                        msg = "cannot start: hold the pole upright and the cart inside the rail limits"
                    else:
                        state, msg = "running", "running"
                        if fake:
                            fake.release()

            # safety: motor off outside the trained envelope
            if state == "running" and not inside:
                falls += 1
                state, msg = "tripped", f"tripped: x={x:+.3f} m, theta={theta:+.3f} rad (motor off)"
                print(msg)
                if not args.web:
                    break

            action = 0.0
            if state == "running":
                obs = torch.tensor([[x, x_dot, theta, theta_dot]], dtype=torch.float32) * scale
                with torch.inference_mode():
                    action = float(policy(obs)[0, 0].clamp(-1.0, 1.0))
            write(action)

            compute = time.perf_counter() - t0
            worst = max(worst, compute)
            if telemetry:
                telemetry.add(
                    {
                        "t": round(k * dt, 4),
                        "x": round(x, 5),
                        "x_dot": round(x_dot, 4),
                        "theta": round(theta, 5),
                        "theta_dot": round(theta_dot, 4),
                        "action": round(action, 4),
                        "voltage": round(action * io["supply_voltage"], 3),
                        "state": state,
                        "msg": msg,
                        "falls": falls,
                        "compute_ms": round(compute * 1e3, 3),
                        "period_ms": round((t0 - last_t) * 1e3, 2),
                    }
                )
            last_t = t0
            if not args.web and k % int(hz) == 0:
                print(f"step {k:5d}  x={x:+.3f} m  theta={theta:+.3f} rad  action={action:+.3f}")
            if fake:
                fake.advance()
            k += 1
            next_t += dt
            time.sleep(max(0.0, next_t - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        write(0.0)
    print(f"worst read+infer+write time: {worst * 1e3:.2f} ms (budget {dt * 1e3:.1f} ms), falls: {falls}")


if __name__ == "__main__":
    main()

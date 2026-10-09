"""Raspberry Pi 4 inference stub: load policy.pt and run it at 60 Hz. No Isaac Lab dependency (torch only).

    python run_policy.py --policy policy.pt --io policy_io.json [--seconds 5]

The two hardware functions below are stubs. On the real rig, read_encoders() should get the latest motor and
pendulum encoder counts from the Pico 2 over serial, and write_pwm() should send the signed duty back. Everything
between them (unit conversion, velocity estimate, obs order and scaling, action clamp) is what the policy was
trained with.
"""

import argparse
import json
import math
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="policy.pt")
    parser.add_argument("--io", default="policy_io.json")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    policy = torch.jit.load(args.policy, map_location="cpu").eval()
    io = json.load(open(args.io))
    scale = torch.tensor(io["obs_scale"], dtype=torch.float32)
    dt = 1.0 / io["control_hz"]
    print("obs order:", io["obs_order"], "scale:", io["obs_scale"])
    with torch.inference_mode():  # TorchScript optimizes on the first calls; keep that out of the control loop
        for _ in range(5):
            policy(torch.zeros(1, 4))

    prev = None
    next_t = time.perf_counter()
    worst = 0.0
    for k in range(int(args.seconds * io["control_hz"])):
        t0 = time.perf_counter()
        cart_c, pole_c = read_encoders()
        x = CART_SIGN * (cart_c - CART_CENTRE_COUNTS) / MOTOR_COUNTS_PER_REV * TRAVEL_PER_REV
        theta = POLE_SIGN * pole_c / POLE_COUNTS_PER_REV * 2 * math.pi - math.pi  # 180 deg offset: 0 = upright
        theta = math.atan2(math.sin(theta), math.cos(theta))  # wrap to [-pi, pi]
        # velocities by finite difference at the control rate (first step: 0)
        x_dot, theta_dot = (0.0, 0.0) if prev is None else ((x - prev[0]) / dt, (theta - prev[1]) / dt)
        prev = (x, theta)

        # safety: stop the motor outside the trained envelope
        if abs(x) > io["cart_limit_m"] or abs(theta) > io["max_pole_angle_rad"]:
            write_pwm(0.0)
            print(f"step {k}: outside envelope (x={x:.3f} m, theta={theta:.3f} rad), motor off")
            break

        obs = torch.tensor([[x, x_dot, theta, theta_dot]], dtype=torch.float32) * scale
        with torch.inference_mode():
            a = float(policy(obs)[0, 0].clamp(-1.0, 1.0))
        write_pwm(a)

        worst = max(worst, time.perf_counter() - t0)
        if k % int(io["control_hz"]) == 0:
            print(f"step {k:4d}  obs={[round(v, 3) for v in obs[0].tolist()]}  action={a:+.3f}")
        next_t += dt
        time.sleep(max(0.0, next_t - time.perf_counter()))

    write_pwm(0.0)
    print(f"worst read+infer+write time: {worst * 1e3:.2f} ms (budget {dt * 1e3:.1f} ms)")


if __name__ == "__main__":
    main()

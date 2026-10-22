"""Top-level routines: serial check, sensor monitor, dry run and live balancing."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator

from config import Config, effective_max_velocity
from interface import ActionConverter, build_observation
from logger import RunLogger
from pico_link import PicoLink
from policy import Policy, load_policy
from safety import Mode, SafetyMonitor
from state import CartPoleState, StateEstimator


def fixed_rate(rate_hz: float) -> Iterator[float]:
    """Yield once per period with the measured dt, using absolute deadlines (no drift)."""
    period = 1.0 / rate_hz
    next_tick = time.perf_counter()
    last = next_tick - period
    while True:
        now = time.perf_counter()
        yield now - last if now > last else period
        last = now
        next_tick += period
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.perf_counter()  # overran: resync instead of bursting


class BalanceController:
    """One control tick: safety gate -> policy -> step rate. Shared by the CLI and the UI."""

    def __init__(self, cfg: Config, policy: Policy):
        self.obs_order = cfg.policy.obs_order
        self.policy = policy
        self.safety = SafetyMonitor(cfg.safety)
        self.converter = ActionConverter(cfg, effective_max_velocity(cfg))
        self.step_rate = 0
        self.action = 0.0

    def reset(self) -> None:
        self.safety.reset()
        self.converter.reset()
        self.step_rate = 0
        self.action = 0.0

    def step(self, state: CartPoleState | None, limit_hit: bool, dt: float) -> Mode:
        """Update `step_rate` and `action` from the newest state (None = missed reply)."""
        mode = self.safety.missed_read() if state is None else self.safety.update(state, limit_hit)
        self.action = 0.0
        if mode is Mode.BALANCING and state is not None:
            self.action = float(self.policy.act(build_observation(state, self.obs_order))[0])
            self.step_rate = self.converter.command(self.action, dt)
        elif mode is Mode.BALANCING:
            pass  # a dropped reply: hold the last command until the safety monitor faults
        else:
            self.converter.reset()
            self.step_rate = 0
        return mode


def check_serial(cfg: Config, link: PicoLink) -> None:
    """Reset the counters and print a few raw telemetry replies."""
    with link as pico:
        pico.reset_counters()
        for _ in range(5):
            print("reply:", pico.exchange(0))
            time.sleep(0.1)


def monitor_sensors(cfg: Config, link: PicoLink, rate_hz: float = 20.0) -> None:
    """Motor held at 0. Move the cart and pole by hand to verify signs, units and zero."""
    estimator = StateEstimator(cfg.hardware, cfg.estimator)
    with link as pico:
        pico.reset_counters()
        print("Motor off. Move cart/pole by hand. Ctrl+C to stop.")
        for dt in fixed_rate(rate_hz):
            tel = pico.exchange(0)
            if tel is None:
                print("no telemetry")
                continue
            s = estimator.update(tel, dt)
            print(
                f"x={s.cart_pos:+.3f} m  dx={s.cart_vel:+.3f} m/s  "
                f"theta={math.degrees(s.pole_pos):+7.2f} deg  dtheta={s.pole_vel:+6.2f} rad/s  "
                f"limit={int(tel.limit_hit)}"
            )


def run_policy(cfg: Config, link: PicoLink, dry_run: bool) -> None:
    """Closed-loop balancing. With dry_run the motor stays at 0 and commands are only printed/logged."""
    controller = BalanceController(cfg, load_policy(cfg.policy))
    estimator = StateEstimator(cfg.hardware, cfg.estimator)

    print(f"policy: {cfg.policy.model_path} @ {cfg.policy.rate_hz:g} Hz, action={cfg.policy.action_type}")
    print(f"cart speed limit: {controller.converter.max_velocity:.3f} m/s"
          f"{'  [DRY RUN — motor stays off]' if dry_run else ''}")
    print(f"Hold the pole upright (<{cfg.safety.engage_angle_deg:g} deg) to engage. Ctrl+C to stop.")

    last_mode = controller.safety.mode
    with link as pico, RunLogger(
        cfg.logging.directory, cfg.logging.enabled, "dry-run" if dry_run else "run"
    ) as log:
        pico.reset_counters()
        t0 = time.perf_counter()
        for dt in fixed_rate(cfg.policy.rate_hz):
            # The command computed last tick goes out with this tick's telemetry
            # request, so the real system has one control period of action delay.
            tel = pico.exchange(0 if dry_run else controller.step_rate)
            state = estimator.update(tel, dt) if tel is not None else None
            mode = controller.step(state, tel is not None and tel.limit_hit, dt)

            if mode is not last_mode:
                print(f"[{time.perf_counter() - t0:7.2f}s] {last_mode.value} -> {mode.value}: "
                      f"{controller.safety.reason}")
                last_mode = mode
            if state is not None:
                log.log(t=time.perf_counter() - t0, dt=dt, action=controller.action,
                        step_rate=controller.step_rate, mode=mode.value, **state.as_dict())
        # unreachable: the loop ends via KeyboardInterrupt; PicoLink.close() stops the motor

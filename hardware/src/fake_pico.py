"""Simulated Pico + cartpole (`--fake`) for testing the UI and CLI without hardware.

Same interface as PicoLink. The cart follows the commanded step rate exactly,
like a stepper that never stalls; the pole is a damped point-mass pendulum.
"""

from __future__ import annotations

import math
import time

from config import Config
from pico_link import Telemetry

GRAVITY = 9.81
POLE_LENGTH_M = 0.3
POLE_DAMPING = 0.05  # 1/s
SUBSTEP_S = 0.001
END_STOP_M = 0.3  # simulated limit switches, this far either side of the reset position


class FakePico:
    def __init__(self, cfg: Config):
        self.hw = cfg.hardware
        self.hard_limit_m = END_STOP_M
        # Start where the config says the pole is at reset, so readings are consistent.
        self._theta = math.pi if self.hw.angle_zero == "down" else 0.0  # rad, 0 = upright, unwrapped
        self._omega = 0.0
        self._steps = 0.0
        self._rate = 0
        self._theta_at_reset = self._theta
        self._steps_at_reset = 0.0
        self._last = time.perf_counter()

    def open(self) -> None:
        self._last = time.perf_counter()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> FakePico:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset_counters(self) -> None:
        self._advance()
        self._theta_at_reset = self._theta
        self._steps_at_reset = self._steps

    def stop(self) -> None:
        self._set_rate(0)

    def exchange(self, step_rate: int) -> Telemetry:
        self._advance()
        self._set_rate(max(-self.hw.max_step_rate, min(self.hw.max_step_rate, int(step_rate))))
        return self._telemetry()

    def place_upright(self, angle: float = 0.05) -> None:
        """Simulates a hand lifting the pole to just off upright and letting go."""
        self._advance()
        self._theta = angle + 2 * math.pi * round(self._theta / (2 * math.pi))
        self._omega = 0.0

    def _x(self) -> float:
        return self.hw.cart_sign * self._steps / self.hw.steps_per_meter

    def _set_rate(self, rate: int) -> None:
        # A stepper changes speed instantly, which kicks the pole: d(omega) = -dv cos(theta) / L.
        dv = self.hw.cart_sign * (rate - self._rate) / self.hw.steps_per_meter
        self._omega -= dv * math.cos(self._theta) / POLE_LENGTH_M
        self._rate = rate

    def _advance(self) -> None:
        now = time.perf_counter()
        remaining, self._last = now - self._last, now
        while remaining > 0:
            h = min(SUBSTEP_S, remaining)
            remaining -= h
            self._omega += (GRAVITY / POLE_LENGTH_M * math.sin(self._theta) - POLE_DAMPING * self._omega) * h
            self._theta += self._omega * h
            self._steps += self._rate * h
            x = self._x()
            if abs(x) > self.hard_limit_m:  # hit the end stop
                self._steps = self.hw.cart_sign * math.copysign(self.hard_limit_m, x) * self.hw.steps_per_meter
                self._set_rate(0)

    def _telemetry(self) -> Telemetry:
        ticks = self.hw.encoder_sign * (self._theta - self._theta_at_reset) / (2 * math.pi) * self.hw.encoder_cpr
        return Telemetry(
            encoder_ticks=round(ticks),
            cart_steps=round(self._steps - self._steps_at_reset),
            limit_hit=abs(self._x()) >= self.hard_limit_m - 1e-9,
        )

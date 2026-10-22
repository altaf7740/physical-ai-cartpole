"""Turn raw Pico counts into the cartpole state in the simulator's frame.

Conventions (must match Isaac Lab): pole angle 0 rad = upright, wrapped to
[-pi, pi); cart position 0 m = where the counters were reset.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from config import EstimatorConfig, HardwareConfig
from pico_link import Telemetry


@dataclass(frozen=True)
class CartPoleState:
    cart_pos: float  # m
    pole_pos: float  # rad, 0 = upright
    cart_vel: float  # m/s
    pole_vel: float  # rad/s

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ticks_to_angle(ticks: int, cpr: int, sign: int = 1, angle_zero: str = "down") -> float:
    """Encoder ticks -> pole angle with 0 = upright.

    angle_zero says where the pole was when the counters were reset: if it was
    hanging "down", upright is half a turn away and we shift by pi.
    """
    angle = sign * ticks / cpr * 2.0 * math.pi
    if angle_zero == "down":
        angle -= math.pi
    return wrap_angle(angle)


def steps_to_position(steps: int, steps_per_meter: float, sign: int = 1) -> float:
    return sign * steps / steps_per_meter


class StateEstimator:
    """Positions from counts, velocities from low-passed finite differences."""

    def __init__(self, hw: HardwareConfig, est: EstimatorConfig):
        self.hw = hw
        self.alpha = est.velocity_alpha
        self.reset()

    def reset(self) -> None:
        self._prev_pos: float | None = None
        self._prev_angle: float | None = None
        self._cart_vel = 0.0
        self._pole_vel = 0.0

    def update(self, tel: Telemetry, dt: float) -> CartPoleState:
        pos = steps_to_position(tel.cart_steps, self.hw.steps_per_meter, self.hw.cart_sign)
        angle = ticks_to_angle(tel.encoder_ticks, self.hw.encoder_cpr, self.hw.encoder_sign, self.hw.angle_zero)

        if self._prev_pos is not None and dt > 0:
            raw_cart_vel = (pos - self._prev_pos) / dt
            # Wrap the difference so crossing +/-pi doesn't look like a huge spin.
            raw_pole_vel = wrap_angle(angle - self._prev_angle) / dt
            self._cart_vel += self.alpha * (raw_cart_vel - self._cart_vel)
            self._pole_vel += self.alpha * (raw_pole_vel - self._pole_vel)

        self._prev_pos, self._prev_angle = pos, angle
        return CartPoleState(pos, angle, self._cart_vel, self._pole_vel)

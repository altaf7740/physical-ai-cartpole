"""The sim <-> real contract: state -> observation, policy action -> step rate."""

from __future__ import annotations

import numpy as np

from config import Config
from state import CartPoleState


def build_observation(state: CartPoleState, obs_order: tuple[str, ...]) -> np.ndarray:
    values = state.as_dict()
    return np.array([values[k] for k in obs_order], dtype=np.float32)


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ActionConverter:
    """Map the raw policy output to a signed Pico step rate.

    velocity     : action * scale is the cart velocity (m/s)
    acceleration : action * scale is integrated into a velocity (m/s^2)
    force        : action * scale / force_mass_kg, then integrated (approximation)
    """

    def __init__(self, cfg: Config, max_velocity: float):
        self.pol = cfg.policy
        self.hw = cfg.hardware
        self.max_velocity = max_velocity
        self.velocity = 0.0

    def reset(self) -> None:
        self.velocity = 0.0

    def to_velocity(self, action: float, dt: float) -> float:
        a = clamp(float(action), self.pol.action_clip) * self.pol.action_scale
        if self.pol.action_type == "velocity":
            self.velocity = a
        else:
            accel = a if self.pol.action_type == "acceleration" else a / self.pol.force_mass_kg
            self.velocity += accel * dt
        self.velocity = clamp(self.velocity, self.max_velocity)
        return self.velocity

    def velocity_to_step_rate(self, velocity: float) -> int:
        rate = round(self.hw.cart_sign * velocity * self.hw.steps_per_meter)
        return int(clamp(rate, self.hw.max_step_rate))

    def command(self, action: float, dt: float) -> int:
        return self.velocity_to_step_rate(self.to_velocity(action, dt))

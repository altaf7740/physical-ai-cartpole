"""Decide whether the policy is allowed to drive the motor."""

from __future__ import annotations

import math
from enum import Enum

from config import SafetyConfig
from state import CartPoleState


class Mode(Enum):
    IDLE = "idle"            # motor stopped, waiting for the pole to be held upright
    BALANCING = "balancing"  # policy drives the cart
    FAULT = "fault"          # latched until restart: limit switch, track limit, lost link


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.engage = math.radians(cfg.engage_angle_deg)
        self.disengage = math.radians(cfg.disengage_angle_deg)
        self.reset()

    def reset(self) -> None:
        self.mode = Mode.IDLE
        self.reason = ""
        self._missed = 0

    def _fault(self, reason: str) -> Mode:
        self.mode, self.reason = Mode.FAULT, reason
        return self.mode

    def missed_read(self) -> Mode:
        """Call when a tick gets no valid telemetry."""
        self._missed += 1
        if self._missed > self.cfg.max_missed_reads and self.mode is not Mode.FAULT:
            return self._fault(f"lost telemetry for {self._missed} ticks")
        return self.mode

    def update(self, state: CartPoleState, limit_hit: bool) -> Mode:
        self._missed = 0
        if self.mode is Mode.FAULT:
            return self.mode
        if limit_hit:
            return self._fault("limit switch hit")
        if abs(state.cart_pos) > self.cfg.track_limit_m:
            return self._fault(f"cart at {state.cart_pos:+.3f} m beyond soft limit")

        angle = abs(state.pole_pos)
        if self.mode is Mode.IDLE and angle < self.engage:
            self.mode, self.reason = Mode.BALANCING, "pole upright"
        elif self.mode is Mode.BALANCING and angle > self.disengage:
            self.mode, self.reason = Mode.IDLE, "pole fell"
        return self.mode

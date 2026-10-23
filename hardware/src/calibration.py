"""Track calibration and homing on the limit switches; both park the cart in the middle.

Calibrate: left switch, right switch, twice, measuring the travel between them.
Home: left switch only, then half the measured travel; needs a saved calibration.

Works in raw Pico steps (negative steps move toward the left switch, see pico_main.py), so it
does not depend on cart_sign or steps_per_meter. The travel in steps between the two switch
trigger points is the real track length; dividing by steps_per_meter gives metres, so comparing
that with a tape measure checks steps_per_meter.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LEFT, RIGHT = 1, 2  # limit bitmask bits from the Pico
LEG_TIMEOUT_S = 30.0  # minimum; a switch that never triggers aborts the run
CENTRE_TOLERANCE_STEPS = 40
CENTRE_GAIN = 4.0  # step rate per step of error while parking (1/s)
CENTRE_MIN_RATE = 500
SETTLE_S = 0.3


@dataclass(frozen=True)
class TrackCalibration:
    travel_steps: int  # between the two switch trigger points
    spread_steps: int  # max - min travel over the passes (repeatability; lost steps show up here)
    passes: tuple[int, ...]
    centre_steps: int  # raw Pico steps at the middle (only valid until the Pico counters reset)
    measured_at: float

    def half_length_m(self, steps_per_meter: float) -> float:
        return self.travel_steps / 2 / steps_per_meter


class TrackCalibrator:
    """One calibration or homing run. Call step() every control tick; it returns the step rate to send.

    With travel_steps (a saved calibration) it only homes: left switch, then half the travel.
    Without it, it measures the track with `passes` left-right passes and sets `result`.
    """

    def __init__(self, rate: int, passes: int = 2, leg_timeout_s: float = LEG_TIMEOUT_S,
                 travel_steps: int | None = None):
        self.rate = abs(int(rate))
        self.leg_timeout_s = leg_timeout_s
        self.travel_steps = travel_steps
        self.legs = [-1] if travel_steps else [-1, 1] * passes  # direction of each leg, left first
        self.leg = 0
        self.hits: dict[int, list[int]] = {-1: [], 1: []}
        self.phase = "homing"  # -> "centring" -> "settling" -> "done" | "failed"
        self.elapsed = 0.0
        self.target: int | None = None
        self.error = ""
        self.result: TrackCalibration | None = None

    @property
    def finished(self) -> bool:
        return self.phase in ("done", "failed")

    def describe(self) -> str:
        if self.phase == "homing":
            if self.travel_steps:
                return "homing: to the left switch"
            side = "left" if self.legs[self.leg] < 0 else "right"
            return f"calibrating, pass {self.leg // 2 + 1}/{len(self.legs) // 2}: to the {side} switch"
        if self.phase in ("centring", "settling"):
            return "parking in the middle"
        if self.phase == "failed":
            return self.error
        return "done"

    def _fail(self, msg: str) -> int:
        self.phase, self.error = "failed", msg
        return 0

    def step(self, steps: int, limits: int, dt: float) -> int:
        if self.finished:
            return 0
        self.elapsed += dt
        if self.phase == "homing":
            d = self.legs[self.leg]
            if limits & (LEFT if d < 0 else RIGHT):
                self.hits[d].append(steps)
                self.leg += 1
                self.elapsed = 0.0
                if self.leg == len(self.legs):
                    self._start_centring()
                return 0
            if self.elapsed > self.leg_timeout_s:
                side = "left" if d < 0 else "right"
                return self._fail(f"no {side} limit switch within {self.leg_timeout_s:.0f} s; check it and the direction")
            return d * self.rate
        if self.phase == "centring":
            err = self.target - steps
            if abs(err) <= CENTRE_TOLERANCE_STEPS:
                self.phase, self.elapsed = "settling", 0.0
                return 0
            if self.elapsed > self.leg_timeout_s:
                return self._fail("could not reach the middle of the track")
            rate = max(CENTRE_MIN_RATE, min(self.rate, abs(err) * CENTRE_GAIN))
            return int(rate) if err > 0 else -int(rate)
        if self.phase == "settling" and self.elapsed >= SETTLE_S:
            self.phase = "done"
        return 0

    def _start_centring(self) -> None:
        self.phase, self.elapsed = "centring", 0.0
        if self.travel_steps:
            self.target = self.hits[-1][-1] + round(self.travel_steps / 2)
            return
        travels = [r - l for l, r in zip(self.hits[-1], self.hits[1])]
        left, right = self.hits[-1][-1], self.hits[1][-1]
        self.target = round((left + right) / 2)
        self.result = TrackCalibration(
            travel_steps=round(sum(travels) / len(travels)),
            spread_steps=max(travels) - min(travels),
            passes=tuple(travels),
            centre_steps=self.target,
            measured_at=time.time(),
        )


def save_calibration(path: Path, cal: TrackCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cal), indent=2) + "\n")


def load_calibration(path: Path) -> TrackCalibration | None:
    try:
        raw = json.loads(path.read_text())
        return TrackCalibration(**{**raw, "passes": tuple(raw["passes"])})
    except (OSError, ValueError, TypeError, KeyError):
        return None

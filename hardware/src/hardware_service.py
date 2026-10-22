"""Background control loop behind the web UI.

One thread owns the Pico link and ticks at the policy rate. Request threads only
set flags and read samples, so serial I/O never happens on a web request.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from config import Config, effective_max_velocity
from controller import BalanceController, fixed_rate
from interface import ActionConverter, clamp
from logger import RunLogger
from pico_link import Telemetry
from policy import Policy, load_policy
from state import CartPoleState, StateEstimator

MANUAL, DRY_RUN, LIVE = "manual", "dry-run", "live"


class CommandError(Exception):
    pass


def _r(value: float | None, digits: int = 5) -> float | None:
    return None if value is None else round(value, digits)


class HardwareService:
    def __init__(self, cfg: Config, link_factory: Callable[[], object], fake: bool = False):
        self.cfg = cfg
        self.fake = fake
        self.max_velocity = effective_max_velocity(cfg)
        self._link_factory = link_factory
        self._jog_converter = ActionConverter(cfg, self.max_velocity)
        self._period = 1.0 / cfg.policy.rate_hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.perf_counter()
        self._samples: deque[dict] = deque(maxlen=int(cfg.policy.rate_hz * 60))
        self._seq = 0
        self._link_state, self._error = "disconnected", ""
        self._control = MANUAL
        self._policy: Policy | None = None
        self._controller: BalanceController | None = None
        self._jog_velocity, self._jog_deadline = 0.0, 0.0
        self._pending: set[str] = set()
        self._recorder: RunLogger | None = None
        self._rate_hz, self._late_ticks, self._missed = 0.0, 0, 0

    # ---- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._link_state, self._error = "connecting", ""
            self._thread = threading.Thread(target=self._run, name="pico-loop", daemon=True)
            self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def close(self) -> None:
        self.disconnect()
        self.set_recording(False)

    # ---- commands (request threads) ----------------------------------------

    def _require_connected(self) -> None:
        if self._link_state != "connected":
            raise CommandError(f"Pico is {self._link_state}")

    def _back_to_manual(self) -> None:
        self._control = MANUAL
        self._controller = None
        self._jog_velocity = 0.0

    def estop(self) -> None:
        with self._lock:
            self._back_to_manual()

    stop_policy = estop

    def reset_counters(self) -> None:
        with self._lock:
            self._require_connected()
            self._pending.add("reset")

    def jog(self, velocity: float) -> None:
        with self._lock:
            self._require_connected()
            if self._control != MANUAL:
                raise CommandError("stop the policy before jogging")
            self._jog_velocity = clamp(velocity, self.max_velocity)
            self._jog_deadline = time.monotonic() + self.cfg.ui.jog_timeout_s

    def start_policy(self, live: bool) -> None:
        with self._lock:
            self._require_connected()
        if self._policy is None:
            self._policy = load_policy(self.cfg.policy)  # slow-ish; keep it outside the lock
        with self._lock:
            self._controller = BalanceController(self.cfg, self._policy)
            self._control = LIVE if live else DRY_RUN
            self._jog_velocity = 0.0

    def clear_fault(self) -> None:
        with self._lock:
            if self._controller is not None:
                self._controller.reset()

    def set_recording(self, on: bool) -> str | None:
        with self._lock:
            if on and self._recorder is None:
                self._recorder = RunLogger(self.cfg.logging.directory, True, "ui")
            elif not on and self._recorder is not None:
                self._recorder.close()
                self._recorder = None
            return str(self._recorder.path) if self._recorder else None

    def lift_pole(self) -> None:
        with self._lock:
            if not self.fake:
                raise CommandError("only available with --fake")
            self._require_connected()
            self._pending.add("lift")

    # ---- reads (request threads) -------------------------------------------

    def info(self) -> dict:
        cfg = self.cfg
        return {
            "fake": self.fake,
            "rate_hz": cfg.policy.rate_hz,
            "engage_deg": cfg.safety.engage_angle_deg,
            "disengage_deg": cfg.safety.disengage_angle_deg,
            "max_velocity": self.max_velocity,
            "max_step_rate": cfg.hardware.max_step_rate,
            "angle_zero": cfg.hardware.angle_zero,
            "model_path": cfg.policy.model_path,
            "action_type": cfg.policy.action_type,
            "obs_order": list(cfg.policy.obs_order),
            "jog_timeout_s": cfg.ui.jog_timeout_s,
        }

    def status(self) -> dict:
        with self._lock:
            return {
                "link": self._link_state,
                "error": self._error,
                "control": self._control,
                "recording": str(self._recorder.path) if self._recorder else None,
                "rate_hz": round(self._rate_hz, 1),
                "late_ticks": self._late_ticks,
                "missed_reads": self._missed,
                "model_exists": Path(self.cfg.policy.model_path).exists(),
            }

    def samples_since(self, seq: int, max_count: int) -> tuple[list[dict], int]:
        with self._lock:
            new = []
            for sample in reversed(self._samples):
                if sample["seq"] <= seq:
                    break
                new.append(sample)
            new.reverse()
            return new[-max_count:], self._seq

    # ---- control loop (loop thread) ----------------------------------------

    def _run(self) -> None:
        link = self._link_factory()
        try:
            link.open()
            link.reset_counters()
        except Exception as e:  # port missing, permission denied, ...
            self._finish(link, f"{type(e).__name__}: {e}")
            return
        with self._lock:
            self._link_state = "connected"

        estimator = StateEstimator(self.cfg.hardware, self.cfg.estimator)
        step_rate, error = 0, ""
        try:
            for dt in fixed_rate(self.cfg.policy.rate_hz):
                if self._stop.is_set():
                    break
                with self._lock:
                    pending, self._pending = self._pending, set()
                if "lift" in pending:
                    link.place_upright()
                    estimator.reset()  # the angle teleports; don't read it as a huge spin
                if "reset" in pending:
                    link.reset_counters()
                    estimator.reset()
                    with self._lock:
                        if self._controller is not None:
                            self._controller.reset()
                tel = link.exchange(step_rate)
                state = estimator.update(tel, dt) if tel is not None else None
                with self._lock:
                    step_rate = self._tick(tel, state, dt)
        except Exception as e:  # e.g. Pico unplugged mid-run
            error = f"{type(e).__name__}: {e}"
        self._finish(link, error)

    def _finish(self, link, error: str) -> None:
        try:
            link.close()  # commands 0 if the port still works
        except Exception:
            pass
        with self._lock:
            self._back_to_manual()
            self._link_state = "error" if error else "disconnected"
            self._error = error

    def _manual_rate(self, state: CartPoleState | None) -> int:
        # No soft limit: the Pico refuses steps toward a pressed limit switch and allows moving away.
        v = self._jog_velocity
        if v == 0.0 or time.monotonic() > self._jog_deadline or state is None:
            return 0
        return self._jog_converter.velocity_to_step_rate(v)

    def _tick(self, tel: Telemetry | None, state: CartPoleState | None, dt: float) -> int:
        """Decide the next step rate and record a sample. Called with the lock held."""
        limit_hit = tel is not None and tel.limit_hit
        safety, reason, action, policy_rate = "", "", 0.0, None
        if self._controller is None:
            send = self._manual_rate(state)
        else:
            mode = self._controller.step(state, limit_hit, dt)
            safety, reason = mode.value, self._controller.safety.reason
            action, policy_rate = self._controller.action, self._controller.step_rate
            send = policy_rate if self._control == LIVE else 0

        if dt > 0:
            rate = 1.0 / dt
            self._rate_hz = rate if self._rate_hz == 0 else self._rate_hz + 0.05 * (rate - self._rate_hz)
        if dt > 1.5 * self._period:
            self._late_ticks += 1
        if tel is None:
            self._missed += 1

        t = time.perf_counter() - self._t0
        self._seq += 1
        self._samples.append({
            "seq": self._seq,
            "t": round(t, 4),
            "ok": tel is not None,
            "ticks": tel.encoder_ticks if tel else None,
            "steps": tel.cart_steps if tel else None,
            "limit": limit_hit,
            "x": _r(state.cart_pos) if state else None,
            "theta": _r(state.pole_pos) if state else None,
            "dx": _r(state.cart_vel) if state else None,
            "dtheta": _r(state.pole_vel) if state else None,
            "cmd": send,
            "policy_rate": policy_rate,
            "action": _r(action),
            "safety": safety,
            "reason": reason,
        })
        if self._recorder is not None and state is not None:
            self._recorder.log(t=t, dt=dt, action=action, step_rate=send,
                               mode=safety or self._control, **state.as_dict())
        return send

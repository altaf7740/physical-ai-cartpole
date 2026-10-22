"""Serial link to the Pico.

Protocol (newline-terminated ASCII, one request -> one reply):
    Pi  -> Pico : "R"            reset encoder and step counters
    Pi  -> Pico : "<step_rate>"  signed half-steps per second, e.g. "-1250"
    Pico -> Pi  : "<encoder_ticks>,<cart_steps>,<limit_hit>"
                  limit_hit: 0 none, 1 left, 2 right, 3 both (any non-zero counts as hit)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from config import SerialConfig


@dataclass(frozen=True)
class Telemetry:
    encoder_ticks: int
    cart_steps: int
    limit_hit: bool


def parse_telemetry(line: str) -> Telemetry | None:
    """Parse one Pico reply; returns None for empty or malformed lines."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        ticks, steps, lim = (int(p) for p in parts)
    except ValueError:
        return None
    return Telemetry(encoder_ticks=ticks, cart_steps=steps, limit_hit=bool(lim))


def format_step_rate(step_rate: int) -> bytes:
    return f"{int(step_rate)}\n".encode()


class PicoLink:
    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self._ser = None

    def open(self) -> None:
        import serial  # imported lazily so the rest of the package works without pyserial

        # exclusive: a second cartpole process (UI + CLI) fails here instead of fighting over the motor.
        self._ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_s, exclusive=True)
        # Opening the port can reboot/re-enumerate the Pico; give it time to start.
        time.sleep(self.cfg.startup_delay_s)
        self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser is None:
            return
        try:
            self.stop()
        finally:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> PicoLink:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _require_open(self):
        if self._ser is None:
            raise RuntimeError("PicoLink is not open")
        return self._ser

    def reset_counters(self) -> None:
        ser = self._require_open()
        ser.write(b"R\n")
        time.sleep(0.05)
        ser.reset_input_buffer()

    def exchange(self, step_rate: int) -> Telemetry | None:
        """Send a step-rate command and return the Pico's telemetry reply."""
        ser = self._require_open()
        # Drop late replies from earlier ticks so we always act on fresh data.
        ser.reset_input_buffer()
        ser.write(format_step_rate(step_rate))
        return parse_telemetry(ser.readline().decode(errors="replace"))

    def stop(self) -> None:
        ser = self._require_open()
        ser.write(format_step_rate(0))
        ser.flush()

"""Per-tick CSV logs, for comparing real rollouts against simulation."""

from __future__ import annotations

import csv
import time
from pathlib import Path

FIELDS = ("t", "dt", "cart_pos", "pole_pos", "cart_vel", "pole_vel", "action", "step_rate", "mode")


class RunLogger:
    def __init__(self, directory: str, enabled: bool, tag: str):
        self._file = None
        self._writer = None
        self.path: Path | None = None
        if not enabled:
            return
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}.csv"
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FIELDS)

    def log(self, **row) -> None:
        if self._writer is not None:
            self._writer.writerow([row.get(k, "") for k in FIELDS])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

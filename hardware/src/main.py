"""CLI: python src/main.py [--fake] --config config.toml {serial,sensors,dry-run,run,ui}"""

from __future__ import annotations

import argparse
import signal
import sys

from config import Config, ConfigError, load_config
from pico_link import PicoLink
import controller


def _exit_on_sigterm(*_):
    # SystemExit unwinds through the `with PicoLink(...)` blocks, which stop the motor.
    raise SystemExit(0)


def make_link(cfg: Config, fake: bool):
    if fake:
        from fake_pico import FakePico

        return FakePico(cfg)
    return PicoLink(cfg.serial)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cartpole", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--fake", action="store_true", help="use a simulated Pico instead of the serial port")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serial", help="reset counters and print raw Pico replies")
    sub.add_parser("sensors", help="print the live state with the motor off")
    sub.add_parser("dry-run", help="run the policy on live sensors, motor off")
    sub.add_parser("run", help="balance the pole with the policy")
    sub.add_parser("ui", help="web UI: live data, jog, policy control")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        cfg = load_config(args.config)
        if args.command == "serial":
            controller.check_serial(cfg, make_link(cfg, args.fake))
        elif args.command == "sensors":
            controller.monitor_sensors(cfg, make_link(cfg, args.fake))
        elif args.command == "ui":
            from hardware_service import HardwareService
            from ui_server import serve

            serve(cfg, HardwareService(cfg, lambda: make_link(cfg, args.fake), fake=args.fake))
        else:
            controller.run_policy(cfg, make_link(cfg, args.fake), dry_run=args.command == "dry-run")
    except KeyboardInterrupt:
        print("\nStopped — motor commanded to 0.")
    except (ConfigError, FileNotFoundError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

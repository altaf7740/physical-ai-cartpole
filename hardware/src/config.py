"""Load and validate config.toml into typed, immutable sections."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

# Every observation the hardware can provide. `policy.obs_order` picks and
# orders these to match the observation vector used during training.
OBS_KEYS = ("cart_pos", "pole_pos", "cart_vel", "pole_vel")
ACTION_TYPES = ("velocity", "acceleration", "force")
ANGLE_ZERO = ("down", "upright")
BACKENDS = ("auto", "onnx", "numpy")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SerialConfig:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    timeout_s: float = 0.01
    startup_delay_s: float = 2.0


@dataclass(frozen=True)
class HardwareConfig:
    encoder_cpr: int = 2400
    steps_per_meter: float = 10000.0
    max_step_rate: int = 4000
    encoder_sign: int = 1
    cart_sign: int = 1
    # Where the pole is when the Pico counters are reset ("R" command).
    angle_zero: str = "down"


@dataclass(frozen=True)
class EstimatorConfig:
    # Low-pass weight on the newest finite-difference velocity sample (0..1].
    velocity_alpha: float = 0.3


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str = "models/policy.onnx"
    backend: str = "auto"
    rate_hz: float = 60.0
    obs_order: tuple[str, ...] = OBS_KEYS
    action_type: str = "velocity"
    action_scale: float = 1.0
    action_clip: float = 1.0
    # Only used when action_type = "force": a = F / mass (rough approximation).
    force_mass_kg: float = 1.0


@dataclass(frozen=True)
class SafetyConfig:
    engage_angle_deg: float = 15.0
    disengage_angle_deg: float = 25.0
    max_cart_vel_mps: float = 0.4
    max_missed_reads: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    directory: str = "logs"


@dataclass(frozen=True)
class UiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    push_hz: float = 20.0
    # A jog command expires after this long unless the browser repeats it (dead-man).
    jog_timeout_s: float = 0.3


@dataclass(frozen=True)
class Config:
    serial: SerialConfig
    hardware: HardwareConfig
    estimator: EstimatorConfig
    policy: PolicyConfig
    safety: SafetyConfig
    logging: LoggingConfig
    ui: UiConfig


_SECTIONS = {
    "serial": SerialConfig,
    "hardware": HardwareConfig,
    "estimator": EstimatorConfig,
    "policy": PolicyConfig,
    "safety": SafetyConfig,
    "logging": LoggingConfig,
    "ui": UiConfig,
}


def _build_section(cls, raw: dict, name: str):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{name}] unknown keys: {', '.join(sorted(unknown))}")
    values = dict(raw)
    if "obs_order" in values:
        values["obs_order"] = tuple(values["obs_order"])
    return cls(**values)


def _validate(cfg: Config) -> None:
    hw, pol, saf, est = cfg.hardware, cfg.policy, cfg.safety, cfg.estimator
    if hw.encoder_sign not in (-1, 1) or hw.cart_sign not in (-1, 1):
        raise ConfigError("[hardware] encoder_sign and cart_sign must be 1 or -1")
    if hw.angle_zero not in ANGLE_ZERO:
        raise ConfigError(f"[hardware] angle_zero must be one of {ANGLE_ZERO}")
    if hw.encoder_cpr <= 0 or hw.steps_per_meter <= 0 or hw.max_step_rate <= 0:
        raise ConfigError("[hardware] encoder_cpr, steps_per_meter, max_step_rate must be > 0")
    if not 0.0 < est.velocity_alpha <= 1.0:
        raise ConfigError("[estimator] velocity_alpha must be in (0, 1]")
    if pol.backend not in BACKENDS:
        raise ConfigError(f"[policy] backend must be one of {BACKENDS}")
    if pol.action_type not in ACTION_TYPES:
        raise ConfigError(f"[policy] action_type must be one of {ACTION_TYPES}")
    if pol.rate_hz <= 0:
        raise ConfigError("[policy] rate_hz must be > 0")
    bad = [k for k in pol.obs_order if k not in OBS_KEYS]
    if bad or len(set(pol.obs_order)) != len(pol.obs_order) or not pol.obs_order:
        raise ConfigError(f"[policy] obs_order must be unique entries from {OBS_KEYS}")
    if pol.action_type == "force" and pol.force_mass_kg <= 0:
        raise ConfigError("[policy] force_mass_kg must be > 0")
    if saf.disengage_angle_deg < saf.engage_angle_deg:
        raise ConfigError("[safety] disengage_angle_deg must be >= engage_angle_deg")
    if saf.max_cart_vel_mps <= 0:
        raise ConfigError("[safety] max_cart_vel_mps must be > 0")
    if not 0 < cfg.ui.port < 65536 or cfg.ui.push_hz <= 0 or cfg.ui.jog_timeout_s <= 0:
        raise ConfigError("[ui] port must be 1-65535; push_hz and jog_timeout_s must be > 0")


def load_config(path: str | Path) -> Config:
    """Read a TOML config. Relative paths inside it resolve against its folder."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path} not found — run `make config` to create it from the template")
    with path.open("rb") as f:
        raw = tomllib.load(f)

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown))}")

    sections = {name: _build_section(cls, raw.get(name, {}), name) for name, cls in _SECTIONS.items()}
    cfg = Config(**sections)

    base = path.resolve().parent
    cfg = replace(
        cfg,
        policy=replace(cfg.policy, model_path=str(base / cfg.policy.model_path)),
        logging=replace(cfg.logging, directory=str(base / cfg.logging.directory)),
    )
    _validate(cfg)
    return cfg


def effective_max_velocity(cfg: Config) -> float:
    """Cart speed limit (m/s) honouring both the safety cap and the step-rate cap."""
    step_limited = cfg.hardware.max_step_rate / cfg.hardware.steps_per_meter
    return min(cfg.safety.max_cart_vel_mps, step_limited)

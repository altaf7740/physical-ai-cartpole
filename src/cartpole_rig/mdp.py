"""Rig-specific MDP terms: DC-motor action, delayed noisy state observation, drive randomization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return torch.empty(n, 1, device=device).uniform_(lo, hi)


def _resolve_ids(env_ids, n: int, device) -> torch.Tensor:
    return torch.arange(n, device=device) if env_ids is None else torch.as_tensor(env_ids, device=device)


##
# Action: normalized motor voltage -> cart force
##


class DCMotorAction(ActionTerm):
    """Action in [-1, 1] = PWM duty, i.e. motor voltage / supply voltage.

    The motor drives the carriage through the belt (rigid, no slip), so the motor shaft speed is omega = v / r
    with v the cart velocity. Per physics step:
        I = (V*a - Ke*omega) / R,  torque = Kt*I,  belt force = clip(torque / r, +-F_max)
        cart force = belt force - b*v - Fc*tanh(v / 0.01)       (rail friction)
    The motor + pulley inertia is the ``cart`` joint armature (see asset.py), and the pulley links in the scene
    are spun at omega so the drive is visible. The command passes a deadband and a per-episode delay of
    ``delay_steps`` control steps first. Kt, Ke, R and the friction terms are per-env tensors that
    :func:`randomize_drive` overwrites on reset.
    """

    cfg: DCMotorActionCfg
    _asset: Articulation

    def __init__(self, cfg: DCMotorActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_name)
        self._pulley_ids, _ = self._asset.find_joints(cfg.pulley_joints)
        n = self.num_envs
        self._raw_actions = torch.zeros(n, 1, device=self.device)
        self._command = torch.zeros(n, 1, device=self.device)  # delayed, deadbanded action in [-1, 1]
        self.force = torch.zeros(n, 1, device=self.device)  # [N] last motor + friction force
        ones = torch.ones(n, 1, device=self.device)
        self.kt = ones * cfg.kt
        self.ke = ones * cfg.ke
        self.resistance = ones * cfg.resistance
        self.viscous = ones * cfg.viscous
        self.coulomb = ones * cfg.coulomb
        self._delay = DelayBuffer(cfg.delay_steps[1], n, self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._command

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = actions.clamp(-1.0, 1.0)
        a = torch.where(a.abs() < self.cfg.deadband, torch.zeros_like(a), a)
        self._command[:] = self._delay.compute(a)

    def apply_actions(self):
        cfg = self.cfg
        v = self._asset.data.joint_vel.torch[:, self._joint_ids]
        current = (self._command * cfg.supply_voltage - self.ke * v / cfg.pulley_radius) / self.resistance
        motor_force = (self.kt * current / cfg.pulley_radius).clamp(-cfg.max_force, cfg.max_force)
        self.force[:] = motor_force - self.viscous * v - self.coulomb * torch.tanh(v / 0.01)
        self._asset.set_joint_effort_target_index(target=self.force, joint_ids=self._joint_ids)
        omega = (v / cfg.pulley_radius).expand(-1, len(self._pulley_ids))
        self._asset.set_joint_velocity_target_index(target=omega, joint_ids=self._pulley_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _resolve_ids(env_ids, self.num_envs, self.device)
        self._raw_actions[ids] = 0.0
        self._command[ids] = 0.0
        self._delay.reset(ids)
        lo, hi = self.cfg.delay_steps
        self._delay.set_time_lag(torch.randint(lo, hi + 1, (len(ids),), device=self.device, dtype=torch.int), ids)


@configclass
class DCMotorActionCfg(ActionTermCfg):
    class_type: type = DCMotorAction
    joint_name: str = "cart"
    pulley_joints: str = ".*_pulley"  # visual pulleys spun at belt speed
    supply_voltage: float = 12.0  # [V]
    kt: float = 0.1  # [N*m/A]
    ke: float = 0.1  # [V*s/rad]
    resistance: float = 20.0  # [ohm]
    pulley_radius: float = 0.00637  # [m]
    max_force: float = 11.0  # [N]
    deadband: float = 0.0  # [-]
    viscous: float = 0.0  # [N*s/m]
    coulomb: float = 0.0  # [N]
    delay_steps: tuple = (1, 1)  # [control steps] min, max


##
# Observation: [x, x_dot, theta, theta_dot] + Gaussian noise, delayed
##


class rig_state(ManagerTermBase):
    """Cart position, cart velocity, pole angle (0 = upright), pole angular velocity, in SI units.

    Adds Gaussian noise (``noise_std`` per channel, times a per-episode factor from ``noise_scale_range``), then
    delays the result by a per-episode number of control steps from ``delay_steps``. Scaling to policy units is
    done by the ObsTerm ``scale``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.asset: Articulation = env.scene["robot"]
        self.cart_id = self.asset.find_joints("cart")[0][0]
        self.pole_id = self.asset.find_joints("pole")[0][0]
        self.noise_std = torch.tensor(p["noise_std"], device=self.device)
        self.noise_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.delay = DelayBuffer(p["delay_steps"][1], self.num_envs, self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = _resolve_ids(env_ids, self.num_envs, self.device)
        p = self.cfg.params
        self.noise_scale[ids] = _uniform(*p["noise_scale_range"], len(ids), self.device)
        self.delay.reset(ids)
        lo, hi = p["delay_steps"]
        self.delay.set_time_lag(torch.randint(lo, hi + 1, (len(ids),), device=self.device, dtype=torch.int), ids)

    def __call__(self, env: ManagerBasedEnv, noise_std, noise_scale_range, delay_steps) -> torch.Tensor:
        pos = self.asset.data.joint_pos.torch
        vel = self.asset.data.joint_vel.torch
        c, p = self.cart_id, self.pole_id
        state = torch.stack([pos[:, c], vel[:, c], pos[:, p], vel[:, p]], dim=1)
        state = state + torch.randn_like(state) * self.noise_std * self.noise_scale
        return self.delay.compute(state)


##
# Event: disturbance taps
##


def kick_cart_pole(env: ManagerBasedEnv, env_ids, d_cart_vel, d_pole_vel):
    """Add d_cart_vel [m/s] to the cart velocity and d_pole_vel [rad/s] to the pole angular velocity (floats, or
    tensors with one value per env in env_ids)."""
    asset: Articulation = env.scene["robot"]
    ids = _resolve_ids(env_ids, env.num_envs, env.device)
    joint_ids = [asset.find_joints("cart")[0][0], asset.find_joints("pole")[0][0]]
    vel = asset.data.joint_vel.torch[ids][:, joint_ids].clone()
    vel[:, 0] += d_cart_vel
    vel[:, 1] += d_pole_vel
    asset.write_joint_velocity_to_sim_index(velocity=vel, joint_ids=joint_ids, env_ids=ids)


def push_cart_pole(env: ManagerBasedEnv, env_ids: torch.Tensor | None, pole_vel: float, cart_vel: float):
    """Add a uniform random kick in [-pole_vel, pole_vel] rad/s to the pole and [-cart_vel, cart_vel] m/s to
    the cart, like a tap on the rod or a bump to the carriage."""
    n = env.num_envs if env_ids is None else len(env_ids)
    d_cart = _uniform(-cart_vel, cart_vel, n, env.device)[:, 0]
    d_pole = _uniform(-pole_vel, pole_vel, n, env.device)[:, 0]
    kick_cart_pole(env, env_ids, d_cart, d_pole)


##
# Event: randomize the motor constants and cart friction held by DCMotorAction
##


def randomize_drive(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_name: str,
    kt_range: tuple[float, float],
    ke_range: tuple[float, float],
    resistance_range: tuple[float, float],
    friction_range: tuple[float, float],
):
    """Scale the nominal Kt, Ke, R and cart friction of the motor action term by uniform factors."""
    term: DCMotorAction = env.action_manager.get_term(action_name)
    cfg = term.cfg
    ids = _resolve_ids(env_ids, env.num_envs, env.device)
    n = len(ids)
    term.kt[ids] = cfg.kt * _uniform(*kt_range, n, env.device)
    term.ke[ids] = cfg.ke * _uniform(*ke_range, n, env.device)
    term.resistance[ids] = cfg.resistance * _uniform(*resistance_range, n, env.device)
    f = _uniform(*friction_range, n, env.device)
    term.viscous[ids] = cfg.viscous * f
    term.coulomb[ids] = cfg.coulomb * f

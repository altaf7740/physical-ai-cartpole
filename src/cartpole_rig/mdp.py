"""Rig-specific MDP terms: stepper action, the Pi's state estimate as observation, disturbance taps, randomization."""

from __future__ import annotations

import math
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


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


##
# Action: cart acceleration -> Pi speed command -> Pico ramp -> stepper
##


class StepperAction(ActionTerm):
    """Action in [-1, 1] = cart acceleration / max_accel, exactly as the Pi runs it (interface.ActionConverter,
    action_type = "acceleration") and the Pico executes it (pico_main.py).

    Per control step (Pi): ``v_cmd = clip(v_cmd + a * max_accel * dt, +-max_speed)``; the command reaches the
    Pico ``delay_steps`` control steps later (it goes out with the next telemetry request).
    Per physics step (Pico): the stepper speed ramps toward the command at ``pico_accel``; from rest it starts at
    up to ``pico_start_speed`` without a ramp, and when stopping or reversing it drops to 0 once it is below that.
    The cart joint's stiff velocity drive follows the stepper speed (a stepper that does not lose steps moves as
    commanded). The term also applies the pole pivot's Coulomb friction and spins the visual pulleys.
    ``accel_scale`` and ``friction_scale`` are per-env factors that :func:`randomize_stepper` sets on reset.
    """

    cfg: StepperActionCfg
    _asset: Articulation

    def __init__(self, cfg: StepperActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._cart_ids, _ = self._asset.find_joints(cfg.joint_name)
        self._pole_ids, _ = self._asset.find_joints(cfg.pole_joint)
        self._pulley_ids, _ = self._asset.find_joints(cfg.pulley_joints)
        n = self.num_envs
        self._control_dt = env.step_dt
        self._physics_dt = env.physics_dt
        self._raw_actions = torch.zeros(n, 1, device=self.device)
        self.v_cmd = torch.zeros(n, 1, device=self.device)  # [m/s] Pi speed command
        self.target = torch.zeros(n, 1, device=self.device)  # [m/s] command the Pico is executing (delayed)
        self.speed = torch.zeros(n, 1, device=self.device)  # [m/s] stepper speed
        self.accel_scale = torch.ones(n, 1, device=self.device)
        self.friction_scale = torch.ones(n, 1, device=self.device)
        self._delay = DelayBuffer(cfg.delay_steps[1], n, self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.target

    def process_actions(self, actions: torch.Tensor):
        cfg = self.cfg
        self._raw_actions[:] = actions
        a = actions.clamp(-1.0, 1.0) * cfg.max_accel * self.accel_scale
        self.v_cmd[:] = (self.v_cmd + a * self._control_dt).clamp(-cfg.max_speed, cfg.max_speed)
        self.target[:] = self._delay.compute(self.v_cmd)

    def apply_actions(self):
        cfg = self.cfg
        h = self._physics_dt
        v, tgt = self.speed, self.target
        # from rest: jump to the start speed (or the target, if lower)
        start = (v == 0) & (tgt != 0)
        v = torch.where(start, torch.sign(tgt) * tgt.abs().clamp(max=cfg.pico_start_speed), v)
        s = v.abs()
        same_dir = (tgt != 0) & (torch.sign(tgt) == torch.sign(v))
        dv = cfg.pico_accel * h
        # same direction: ramp toward |target|; otherwise decelerate and stop below the start speed
        s_same = torch.where(s < tgt.abs(), torch.minimum(s + dv, tgt.abs()), torch.maximum(s - dv, tgt.abs()))
        s_stop = s - dv
        s_new = torch.where(same_dir, s_same, torch.where(s_stop <= cfg.pico_start_speed, 0.0, s_stop))
        self.speed[:] = torch.sign(v) * s_new
        self._asset.set_joint_velocity_target_index(target=self.speed, joint_ids=self._cart_ids)

        # pivot Coulomb friction (smoothed sign)
        w = self._asset.data.joint_vel.torch[:, self._pole_ids]
        tau = -cfg.pivot_coulomb_torque * self.friction_scale * torch.tanh(w / 0.02)
        self._asset.set_joint_effort_target_index(target=tau, joint_ids=self._pole_ids)

        omega = (self.speed / cfg.pulley_radius).expand(-1, len(self._pulley_ids))
        self._asset.set_joint_velocity_target_index(target=omega, joint_ids=self._pulley_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _resolve_ids(env_ids, self.num_envs, self.device)
        self._raw_actions[ids] = 0.0
        self.v_cmd[ids] = 0.0
        self.target[ids] = 0.0
        self.speed[ids] = 0.0
        self._delay.reset(ids)
        lo, hi = self.cfg.delay_steps
        self._delay.set_time_lag(torch.randint(lo, hi + 1, (len(ids),), device=self.device, dtype=torch.int), ids)


@configclass
class StepperActionCfg(ActionTermCfg):
    class_type: type = StepperAction
    joint_name: str = "cart"
    pole_joint: str = "pole"
    pulley_joints: str = ".*_pulley"  # visual pulleys spun at belt speed
    max_speed: float = 0.25  # [m/s]
    max_accel: float = 5.0  # [m/s^2] action +1
    pico_accel: float = 5.0  # [m/s^2]
    pico_start_speed: float = 0.03  # [m/s]
    pulley_radius: float = 0.00637  # [m]
    pivot_coulomb_torque: float = 0.0  # [N*m]
    delay_steps: tuple = (1, 1)  # [control steps] min, max


##
# Observation: [x, x_dot, theta, theta_dot] as the Pi estimates them
##


class rig_state(ManagerTermBase):
    """Cart position, cart velocity, pole angle (0 = upright), pole angular velocity, in SI units, computed the way
    hardware/src/state.py does on the Pi:

    - x is a whole number of steps (1 / steps_per_meter), theta a whole number of encoder counts, plus a
      per-episode angle-zero offset (uniform in +-angle_offset; the zero is set with the pole hanging);
    - velocities are finite differences of those over one control step, low-passed with weight ``alpha``.

    Then adds Gaussian noise (``noise_std`` per channel, times a per-episode factor from ``noise_scale_range``) and
    delays the result by a per-episode number of control steps from ``delay_steps``. Scaling to policy units is
    done by the ObsTerm ``scale``. The estimator advances once per env step (repeated calls in the same step
    return the same value).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        n = self.num_envs
        self.asset: Articulation = env.scene["robot"]
        self.cart_id = self.asset.find_joints("cart")[0][0]
        self.pole_id = self.asset.find_joints("pole")[0][0]
        self.dt = env.step_dt
        self.noise_std = torch.tensor(p["noise_std"], device=self.device)
        self.noise_scale = torch.ones(n, 1, device=self.device)
        self.offset = torch.zeros(n, device=self.device)
        self.prev = torch.zeros(n, 2, device=self.device)  # last quantized (x, theta)
        self.vel = torch.zeros(n, 2, device=self.device)  # filtered (x_dot, theta_dot)
        self.fresh = torch.ones(n, dtype=torch.bool, device=self.device)
        self.max_delay = p["delay_steps"][1]
        self.delay = DelayBuffer(self.max_delay, n, self.device) if self.max_delay > 0 else None
        self._last_step = -1
        self._out = torch.zeros(n, 4, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = _resolve_ids(env_ids, self.num_envs, self.device)
        p = self.cfg.params
        self.noise_scale[ids] = _uniform(*p["noise_scale_range"], len(ids), self.device)
        self.offset[ids] = _uniform(-p["angle_offset"], p["angle_offset"], len(ids), self.device)[:, 0]
        self.fresh[ids] = True
        if self.delay is not None:
            self.delay.reset(ids)
            lo, hi = p["delay_steps"]
            self.delay.set_time_lag(torch.randint(lo, hi + 1, (len(ids),), device=self.device, dtype=torch.int), ids)
        self._last_step = -1  # recompute on the next call

    def __call__(self, env: ManagerBasedEnv, noise_std, noise_scale_range, delay_steps, steps_per_meter,
                 encoder_cpr, alpha, angle_offset) -> torch.Tensor:
        step = int(getattr(env, "common_step_counter", 0))
        if step == self._last_step:
            return self._out
        self._last_step = step

        pos = self.asset.data.joint_pos.torch
        vel = self.asset.data.joint_vel.torch
        c, p = self.cart_id, self.pole_id
        q_ang = 2 * math.pi / encoder_cpr
        qx = torch.round(pos[:, c] * steps_per_meter) / steps_per_meter
        qth = torch.round(_wrap(pos[:, p] + self.offset) / q_ang) * q_ang
        q = torch.stack([qx, qth], dim=1)

        raw = torch.stack([(qx - self.prev[:, 0]) / self.dt, _wrap(qth - self.prev[:, 1]) / self.dt], dim=1)
        self.vel += alpha * (raw - self.vel)
        # a just-reset env: start the filter at the true velocities (the Pi's estimator runs before it engages)
        fresh = self.fresh.unsqueeze(1)
        self.vel = torch.where(fresh, torch.stack([vel[:, c], vel[:, p]], dim=1), self.vel)
        self.prev = q
        self.fresh[:] = False

        state = torch.stack([q[:, 0], self.vel[:, 0], q[:, 1], self.vel[:, 1]], dim=1)
        state = state + torch.randn_like(state) * self.noise_std * self.noise_scale
        self._out = self.delay.compute(state) if self.delay is not None else state
        return self._out


##
# Event: disturbance taps
##


def kick_cart_pole(env: ManagerBasedEnv, env_ids, d_cart_vel, d_pole_vel):
    """Add d_cart_vel [m/s] to the cart velocity and d_pole_vel [rad/s] to the pole angular velocity (floats, or
    tensors with one value per env in env_ids). The stepper drive cancels a cart kick within a physics step."""
    asset: Articulation = env.scene["robot"]
    ids = _resolve_ids(env_ids, env.num_envs, env.device)
    joint_ids = [asset.find_joints("cart")[0][0], asset.find_joints("pole")[0][0]]
    vel = asset.data.joint_vel.torch[ids][:, joint_ids].clone()
    vel[:, 0] += d_cart_vel
    vel[:, 1] += d_pole_vel
    asset.write_joint_velocity_to_sim_index(velocity=vel, joint_ids=joint_ids, env_ids=ids)


def push_pole(env: ManagerBasedEnv, env_ids: torch.Tensor | None, pole_vel: float):
    """Add a uniform random kick in [-pole_vel, pole_vel] rad/s to the pole, like a tap on the rod."""
    n = env.num_envs if env_ids is None else len(env_ids)
    kick_cart_pole(env, env_ids, 0.0, _uniform(-pole_vel, pole_vel, n, env.device)[:, 0])


##
# Event: randomize the stepper command integration and the pivot friction held by StepperAction
##


def randomize_stepper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_name: str,
    accel_range: tuple[float, float],
    friction_range: tuple[float, float],
):
    term: StepperAction = env.action_manager.get_term(action_name)
    ids = _resolve_ids(env_ids, env.num_envs, env.device)
    term.accel_scale[ids] = _uniform(*accel_range, len(ids), env.device)
    term.friction_scale[ids] = _uniform(*friction_range, len(ids), env.device)

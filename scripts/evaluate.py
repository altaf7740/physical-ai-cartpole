"""Stress test: one full episode in each of N envs (all domain randomization on), then report the failure rate,
the safety margin, and which randomized parameters the failures come from.

    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\evaluate.py --headless                   # taps as in training
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\evaluate.py --headless --push_scale 0     # no disturbances
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\evaluate.py --headless --push_scale 2     # taps twice as hard

Per-episode results are written to <run>/eval_push<scale>.csv.
"""

import argparse
import csv
import importlib.metadata as metadata
import os

from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation

parser = argparse.ArgumentParser(description="Stress-test the cart-pole rig policy.")
parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt; default: latest in logs/.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of episodes (one per env).")
parser.add_argument("--push_scale", type=float, default=1.0, help="Tap strength vs. RIG.push_*; 0 = no taps.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import cartpole_rig  # noqa: E402
from cartpole_rig.rig_cfg import RIG  # noqa: E402


def episode_params(env) -> dict[str, torch.Tensor]:
    """Each env's randomized parameters for the episode that just started, as ratios to nominal (delays in steps)."""
    robot = env.scene["robot"]
    motor = env.action_manager.get_term("motor")
    # the observation manager has no public accessor for a term instance in Isaac Lab 3.0
    obs_term = env.observation_manager._group_obs_class_term_cfgs["policy"][0].func
    j = {n: i for i, n in enumerate(robot.joint_names)}
    b = {n: i for i, n in enumerate(robot.body_names)}
    return {
        "cart_mass": robot.data.body_mass.torch[:, b["cart"]] / RIG.cart_mass,
        "pole_mass": robot.data.body_mass.torch[:, b["pole"]] / RIG.pole_mass,
        "pole_com": robot.data.body_com_pos_b.torch[:, b["pole"], 2] / RIG.pole_com,
        "motor_inertia": robot.data.joint_armature.torch[:, j["cart"]] / RIG.reflected_mass,
        "pivot_friction": robot.data.joint_damping.torch[:, j["pole"]] / RIG.pivot_viscous,
        "kt": motor.kt[:, 0] / RIG.motor_kt,
        "ke": motor.ke[:, 0] / RIG.motor_ke,
        "resistance": motor.resistance[:, 0] / RIG.motor_resistance,
        "cart_friction": motor.viscous[:, 0] / RIG.cart_viscous,
        "obs_noise": obs_term.noise_scale[:, 0],
        "obs_delay": obs_term.delay.time_lags.float(),
        "action_delay": motor._delay.time_lags.float(),
    }


def main():
    env_cfg = load_cfg_from_registry(cartpole_rig.TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(cartpole_rig.TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.push_scale == 0:
        env_cfg.events.push = None
    else:
        env_cfg.events.push.params = {
            "pole_vel": RIG.push_pole_vel * args_cli.push_scale,
            "cart_vel": RIG.push_cart_vel * args_cli.push_scale,
        }

    with launch_simulation(env_cfg, args_cli):
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ckpt = args_cli.checkpoint or get_checkpoint_path(log_root, ".*", "model_.*.pt")
        print(f"[INFO] checkpoint: {ckpt}")
        env = RslRlVecEnvWrapper(gym.make(cartpole_rig.TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        uenv, n, dev = env.unwrapped, env.num_envs, env.unwrapped.device
        robot = uenv.scene["robot"]
        cart_id, pole_id = robot.find_joints("cart")[0][0], robot.find_joints("pole")[0][0]
        params = episode_params(uenv)  # sampled by the initial reset
        active = torch.ones(n, dtype=torch.bool, device=dev)  # still in its first episode
        outcome = torch.zeros(n, dtype=torch.long, device=dev)  # 0 balanced, 1 pole fell, 2 left the rail
        length = torch.zeros(n, device=dev)
        max_theta = torch.zeros(n, device=dev)
        max_x = torch.zeros(n, device=dev)

        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(uenv.max_episode_length + 1):
                # true state (not the noisy, delayed observation) for the margin numbers
                theta = robot.data.joint_pos.torch[:, pole_id].abs()
                x = robot.data.joint_pos.torch[:, cart_id].abs()
                max_theta = torch.where(active, torch.maximum(max_theta, theta), max_theta)
                max_x = torch.where(active, torch.maximum(max_x, x), max_x)
                obs, _, dones, extras = env.step(policy(obs))
                length += active.float()
                done = dones.bool() & active
                if done.any():
                    tm = uenv.termination_manager
                    outcome[done & tm.get_term("pole_fallen").bool()] = 1
                    outcome[done & tm.get_term("cart_out_of_bounds").bool()] = 2
                    active &= ~done
                if not active.any():
                    break

        dt = uenv.step_dt
        ok = outcome == 0
        fail = ~ok
        print(f"\n==== stress test: {n} episodes x {RIG.episode_s:.0f} s, push_scale {args_cli.push_scale} ====")
        if args_cli.push_scale:
            print(
                f"taps every {RIG.push_interval_s[0]:.0f}-{RIG.push_interval_s[1]:.0f} s: pole +-"
                f"{RIG.push_pole_vel * args_cli.push_scale:.2f} rad/s, cart +-{RIG.push_cart_vel * args_cli.push_scale:.2f} m/s"
            )
        print(f"balanced full episode: {int(ok.sum())}/{n} ({100 * ok.float().mean():.2f} %)")
        print(f"pole fell: {int((outcome == 1).sum())}   left rail: {int((outcome == 2).sum())}")
        if fail.any():
            print(f"failed episodes lasted {length[fail].mean() * dt:.2f} s on average")
        q = torch.tensor([0.5, 0.95, 1.0], device=dev)
        mt = torch.quantile(max_theta[ok], q).tolist() if ok.any() else [float("nan")] * 3
        mx = torch.quantile(max_x[ok], q).tolist() if ok.any() else [float("nan")] * 3
        print(
            f"margin in balanced episodes (median / p95 / max): |theta| {mt[0]:.3f} / {mt[1]:.3f} / {mt[2]:.3f} rad"
            f" (limit {RIG.max_pole_angle:.3f}),  |x| {mx[0]:.3f} / {mx[1]:.3f} / {mx[2]:.3f} m (limit {RIG.cart_limit})"
        )

        # failure rate in the low / middle / high third of each randomized parameter
        print("\nfailure rate by parameter third (values are ratios to nominal; delays in steps):")
        print(f"{'parameter':>15} | {'low third':>18} | {'middle third':>18} | {'high third':>18}")
        for name, v in params.items():
            if v.max() - v.min() < 1e-6:
                continue
            cells = []
            edges = torch.quantile(v, torch.tensor([1 / 3, 2 / 3], device=dev))
            for lo_mask in (v <= edges[0], (v > edges[0]) & (v <= edges[1]), v > edges[1]):
                if lo_mask.sum() == 0:  # discrete values (delays) can leave a third empty
                    cells.append(f"{'-':>18}")
                    continue
                rate = 100 * fail[lo_mask].float().mean()
                cells.append(f"{rate:6.2f} % @{v[lo_mask].mean():6.2f}".rjust(18))
            print(f"{name:>15} | " + " | ".join(cells))

        out = os.path.join(os.path.dirname(ckpt), f"eval_push{args_cli.push_scale:g}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["outcome", "length_s", "max_abs_theta", "max_abs_x", *params.keys()])
            cols = [outcome, length * dt, max_theta, max_x, *params.values()]
            for row in torch.stack([c.float() for c in cols], dim=1).tolist():
                w.writerow([int(row[0]), *(round(c, 5) for c in row[1:])])
        print(f"\n[INFO] per-episode results: {out}")
        env.close()


if __name__ == "__main__":
    main()

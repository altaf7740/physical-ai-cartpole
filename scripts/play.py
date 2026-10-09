"""Export a trained checkpoint and run it in sim.

Writes <run>/exported/{policy.pt, policy.onnx, policy_io.json}, then drives the env with the exported
TorchScript policy (the same file the Pi loads) and prints episode statistics.

    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\play.py --headless              # latest run, stats only
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\play.py --viz kit --num_envs 16  # watch it
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\play.py --viz kit --num_envs 16 --steps 0 --real-time --keyboard
"""

import argparse
import dataclasses
import importlib.metadata as metadata
import json
import os
import time

from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation

parser = argparse.ArgumentParser(description="Export and play the cart-pole rig policy.")
parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt; default: latest in logs/.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1800, help="Control steps to run (0 = forever). 1800 = 30 s.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--keyboard", action="store_true", help="Tap the pole / bump the cart from the viewer keyboard (no random taps)."
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import cartpole_rig  # noqa: E402
from cartpole_rig import mdp  # noqa: E402
from cartpole_rig.rig_cfg import RIG  # noqa: E402


def policy_io(rig) -> dict:
    """Everything the Pi needs to build the policy input and use its output."""
    return {
        "obs_order": ["cart_pos_m", "cart_vel_m_s", "pole_angle_rad", "pole_angvel_rad_s"],
        "obs_scale": list(rig.obs_scale),
        "obs_formula": "policy_input[i] = raw[i] * obs_scale[i]; no offset, no clipping, no running normalizer",
        "conventions": "x=0 rail centre, +x = direction of travel for +action; theta=0 upright, "
        "theta>0 when the pole leans toward +x",
        "action": "a = clamp(policy(obs)[0], -1, 1); PWM duty = a, signed; motor voltage = a * supply_voltage",
        "supply_voltage": rig.supply_voltage,
        "deadband": rig.deadband,
        "control_hz": rig.control_hz,
        "cart_limit_m": rig.cart_limit,
        "max_pole_angle_rad": rig.max_pole_angle,
        "trained_obs_delay_steps": list(rig.obs_delay_steps),
        "trained_action_delay_steps": list(rig.action_delay_steps),
        "trained_obs_noise_std": list(rig.obs_noise_std),
        # all rig parameters (used by pi/fake_rig.py to simulate the rig for testing the Pi page)
        "rig": dataclasses.asdict(rig),
    }


def main():
    env_cfg = load_cfg_from_registry(cartpole_rig.TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(cartpole_rig.TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.keyboard:
        env_cfg.events.push = None  # only your taps
    # the Kit viewer costs ~16 ms per render on top of ~16 ms per step, so render every 2nd control step
    # (physics, policy and the signal panel still run every step); no effect headless
    env_cfg.sim.render_interval = 2 * env_cfg.decimation

    with launch_simulation(env_cfg, args_cli):
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ckpt = args_cli.checkpoint or get_checkpoint_path(log_root, ".*", "model_.*.pt")
        print(f"[INFO] checkpoint: {ckpt}")

        env = gym.make(cartpole_rig.TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # export
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        out_dir = os.path.join(os.path.dirname(ckpt), "exported")
        runner.export_policy_to_jit(path=out_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=out_dir, filename="policy.onnx")
        with open(os.path.join(out_dir, "policy_io.json"), "w") as f:
            json.dump(policy_io(RIG), f, indent=2)
        print(f"[INFO] exported policy.pt, policy.onnx, policy_io.json to {out_dir}")

        # run the exported TorchScript file, not the in-memory model
        policy = torch.jit.load(os.path.join(out_dir, "policy.pt"), map_location=env.unwrapped.device).eval()
        ref = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        with torch.inference_mode():
            diff = (policy(obs["policy"]) - ref(obs)).abs().max().item()
        print(f"[INFO] exported vs in-memory policy max |diff|: {diff:.2e}")

        n = env.num_envs
        ep_len = torch.zeros(n, device=env.unwrapped.device)
        lengths, timeouts, falls = [], 0, 0
        max_angle = max_pos = 0.0
        step, dt = 0, env.unwrapped.step_dt
        sim = env.unwrapped.sim
        had_viewer = bool(sim.visualizers)
        # viewer-only helpers (Kit UI / input; imported after Kit has started)
        if args_cli.keyboard or sim.visualizers:
            from viewer_ui import KeyTaps, SignalPanel
        keys = KeyTaps(RIG) if args_cli.keyboard else None
        # live signal panel next to the Kit viewer
        panel = None
        if any(type(v).__name__ == "KitVisualizer" for v in sim.visualizers):
            panel = SignalPanel(RIG)
            robot = env.unwrapped.scene["robot"]
            motor = env.unwrapped.action_manager.get_term("motor")
            cart_id, pole_id = robot.find_joints("cart")[0][0], robot.find_joints("pole")[0][0]
        speed, t_last = 1.0, time.time()  # sim time / wall time, smoothed
        while args_cli.steps == 0 or step < args_cli.steps:
            # Isaac Lab drops the viewer when its window is closed and keeps simulating; stop instead
            if had_viewer and not sim.visualizers:
                print("[INFO] viewer closed, stopping")
                break
            t0 = time.time()
            for d_cart, d_pole in keys.pop() if keys else []:
                mdp.kick_cart_pole(env.unwrapped, None, d_cart, d_pole)
            with torch.inference_mode():
                actions = policy(obs["policy"]).clamp(-1.0, 1.0)
                obs, _, dones, extras = env.step(actions)
            raw = obs["policy"] / torch.tensor(RIG.obs_scale, device=obs["policy"].device)
            max_pos = max(max_pos, raw[:, 0].abs().max().item())
            max_angle = max(max_angle, raw[:, 2].abs().max().item())
            if panel:
                speed = 0.97 * speed + 0.03 * dt / max(time.time() - t_last, 1e-6)
                t_last = time.time()
                q, qd = robot.data.joint_pos.torch[0], robot.data.joint_vel.torch[0]
                true = [q[cart_id].item(), qd[cart_id].item(), q[pole_id].item(), qd[pole_id].item()]
                panel.update(
                    raw[0].tolist(),
                    true,
                    actions[0, 0].item(),
                    motor.processed_actions[0, 0].item() * RIG.supply_voltage,
                    motor.force[0, 0].item(),
                    speed,
                )
            ep_len += 1
            done = dones.bool()
            if done.any():
                to = extras["time_outs"].bool() & done
                timeouts += int(to.sum())
                falls += int((done & ~to).sum())
                lengths += ep_len[done].tolist()
                ep_len[done] = 0
            step += 1
            remaining = dt - (time.time() - t0)
            if args_cli.real_time and remaining > 0:
                time.sleep(remaining)

        unfinished = ep_len.tolist()
        print("---- play summary ----")
        print(f"control steps: {step} x {n} envs ({step * dt:.1f} s sim each, episode = {RIG.episode_s:.0f} s)")
        print(f"episodes ended: {timeouts} time-outs (balanced the full episode), {falls} falls/out-of-rail")
        if lengths:
            print(f"mean finished episode length: {sum(lengths) / len(lengths) * dt:.2f} s")
        print(f"still running at stop: {len(unfinished)} envs, mean age {sum(unfinished) / n * dt:.2f} s")
        print(f"max |x| observed {max_pos:.3f} m, max |theta| observed {max_angle:.3f} rad (noisy obs)")
        if keys:
            keys.close()
        if panel:
            panel.destroy()
        env.close()


if __name__ == "__main__":
    main()

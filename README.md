# Cart-pole rig: balance policy (Isaac Lab 3.0)

Balance-only cart-pole matched to a real belt-driven rig, for sim-to-real transfer. Structure follows Isaac
Lab's `Isaac-Cartpole-v0` (manager-based env, RSL-RL PPO). Built against Isaac Lab 3.0.0, Isaac Sim 6.0.1,
RSL-RL 5.0.1.

- Task id: `Cartpole-Rig-Balance-v0`
- Control: 60 Hz (physics 240 Hz, decimation 4)
- Action (1): cart acceleration / `max_accel` in [-1, 1]. As on the rig, it is integrated into a speed command
  (capped at `max_speed` = 0.25 m/s), which reaches the stepper one control step later through the Pico's ramp;
  the cart follows the stepper exactly (stiff velocity drive, no lost steps)
- Observation (4): cart position, cart velocity, pole angle, pole angular velocity, computed like the Pi does
  (whole steps and encoder counts, low-passed finite-difference velocities, an angle-zero offset), then scaled
- Pole: the measured swing period (1.00 s, i.e. a uniform 0.375 m rod) and Coulomb pivot friction
- Episode: 15 s; starts within 5 deg of upright; ends early if |pole angle| > 17 deg or |cart position| > 0.35 m

## Layout

```
src/cartpole_rig/
  rig_cfg.py               ALL physical parameters, delays, noise, reward weights, DR ranges (edit this)
  usd_builder.py           builds assets/cartpole_rig.usda from rig_cfg (pxr only): rail + end blocks, gear
                           motor, GT2 pulleys and belt, carriage with the pendulum encoder, pendulum
  asset.py                 Isaac Lab articulation + actuators; rebuilds the USD each time the scene spawns
  mdp.py                   StepperAction, rig_state observation (the Pi's estimator), taps, randomize_stepper
  env_cfg.py               manager-based env cfg (scene, actions, obs, events, rewards, terminations)
  agents/rsl_rl_ppo_cfg.py PPO, 64x64 ELU actor and critic, no observation normalizer
scripts/
  train.py                 Isaac Lab's stock rsl_rl/train.py with this task filled in
  play.py                  export policy.pt / policy.onnx / policy_io.json, then run and print statistics
  evaluate.py              stress test: one episode per env (4096), failure rate, margin, failures by parameter
  isaacsim_play.py         plain Isaac Sim player (`make run-sim`): policy + stepper on Isaac Sim's physics steps
  viewer_ui.py             signal panel + keyboard taps (Kit UI only; used by play.py and isaacsim_play.py)
policy/                    the current trained policy, committed to git (`make policy`):
                           policy.pt, policy_pi.onnx, policy_io.json, SOURCE.txt (which run it came from)
hardware/                  the real rig: Pi 4 policy runner + web UI, the Pico firmware (encoder + TB6600
                           step/dir, limit switches), the rig's config and track calibration; see hardware/README.md
```

## Setup (once)

```powershell
C:\isaac\IsaacLab\isaaclab.bat -p -m pip install -e C:\Users\Admin\Desktop\rnd\cartpole --no-deps
```

## Shortcuts (`make`, run in this folder)

| command | what it does |
|---|---|
| `make` / `make run-lab` | viewer + signal panel; LEFT/RIGHT tap the pole, UP/DOWN tap strength |
| `make run-lab-random` | viewer + signal panel with random taps, as in training |
| `make run-sim` | the rig in the **plain Isaac Sim app** (no Isaac Lab): nominal parameters, newest exported `policy.pt` (or `POLICY=...`), signal panel, same keys plus **R** = reset; it resets by itself after a fall |
| `make install` | install this package into Isaac Lab's Python (once) |
| `make train` | train headless (600 iterations, about 7 min) |
| `make play` | headless: export `policy.pt` / `.onnx` / `policy_io.json` and print statistics |
| `make eval` | stress test, 4096 episodes (`PUSH=0` no taps, `PUSH=2` twice as hard) |
| `make policy` | copy the newest export into `policy/` (commit it to publish a new policy) |
| `make help` | list the targets |

Options: `ENVS=4` sets how many rigs are shown and `CKPT=logs\...\model_599.pt` picks a checkpoint (default:
latest run). If Isaac Lab is not at `C:\isaac\IsaacLab`, set `ISAACLAB=...`.

## Train, play, export

Run these from this folder. Logs go to `logs\rsl_rl\cartpole_rig\<date>`.

```powershell
# train (600 iterations by default)
C:\isaac\IsaacLab\isaaclab.bat -p scripts\train.py --headless
# export the latest checkpoint, then run 30 s headless and print balance statistics
C:\isaac\IsaacLab\isaaclab.bat -p scripts\play.py --headless
# watch it (Kit viewer; runs until you close the window, then prints the statistics)
C:\isaac\IsaacLab\isaaclab.bat -p scripts\play.py --viz kit --num_envs 16 --steps 0 --real-time
# watch it and disturb it yourself from the keyboard (random taps are off; click the viewer window first):
#   LEFT / RIGHT tap the pole, UP / DOWN stronger / weaker
C:\isaac\IsaacLab\isaaclab.bat -p scripts\play.py --viz kit --num_envs 16 --steps 0 --real-time --keyboard
# With the Kit viewer, a "Rig signals (env 0)" panel shows the policy inputs (x, x_dot, theta, theta_dot as
# the Pi would estimate them, next to the true values), the policy output (cart acceleration), the speed
# command the Pico is executing and the stepper speed, with 5 s plots. Its top line shows the sim
# speed: the viewer runs at about 0.8x real time on this PC, because each Kit frame costs ~16 ms on top of
# ~16 ms per step. Headless runs are not slowed down.
# a specific checkpoint
C:\isaac\IsaacLab\isaaclab.bat -p scripts\play.py --headless --checkpoint logs\rsl_rl\cartpole_rig\<run>\model_299.pt
```

Stress test before trying a policy on the rig. It runs 4096 episodes with all randomization on and the taps
described below:
```powershell
C:\isaac\IsaacLab\isaaclab.bat -p scripts\evaluate.py --headless                   # taps as in training
C:\isaac\IsaacLab\isaaclab.bat -p scripts\evaluate.py --headless --push_scale 0     # no taps
C:\isaac\IsaacLab\isaaclab.bat -p scripts\evaluate.py --headless --push_scale 2     # taps twice as hard
```
It prints the balanced-episode rate, the worst |theta| and |x| reached (compared with the limits), and the
failure rate in the low / middle / high third of each randomized parameter. That last table shows where the
policy is weak, for example "fails mostly when the pivot friction is high". Per-episode rows go to
`<run>\eval_push<scale>.csv`.

`play.py` writes `<run>\exported\policy.pt` (TorchScript), `policy.onnx` and `policy_io.json`. RSL-RL's
`policy.onnx` keeps its weights in a separate `policy.onnx.data` file and uses opset 18, so `play.py` also
writes **`policy_pi.onnx`**: one self-contained file in opset 13 / IR 7, which any ONNX runtime from about 2020
on loads (onnxruntime on the Pi, OpenCV DNN, ...). The network only uses Gemm and Elu, so the conversion
changes nothing; `play.py` checks it against `policy.pt` (max difference about 1e-6). Input `obs` is
float32 [1, 4], output `actions` is [1, 1]; the same scaling and clamping as `policy.pt` apply. It then runs
the sim with the exported `policy.pt` file itself, so the file you copy to the Pi is the one that was tested.

## Policy interface (what the Pi must do)

Input is a float32 tensor of shape (1, 4):

| idx | quantity | unit | scale (multiply) | sign convention |
|----:|---|---|---|---|
| 0 | cart position x | m | 1 / 0.35 | 0 at rail centre; + is where the cart goes for a + action |
| 1 | cart velocity | m/s | 1 / 0.25 | d/dt of idx 0 |
| 2 | pole angle theta | rad | 1 / 0.1 | 0 = upright; + when the pole leans toward +x; wrap to [-pi, pi] |
| 3 | pole angular velocity | rad/s | 1 / 1.0 | d/dt of idx 2 |

`policy_input = [x, x_dot, theta, theta_dot] * obs_scale`. Do not add an offset, clip or apply a running
normalizer. The exported model contains no normalizer (`obs_normalization=False`). The env computes these the
way `hardware/src/state.py` does: x from whole steps, theta from whole encoder counts, velocities as finite
differences over one control step low-passed with `velocity_alpha` (0.3).

Output has shape (1, 1). Use `a = clamp(output, -1, 1)`; the output is not clipped inside the model. `a` is the
cart acceleration in units of `max_accel` (5 m/s²): every control step the Pi does
`v_cmd = clip(v_cmd + a * max_accel * dt, ±max_speed)` and sends `v_cmd * steps_per_meter` to the Pico on the
next tick (`action_type = "acceleration"`, `action_scale = 5.0` in `hardware/config.toml`).

`policy_io.json` holds the same numbers (taken from `rig_cfg.py` at export time); its `pi_config` block lists
the `hardware/config.toml` values the policy was trained for. If you change `obs_scale` you must retrain.

**Signs.** `encoder_sign = 1` and `cart_sign = 1` match the sim. Checked on the rig from the hanging pole:
accelerating the cart toward +x makes the encoder count go up (the pole lags toward -x, so theta passes
through ±pi increasing), in 11 of 12 jog starts.

## Model details

**Stepper** (`mdp.StepperAction`, `asset.py`). A stepper that does not lose steps moves exactly as commanded,
so the cart joint has a stiff velocity drive (`cart_drive_damping`) whose target is the stepper speed. Cart
mass and rail friction therefore do not affect the cart motion. Per control step the Pi integrates the action
into `v_cmd` (capped at `max_speed`); one control step later the Pico executes it. Per physics step the Pico
ramp moves the stepper speed toward the command at `pico_accel`: from rest it jumps to at most
`pico_start_speed` (Pico `START_HZ`), and when stopping or reversing it drops to 0 below that.

`max_speed` = 0.25 m/s because the stepper at 1.0 A loses steps above about 0.3 m/s (measured 2026-10-23:
0 steps lost at 0.1 / 0.2 m/s, 31 at 0.3 m/s, 702 at 0.4 m/s over six ±8 cm jogs at 5 m/s²). Lost steps are
not modelled; the cart position is only the count of commanded steps, so they would show as a drifting x.

**Pole.** The hanging swing period is 1.00 s, so J_pivot / (m l) = 0.250 m, the same as a uniform 0.375 m
rod (`pole_length`). The pivot friction is Coulomb: the swing amplitude drops by a constant 2.35° per cycle,
which is a friction deceleration of 0.40 rad/s² (`pivot_coulomb_accel`, applied as a joint effort).

**Latency.** The Pi acts on the reply it just got (`obs_delay_steps` = 0) and its command goes out with the next
request (`action_delay_steps` = 1).

**Noise and angle zero.** On top of the modelled quantization and filter, a little Gaussian noise
(`obs_noise_std` × a per-episode `dr_obs_noise` factor). The angle zero is set with the pole hanging, where the
pivot friction lets it rest up to ~0.8° off, so each episode adds a constant angle offset of up to ±1°.

**Disturbances.** Every 2–4 s each env gets a random tap: the pole's angular velocity jumps by up to
±0.4 rad/s (`push_pole_vel`). At 0.25 m/s the rig can only absorb about 0.65 rad/s from upright, and catch up
to about 6.5° of tilt, so episodes start within ±5°; on the rig the policy engages below 5°.

**Domain randomization** (`EventCfg` in `env_cfg.py`; ranges are `dr_*` in `rig_cfg.py`):

| what | how | when |
|---|---|---|
| pole mass (+ inertia) | `randomize_rigid_body_mass`, scale | startup |
| pole length | pole COM shifted along the rod (`randomize_rigid_body_com`); the visual rod does not change | startup |
| pivot Coulomb friction, command integration (`max_accel`) | `mdp.randomize_stepper` | every reset |
| action latency, sensor-noise level, angle offset | re-drawn in the terms' own `reset()` | every reset |

## After measuring the rig

1. Edit `src/cartpole_rig/rig_cfg.py`. The USD regenerates automatically on the next run.
2. Retrain, `make play`, `make policy`, commit and push; on the Pi `git pull` in `~/physical-ai-cartpole`.
3. Make `hardware/config.toml` match `policy/policy_io.json` → `pi_config`, then run it on the rig (see
   `hardware/README.md`).

## Running on the rig

Everything for the real rig is in `hardware/`: the Pi 4 runner and web UI, the Pico firmware, the rig's
config and its track calibration. On the Pi the repo is cloned to `~/physical-ai-cartpole`:

```bash
ssh cartpole-pi 'cd ~/physical-ai-cartpole/hardware && make ui-start'   # web UI on http://<pi-address>:8000
```

The UI homes the cart on the limit switches as soon as it connects. See `hardware/README.md` for wiring,
firmware (`make flash`), calibration and the policy bring-up steps.

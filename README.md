# Cart-pole rig: balance policy (Isaac Lab 3.0)

Balance-only cart-pole matched to a real belt-driven rig, for sim-to-real transfer. Structure follows Isaac
Lab's `Isaac-Cartpole-v0` (manager-based env, RSL-RL PPO). Built against Isaac Lab 3.0.0, Isaac Sim 6.0.1,
RSL-RL 5.0.1.

- Task id: `Cartpole-Rig-Balance-v0`
- Control: 60 Hz (physics 120 Hz, decimation 2)
- Action (1): normalized motor voltage (PWM duty) in [-1, 1], turned into cart force by a DC-motor model
- Observation (4): cart position, cart velocity, pole angle, pole angular velocity (order and scaling below)
- Episode: 15 s; ends early if |pole angle| > 17 deg or |cart position| > 0.35 m

## Layout

```
src/cartpole_rig/
  rig_cfg.py               ALL physical parameters, delays, noise, reward weights, DR ranges (edit this)
  usd_builder.py           builds assets/cartpole_rig.usda from rig_cfg (pxr only): rail + end blocks, gear
                           motor, GT2 pulleys and belt, carriage with the pendulum encoder, pendulum
  asset.py                 Isaac Lab articulation + actuators; rebuilds the USD each time the scene spawns
  mdp.py                   DCMotorAction, rig_state observation (noise + delay), randomize_drive event
  env_cfg.py               manager-based env cfg (scene, actions, obs, events, rewards, terminations)
  agents/rsl_rl_ppo_cfg.py PPO, 64x64 ELU actor and critic, no observation normalizer
scripts/
  train.py                 Isaac Lab's stock rsl_rl/train.py with this task filled in
  play.py                  export policy.pt / policy.onnx / policy_io.json, then run and print statistics
  evaluate.py              stress test: one episode per env (4096), failure rate, margin, failures by parameter
  isaacsim_play.py         plain Isaac Sim player (`make run-sim`): policy + DC motor on Isaac Sim's physics steps
  viewer_ui.py             signal panel + keyboard taps (Kit UI only; used by play.py and isaacsim_play.py)
pi/
  run_policy.py            standalone 60 Hz inference stub for the Raspberry Pi (torch only)
```

## Setup (once)

```powershell
C:\isaac\IsaacLab\isaaclab.bat -p -m pip install -e C:\Users\Admin\Desktop\rnd\cartpole --no-deps
```

## Shortcuts (`make`, run in this folder)

| command | what it does |
|---|---|
| `make` / `make run-lab` | viewer + signal panel; LEFT/RIGHT tap the pole, A/D bump the cart, UP/DOWN tap strength |
| `make run-lab-random` | viewer + signal panel with random taps, as in training |
| `make run-sim` | the rig in the **plain Isaac Sim app** (no Isaac Lab): nominal parameters, newest exported `policy.pt` (or `POLICY=...`), signal panel, same keys plus **R** = reset; it resets by itself after a fall |
| `make install` | install this package into Isaac Lab's Python (once) |
| `make train` | train headless (600 iterations, about 7 min) |
| `make play` | headless: export `policy.pt` / `.onnx` / `policy_io.json` and print statistics |
| `make eval` | stress test, 4096 episodes (`PUSH=0` no taps, `PUSH=2` twice as hard) |
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
#   LEFT / RIGHT tap the pole, A / D bump the cart, UP / DOWN stronger / weaker
C:\isaac\IsaacLab\isaaclab.bat -p scripts\play.py --viz kit --num_envs 16 --steps 0 --real-time --keyboard
# With the Kit viewer, a "Rig signals (env 0)" panel shows the policy inputs (x, x_dot, theta, theta_dot as
# measured, i.e. noisy and delayed, next to the true values), the policy output (PWM duty), the motor voltage
# actually applied after the action delay, and the cart force, with 5 s plots. Its top line shows the sim
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
policy is weak, for example "fails mostly when the motor is weak". Per-episode rows go to
`<run>\eval_push<scale>.csv`.

`play.py` writes `<run>\exported\policy.pt` (TorchScript), `policy.onnx` and `policy_io.json`. The ONNX
weights are stored in a separate `policy.onnx.data` file next to it; copy both files together. It then runs
the sim with the exported `policy.pt` file itself, so the file you copy to the Pi is the one that was tested.

## Policy interface (what the Pi must do)

Input is a float32 tensor of shape (1, 4):

| idx | quantity | unit | scale (multiply) | sign convention |
|----:|---|---|---|---|
| 0 | cart position x | m | 1 / 0.35 | 0 at rail centre; + is where the cart goes for a + action |
| 1 | cart velocity | m/s | 1 / 0.7 | d/dt of idx 0 |
| 2 | pole angle theta | rad | 1 / 0.3 | 0 = upright; + when the pole leans toward +x; wrap to [-pi, pi] |
| 3 | pole angular velocity | rad/s | 1 / 3.0 | d/dt of idx 2 |

`policy_input = [x, x_dot, theta, theta_dot] * obs_scale`. Do not add an offset, clip or apply a running
normalizer. The exported model contains no normalizer (`obs_normalization=False`).

Output has shape (1, 1). Use `a = clamp(output, -1, 1)`. The output is not clipped inside the model, so you
must clamp it. `a` is the signed PWM duty: motor voltage = a × 12 V, and positive drives the cart toward +x.

`policy_io.json` holds the same numbers (taken from `rig_cfg.py` at export time), and `pi/run_policy.py`
reads it. If you change `obs_scale` you must retrain.

**Encoders on the rig.** Cart position comes from the motor encoder:
x = (counts − centre counts) / counts per output rev × 0.040 m. The pendulum encoder reads 0 with the pole
**hanging down**, but the policy wants 0 = **upright**. The Pi therefore subtracts 180° and wraps the result
to [−π, π]. `pi/run_policy.py` already does this. Home the cart at the rail centre at start-up so that
`CART_CENTRE_COUNTS` is right.

**Sign check on the rig:** with the pole held upright, command a small positive duty. The cart must move
toward +x, and x must increase (otherwise flip `CART_SIGN`). Tilt the pole toward +x: theta must go positive
(otherwise flip `POLE_SIGN`). In the viewer, +x is to the right and the motor sits on the left end block.
Accelerating the cart toward +x makes theta go negative. That is the physically correct reaction, and a
quick way to confirm both signs at once.

## Model details

**Drivetrain: motor → pulley → belt → carriage** (`mdp.DCMotorAction` and `asset.py`). The belt is treated
as rigid with no slip, so the motor shaft turns at omega = v_cart / r_pulley. Every physics step:
```
I            = (a * V_supply - Ke * omega) / R
motor torque = Kt * I
belt force   = clip(torque / r_pulley, +-11 N)
cart force   = belt force - b * v_cart - Fc * tanh(v_cart / 0.01)      (rail friction)
```
The constants are referred to the gearbox output shaft. The motor + gearbox + pulley inertia `motor_inertia`
is put on the cart joint as armature J / r², so the cart feels it as extra mass: 0.37 kg at the placeholder
1.5e-5 kg·m², almost half the cart mass. This is the biggest unknown in the model; measure it, for example
from the cart's step response at a fixed duty. The motor and idler pulleys in the scene are driven at belt
speed so you can see the drive turning; they carry no inertia of their own.

Why not a real pulley joint driving the carriage: in PhysX the only joint coupling is `PhysxMimicJointAPI`.
It accepts rotational axes only (and is deprecated), so it cannot tie a pulley joint to the prismatic
carriage joint. For a rigid belt the armature form gives the same equations of motion.
- Nominal Ke = Kt = 0.115 comes from 1000 RPM no-load at 12 V. R = 20 ohm, chosen so the stall force at 12 V
  is about 11 N; this treats the 0.7 kg·cm rating as the stall torque. With these values the cart reaches at
  most about 0.64 m/s (checked in sim). Back-EMF acts as roughly 16 N·s/m of damping. If your motor's stall
  torque is really higher, lower R.
- `deadband`: |a| below it gives 0 V. Default 0. Set it from the smallest duty that actually moves the cart.
  This is the only belt-backlash stand-in; no real backlash is modelled.

**Latency.** The policy is trained with 0–1 control steps of observation delay plus 1 step of action delay
(`obs_delay_steps`, `action_delay_steps`), re-drawn for each episode. Measure the real loop latency (encoder
read → serial → Pi → serial → PWM) and set these ranges to match.

**Noise.** Gaussian, in physical units, added before scaling: `obs_noise_std`, multiplied by a per-episode
factor in `dr_obs_noise`. On the rig the velocities come from finite differences of encoder counts, so set
the velocity noise to at least `count_size / dt / sqrt(12)`. For example, a 2400-count pendulum encoder at
60 Hz gives about 0.045 rad/s.

**Disturbances.** Every 2–4 s (`push_interval_s`), each env gets a random "tap": the pole's angular
velocity jumps by up to ±1 rad/s (`push_pole_vel`) and the cart's velocity by up to ±0.1 m/s
(`push_cart_vel`). For scale, a 0.01 N·s tap at the rod tip gives about 0.3 rad/s, and 1 rad/s is a firm
flick. The policy learns to recover from these, and to do so without running the cart into the end stops.

**Domain randomization** (`EventCfg` in `env_cfg.py`; ranges are `dr_*` in `rig_cfg.py`):

| what | how | when |
|---|---|---|
| cart mass, pole mass (+ inertia) | `randomize_rigid_body_mass`, scale | startup |
| pole length | pole COM shifted along the rod (`randomize_rigid_body_com`); the visual rod does not change | startup |
| pivot friction | pole joint damping (`randomize_actuator_gains`) | startup |
| motor + pulley inertia | cart joint armature (`randomize_joint_parameters`) | startup |
| Kt, Ke, R, cart friction | `mdp.randomize_drive` | every reset |
| obs / action latency, sensor-noise level | re-drawn in the terms' own `reset()` | every reset |

Latency and noise level are drawn inside the terms rather than by an event. Isaac Lab 3.0 has no public way
to reach an observation-term instance from an event.

## After measuring the rig

1. Edit `src/cartpole_rig/rig_cfg.py`: masses, `pole_com`, `pole_length`, motor constants, `motor_inertia`,
   friction, delays and noise. The USD regenerates automatically on the next run.
2. Retrain, run `play.py`, and copy `exported\policy.pt` and `exported\policy_io.json` to the Pi.
3. On the Pi: `python run_policy.py --policy policy.pt --io policy_io.json`. Fill in the encoder constants
   and the serial I/O in the two stub functions first.

Simple ways to measure the motor (needs no current sensor): Ke = V / omega_no_load. R = V / I_stall (measure
the current with the motor shaft held). Kt = Ke in SI units. For cart friction, apply a constant duty, log the
steady-state velocity, and fit the force balance above.

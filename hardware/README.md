# physical-ai-cartpole

Run a cartpole balancing policy trained in **Isaac Lab** on the real rig.

```
 Pendulum encoder (600 PPR quad) ─┐
 Limit switch ────────────────────┼──► Raspberry Pi Pico  (counts ticks, generates steps)
 TB6600 → NEMA17 → GT2 ◄──────────┘        │ USB serial 115200, "/dev/ttyACM0"
                                           ▼
                              Raspberry Pi 4 — this repo (policy loop)
```

Pico wiring (`src/pico_main.py`, flashed with `make flash` on the Pi):

| Pico | Goes to |
|------|---------|
| GP16 | TB6600 PUL+ |
| GP17 | TB6600 DIR+ |
| GND | TB6600 PUL-, DIR-, ENA- (ENA+ open: driver always enabled) |
| GP2 / GP3 | encoder A / B (internal pull-ups on) |

The TB6600 DIP switches are on 1/16 microstepping (3200 steps/rev), so `steps_per_meter = 80000`;
change both together. No limit switch is wired, so the Pico always reports it as open.

## Layout

```
physical-ai-cartpole/
├── Makefile               # make help
├── config.toml.template   # copy to config.toml (make config)
├── requirements.txt
├── models/                # put policy.onnx / policy.npz here (placeholder)
├── logs/                  # one CSV per run
└── src/
    ├── main.py            # CLI: serial | sensors | dry-run | run
    ├── config.py          # TOML -> typed, validated config
    ├── pico_link.py       # serial protocol with the Pico
    ├── state.py           # counts -> (x, theta, dx, dtheta), 0 rad = upright
    ├── interface.py       # sim<->real contract: observation order, action -> step rate
    ├── policy.py          # ONNX or numpy MLP inference
    ├── safety.py          # engage/disengage, soft track limit, limit switch, lost link
    ├── logger.py          # per-tick CSV
    ├── controller.py      # fixed-rate loops + BalanceController (one control tick)
    ├── hardware_service.py# UI backend: loop thread owning the Pico, commands, sample buffer
    ├── ui_server.py       # stdlib HTTP server: page, SSE telemetry, POST commands
    ├── ui.html            # the browser UI (no external dependencies)
    ├── fake_pico.py       # simulated Pico + pendulum for --fake
    ├── pico_main.py       # Pico firmware (MicroPython): encoder + PIO step/dir for a TB6600
    ├── pico_main_l298n.py # firmware for an L298N (PIO coil patterns on GP14-17)
    ├── pico_main_hat.py   # firmware for a PWM+DIR dual H-bridge HAT (untested)
    └── tests/             # hardware-free unit tests (make test)
```

## Web UI

```bash
make ui-remote   # Mac: deploys, starts the UI on the Pi + SSH tunnel -> http://localhost:8000
make ui-fake     # Mac: same UI against a simulated Pico (no hardware)
make pi-ui       # Mac: deploy + (re)start the UI in the background on the Pi
                 #      -> http://cartpole-pi.local:8000 from any machine on the LAN
make ui          # Pi: run the server in the foreground (Ctrl+C stops it)
make ui-stop     # Pi: stop the background UI (motor commanded to 0)
```

- **Live data:** θ, θ̇, x, ẋ, raw ticks and steps, the limit switch and the step rate sent, as 10-second charts.
  Also a cart/pole drawing to check signs, and loop health (actual Hz, late ticks, missed replies).
- **Control:** hold-to-jog (buttons or ← →) with a speed slider, reset counters, dry run, and LIVE
  (needs the *Armed* checkbox), clear fault, and CSV recording. **E-STOP** is the button, `Space` or `Esc`.
- **Safety:** a jog only lasts `jog_timeout_s` unless the browser keeps repeating it, so a closed
  tab, lost focus or dropped network stops the cart. Past the soft limit or on the limit switch, a jog only
  moves back toward centre. The server listens on `0.0.0.0:8000` with **no login**, so anyone on the LAN can drive the motor.
  On an untrusted network, set `host = "127.0.0.1"` and use `make ui-remote`.
- The UI and the CLI commands can't run at the same time, since both need the serial port.

## Pico protocol

One request, one reply, newline-terminated ASCII:

| Pi → Pico | Meaning |
|---|---|
| `R` | reset encoder and step counters to 0 |
| `<int>` | signed step rate in half-steps/s, e.g. `-1250` |

Reply: `<encoder_ticks>,<cart_steps>,<limit_hit>`.

> **Pico firmware should include a watchdog:** if no command arrives for ~100 ms,
> stop stepping. If the Pi crashes, the Pico otherwise keeps running at the last rate.

## Bring-up, in order

```bash
# Mac
make deploy                 # rsync to cartpole-pi:~/physical-ai-cartpole

# Pi (make ssh)
make venv config            # once
make serial                 # 1. does the Pico answer?
make sensors                # 2. move by hand: check signs, units, zero
make dry-run                # 3. policy on real sensors, motor off
make run                    # 4. motor on
```

At step 2, change `config.toml` until all of these are true:

- the pole hanging down reads about ±180°, and held upright reads 0°;
- the pole tilting in the sim's +θ direction reads positive (otherwise set `encoder_sign = -1`);
- jogging the cart toward the sim's +x direction reads positive (otherwise set `cart_sign = -1`);
- jogging until x reads 0.100 moves the cart 10 cm on a ruler (otherwise fix `steps_per_meter`).

The cart position has **no sensor**: x is counted from the steps the Pico commands.
Pushing the cart by hand doesn't change it, and skipped steps make it drift.
Check it with jogs, not by hand. Re-centre the cart and reset the counters after moving it by hand.

## Matching the training setup

These values in `config.toml` must match the Isaac Lab environment exactly:

| Key | What it must equal |
|---|---|
| `policy.rate_hz` | `1 / (sim.dt * decimation)`. The Isaac Lab default is 1/120 × 2 = 60 Hz. |
| `policy.obs_order` | The order of the observation vector in training. |
| `policy.action_type` / `action_scale` | What the action means physically, and its scale. |

The default Isaac Lab cartpole action is a **force**, but a stepper motor follows a speed command.
`action_type = "force"` only approximates force as `a = F / force_mass_kg`.
For a reliable transfer, train with an action the stepper can follow: cart **velocity** or **acceleration**.
Also cap it to the real speed limit (`max_step_rate / steps_per_meter`, 0.4 m/s by default).
The real loop has **one control period of action delay**, so model that in the sim too.

## Safety

- The policy only drives the motor while |θ| < `engage_angle_deg`. It lets go above `disengage_angle_deg`.
- These faults latch until restart: the limit switch, |x| > `track_limit_m`, or more than `max_missed_reads` missed replies in a row.
- Ctrl+C and SIGTERM both send `0` to the Pico before exiting.

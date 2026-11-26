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
| GP14 / GP15 | encoder A / B (internal pull-ups on; index C/Z not connected) |
| GP13 / GP1 | left / right limit switch, COM + NC to GND (internal pull-up; pressed or broken wire = high) |

The TB6600 DIP switches are on 1/8 microstepping (1600 steps/rev: S1 off, S2 on, S3 off), so
`steps_per_meter = 40000` and the 20 kHz step limit is 0.5 m/s; change both together, scale `START_HZ` and
`ACCEL` in `pico_main.py` (they are in steps), and run Calibrate track again. The Pico never steps toward a pressed limit switch (moving away is allowed), and
negative steps must move the cart toward the left switch.

## Layout

```
hardware/                  # on the Pi: ~/physical-ai-cartpole/hardware
├── Makefile               # make help
├── config.toml            # this rig's config (committed)
├── config.toml.template   # every key, documented
├── track_calibration.json # measured track travel (Calibrate track writes it)
├── mcp_tokens.json        # MCP access tokens, hashed (made in the UI, not in git)
├── requirements.txt
├── models/                # put policy.onnx / policy.npz here (placeholder)
├── logs/                  # one CSV per run, ui.log
├── pico-backup/           # the Pico's earlier main.py files, for reference
└── src/
    ├── main.py            # CLI: serial | sensors | dry-run | run
    ├── config.py          # TOML -> typed, validated config
    ├── pico_link.py       # serial protocol with the Pico
    ├── state.py           # counts -> (x, theta, dx, dtheta), 0 rad = upright
    ├── interface.py       # sim<->real contract: observation order, action -> step rate
    ├── policy.py          # ONNX or numpy MLP inference
    ├── safety.py          # engage/disengage, limit switch, lost link
    ├── logger.py          # per-tick CSV
    ├── controller.py      # fixed-rate loops + BalanceController (one control tick)
    ├── calibration.py     # home on both limit switches, measure the track, x = 0 at the centre
    ├── hardware_service.py# UI backend: loop thread owning the Pico, commands, sample buffer
    ├── ui_server.py       # stdlib HTTP server: page, SSE telemetry, POST commands
    ├── ui.html            # the browser UI (no external dependencies)
    ├── mcp_protocol.py    # MCP JSON-RPC messages (both protocol eras), no I/O
    ├── mcp_tools.py       # the MCP tools: wrappers over hardware_service, scope tiers
    ├── mcp_server.py      # POST /mcp: auth, origin check, confirmation, audit log + activity
    ├── mcp_tokens.py      # named, scoped, expiring access tokens (hashed JSON file)
    ├── fake_pico.py       # simulated Pico + pendulum for --fake
    ├── pico_main.py       # Pico firmware (MicroPython): encoder + PIO step/dir for a TB6600
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
- **Calibrate track:** drives to the left limit switch, then the right one, twice (at the jog speed), then
  parks in the middle and makes that x = 0. The travel between the switches (in steps, and in metres at
  `steps_per_meter`) and the two passes' spread are shown and saved to `track_calibration.json`; the
  drawing then shows the real rail. If a tape measure disagrees with the length, fix `steps_per_meter`
  (= travel steps / real metres). Run it again after changing the rail, pulley or microstepping.
- **Home:** left switch, then half the saved travel, so x = 0 is the centre (a full calibration if none is
  saved). With `[ui] home_on_connect = true` it runs **by itself every time the Pico connects**, so the cart
  moves when the UI starts; E-STOP aborts it. Once homed, *Reset counters* zeroes θ only.
- **Safety:** a jog only lasts `jog_timeout_s` unless the browser keeps repeating it, so a closed
  tab, lost focus or dropped network stops the cart. There is no soft limit: the Pico stops the cart at
  either limit switch and only lets it move away from a pressed one. The server listens on `0.0.0.0:8000` with **no login**, so anyone on the LAN can drive the motor.
  On an untrusted network, set `host = "127.0.0.1"` and use `make ui-remote`.
- The UI and the CLI commands can't run at the same time, since both need the serial port.

## MCP (AI agents)

The UI server also answers MCP at **`POST /mcp`** (Streamable HTTP, stateless JSON; protocol 2024-11-05 to
2026-07-28), so an agent such as Claude Code can watch and drive the rig. It shares the UI's
`HardwareService`: an agent's `emergency_stop` stops a jog from the page and vice versa. Same design as
I-Vision's MCP server: hand-written JSON-RPC, no extra dependencies.

**AI agents** in the UI's top bar (or System tab, or `http://cartpole-pi.local:8000/#agents`) opens a dialog
like I-Vision's Agent access page:

- **Access tokens:** the server URL; create a token with a name, access (*Read only* or *Read and move*) and
  expiry. It is shown once, with the `claude mcp add ...` command to copy. The list shows each token's
  access, created / last used / expiry, and Revoke.
- **Activity:** every tool call, newest first and updating live, filterable by result (*Done*, *Asked you
  first*, *Declined by you*, *Refused*, with the reason), action and token, plus search.
- While an agent is calling tools, an **Agent** chip appears in the top bar (click it for the activity).

From the Mac without the UI: `make mcp-token NAME=claude-code SCOPE=operate` creates a token and prints it.

| tool | access | what it does |
|---|---|---|
| `get_status` | read | link, control mode, faults, loop health, track/homing, latest sample (θ in degrees), next-step hint |
| `get_rig_info` | read | control rate, engage/disengage angles, max speed, policy and observation config |
| `get_telemetry` | read | last N s of 60 Hz samples, downsampled, plus max/mean \|θ\|, x range, balancing fraction |
| `emergency_stop` | read | stop the motor, policy, jog and homing; every token can call it |
| `home` / `calibrate_track` | operate, moves | the UI's Home / Calibrate track |
| `jog` | operate, moves | `velocity_mps` for `duration_s` (≤ `max_jog_s`), then stops by itself |
| `start_balancing` | operate, moves | policy with the motor on (engages once the pole is held near upright) |
| `start_dry_run` | operate | policy on live sensors, motor off |
| `reset_counters`, `clear_fault`, `set_recording`, `connect_pico`, `disconnect_pico` | operate | as in the UI (`connect_pico` homes if `home_on_connect`) |
| `lift_pole` | operate | `--fake` only: put the simulated pole near upright |

`[mcp]` in `config.toml`:

- **Tokens** (`tokens_file`, default `mcp_tokens.json`): only a SHA-256 of each is stored (mode 0600, not in
  git, kept by `make deploy`). Agents send `Authorization: Bearer cpmcp_...`. Requests carrying a browser
  `Origin` are refused unless listed in `allowed_origins`, so a web page on the LAN can't call `/mcp`.
- **`require_confirmation = true`:** tools marked *moves* don't run on the first call. The reply tells the
  model a confirmation is pending and gives the host a 5-minute token in `result._meta["cartpole/confirm"]`
  (an HMAC over that token, tool and exact arguments); the host asks the person and repeats the call with
  it in `params._meta`. 2026-07-28 clients that can show forms get a native elicitation instead. Off by
  default, as in I-Vision; Claude Code already asks before each tool call unless you allow the tool.
- **Audit** (`audit_file`): one JSON line per call, kept 90 days; the Activity tab reads it.

The web UI itself still has no login, so anyone who can open it can also create a token, just as they can
drive the motor from it. The tokens keep programs out of `/mcp` unless someone at the UI let them in.

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
# Pi: the code is a clone of the repo; work in ~/physical-ai-cartpole/hardware
make venv                   # once
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
| `policy.obs_order` / `obs_scale` | The order of the observation vector in training, and its scale. |
| `policy.action_type` / `action_scale` | What the action means physically, and its scale. |
| `safety.max_cart_vel_mps` | The speed the policy was trained with (`RIG.max_speed`). |

This repo's policy (`../policy/`, trained by `../src/cartpole_rig`) outputs a cart **acceleration**
(`action_type = "acceleration"`, `action_scale = RIG.max_accel`); the Pi integrates it into the speed command,
and the env models exactly that, the one control period of action delay, the Pico ramp, and this estimator
(step / encoder quantization, `velocity_alpha` filter). `../policy/policy_io.json` lists the `pi_config`
values a policy was trained for; `config.toml` must match them.

## Safety

- The policy only drives the motor while |θ| < `engage_angle_deg`. It lets go above `disengage_angle_deg`.
- These faults latch until restart: a limit switch, or more than `max_missed_reads` missed replies in a row.
- Ctrl+C and SIGTERM both send `0` to the Pico before exiting.

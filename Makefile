# Cart-pole rig shortcuts (GNU Make on Windows; recipes run in cmd.exe).
#   make            open the viewer: trained policy + signal panel + keyboard taps
#   make run-sim    same rig in the plain Isaac Sim app (no Isaac Lab), keyboard taps, R = reset
#   make pi-demo    the Pi controller + its live web page on this PC, with a simulated rig
#   make help       list all targets
# Variables: ISAACLAB (Isaac Lab folder), ENVS (rigs shown), CKPT (checkpoint, default = latest run),
#   PUSH (tap strength for eval), POLICY (exported policy.pt for `make run-sim`, default = newest),
#   e.g.  make run-lab ENVS=4  /  make eval PUSH=0.5

SHELL := cmd.exe
ISAACLAB ?= C:\isaac\IsaacLab
ENVS ?= 16
CKPT ?=
PUSH ?= 1
POLICY ?=
PORT ?= 8000

RUN := "$(ISAACLAB)\isaaclab.bat" -p
CKPT_ARG := $(if $(CKPT),--checkpoint $(CKPT),)
# newest exported policy folder (written by `make play`)
EXPORTED := $(lastword $(sort $(wildcard logs/rsl_rl/cartpole_rig/*/exported)))

.PHONY: run-lab run-lab-random run-sim pi-demo help install train play eval

run-lab: ## viewer with signal panel; LEFT/RIGHT tap the pole, A/D bump the cart, UP/DOWN strength
	$(RUN) scripts\play.py --viz kit --num_envs $(ENVS) --steps 0 --real-time --keyboard $(CKPT_ARG)

run-lab-random: ## viewer with signal panel and random taps (as in training)
	$(RUN) scripts\play.py --viz kit --num_envs $(ENVS) --steps 0 --real-time $(CKPT_ARG)

run-sim: ## plain Isaac Sim app (no Isaac Lab): nominal rig + exported policy.pt; arrows/A/D taps, R reset
	set "CARTPOLE_ROOT=$(CURDIR)" && set "CARTPOLE_POLICY=$(POLICY)" && "$(ISAACLAB)\_isaac_sim\isaac-sim.bat" --exec "$(CURDIR)/scripts/isaacsim_play.py"

pi-demo: ## Pi controller with web page (http://localhost:$(PORT)) and a simulated rig; Ctrl+C to stop
	$(RUN) pi/run_policy.py --policy $(EXPORTED)/policy.pt --io $(EXPORTED)/policy_io.json --web --fake --port $(PORT)

help: ## list targets
	@findstr /R /C:"^[a-z-]*:.*##" Makefile

install: ## install this package into Isaac Lab's Python (once)
	$(RUN) -m pip install -e . --no-deps

train: ## train headless (600 iterations, ~7 min); logs go to logs\rsl_rl\cartpole_rig
	$(RUN) scripts\train.py --headless

play: ## headless: export policy.pt / policy.onnx / policy_io.json and print 30 s of statistics
	$(RUN) scripts\play.py --headless $(CKPT_ARG)

eval: ## headless stress test, 4096 episodes (PUSH=0 no taps, PUSH=2 twice as hard)
	$(RUN) scripts\evaluate.py --headless --push_scale $(PUSH) $(CKPT_ARG)

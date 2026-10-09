# Cart-pole rig shortcuts (GNU Make on Windows; recipes run in cmd.exe).
#   make            open the viewer: trained policy + signal panel + keyboard taps
#   make help       list all targets
# Variables: ISAACLAB (Isaac Lab folder), ENVS (rigs shown), CKPT (checkpoint, default = latest run),
#   PUSH (tap strength for eval), e.g.  make gui ENVS=4  /  make eval PUSH=0.5

SHELL := cmd.exe
ISAACLAB ?= C:\isaac\IsaacLab
ENVS ?= 16
CKPT ?=
PUSH ?= 1

RUN := "$(ISAACLAB)\isaaclab.bat" -p
CKPT_ARG := $(if $(CKPT),--checkpoint $(CKPT),)

.PHONY: gui gui-random help install train play eval

gui: ## viewer with signal panel; LEFT/RIGHT tap the pole, A/D bump the cart, UP/DOWN strength
	$(RUN) scripts\play.py --viz kit --num_envs $(ENVS) --steps 0 --real-time --keyboard $(CKPT_ARG)

gui-random: ## viewer with signal panel and random taps (as in training)
	$(RUN) scripts\play.py --viz kit --num_envs $(ENVS) --steps 0 --real-time $(CKPT_ARG)

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

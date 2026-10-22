# physical-ai-cartpole — run `make help` for targets.
# Typical flow: on the Mac `make deploy`, then on the Pi `make venv config sensors`.

PI_HOST ?= cartpole-pi
PI_DIR  ?= ~/physical-ai-cartpole
CONFIG  ?= config.toml
VENV    ?= .venv
PYTHON  := $(VENV)/bin/python
CLI     := $(PYTHON) -u src/main.py --config $(CONFIG)
UI_PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv config test flash serial sensors dry-run run ui ui-start ui-stop ui-fake ui-remote pi-ui deploy ssh clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-9s %s\n", $$1, $$2}'

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@touch $@

venv: $(VENV)/.installed ## Create .venv and install requirements

$(CONFIG):
	cp config.toml.template $(CONFIG)
	@echo "Created $(CONFIG) — edit it for your rig."

config: $(CONFIG) ## Create config.toml from the template (never overwrites)

test: venv ## Run the hardware-free unit tests
	PYTHONPATH=src $(PYTHON) -m unittest discover -s src/tests -t src -v

serial: venv $(CONFIG) ## Pi: reset counters and print raw Pico replies
	$(CLI) serial

sensors: venv $(CONFIG) ## Pi: live state with the motor off (check signs and zero)
	$(CLI) sensors

dry-run: venv $(CONFIG) ## Pi: run the policy on live sensors, motor off
	$(CLI) dry-run

run: venv $(CONFIG) ## Pi: balance with the policy (motor ON)
	$(CLI) run

ui: venv $(CONFIG) ## Pi: web UI on 0.0.0.0:8000 (live data, jog, policy)
	$(CLI) ui

ui-start: venv $(CONFIG) ## Pi: start the UI in the background (log: logs/ui.log)
	@mkdir -p logs
	@if [ -f logs/ui.pid ] && kill -0 $$(cat logs/ui.pid) 2>/dev/null; then \
		echo "UI already running (pid $$(cat logs/ui.pid))"; \
	else \
		setsid nohup $(CLI) ui > logs/ui.log 2>&1 < /dev/null & echo $$! > logs/ui.pid; \
		sleep 3; cat logs/ui.log; echo "pid $$(cat logs/ui.pid)"; \
	fi

ui-stop: ## Pi: stop the background UI (SIGTERM -> motor commanded to 0)
	@if [ -f logs/ui.pid ] && kill -0 $$(cat logs/ui.pid) 2>/dev/null; then \
		pid=$$(cat logs/ui.pid); kill -TERM $$pid; \
		for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $$pid 2>/dev/null || break; sleep 0.3; done; \
		echo "UI stopped (pid $$pid)"; \
	else \
		echo "UI not running"; \
	fi
	@rm -f logs/ui.pid

ui-fake: venv $(CONFIG) ## Mac: web UI against a simulated Pico — no hardware needed
	$(CLI) --fake ui

ui-remote: deploy ## Mac: deploy, run the UI on the Pi, tunnel it to http://localhost:8000
	@echo "Open http://localhost:$(UI_PORT) — Ctrl+C here stops the UI and the motor."
	ssh -t -L $(UI_PORT):localhost:$(UI_PORT) $(PI_HOST) 'cd $(PI_DIR) && make ui'

pi-ui: deploy ## Mac: deploy, then (re)start the UI on the Pi -> http://cartpole-pi.local:8000
	ssh $(PI_HOST) 'cd $(PI_DIR) && make --no-print-directory ui-stop ui-start'

PICO_PORT ?= /dev/ttyACM0
MPREMOTE  ?= $(HOME)/.local/bin/mpremote

flash: ## Pi: copy src/pico_main.py to the Pico as main.py and restart it (stop the UI first)
	$(MPREMOTE) connect $(PICO_PORT) fs cp src/pico_main.py :main.py + reset

deploy: ## Mac: copy the project to the Pi (keeps the Pi's config.toml, logs/ and venv)
	rsync -av --delete --exclude .venv --exclude config.toml --exclude 'logs/' \
		--exclude __pycache__ --exclude .git ./ $(PI_HOST):$(PI_DIR)/

ssh: ## Mac: open a shell in the project folder on the Pi
	ssh -t $(PI_HOST) 'cd $(PI_DIR) && exec $$SHELL -l'

clean: ## Remove the venv and caches
	rm -rf $(VENV)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

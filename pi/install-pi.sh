#!/usr/bin/env bash
# Install (or update) the cart-pole controller on a Raspberry Pi 4 running 64-bit Raspberry Pi OS.
#
#   sudo ./install-pi.sh <exported-dir>
#
# <exported-dir> holds policy.pt and policy_io.json (from `make play` on the training PC, folder
# logs/rsl_rl/cartpole_rig/<run>/exported; copy it to the Pi first, e.g. with scp).
# Re-run it any time to update the code or the policy; it restarts the service.
#
# What it does: installs python3-venv, creates the system user "cartpole" (in group dialout for the Pico's
# serial port), copies this folder's controller files and the policy to /opt/cartpole, creates a venv there
# with torch + pyserial, writes /etc/default/cartpole (only the first time), installs and starts
# cartpole.service. Then open http://<pi-address>:8000.
set -euo pipefail

APP=/opt/cartpole
SERVICE=cartpole
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=(run_policy.py web_server.py web_ui.html fake_rig.py)

die() { echo "error: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo: sudo $0 <exported-dir>"
[[ $# -eq 1 ]] || die "usage: sudo $0 <exported-dir>   (folder with policy.pt and policy_io.json)"
EXPORTED="$(cd "$1" && pwd)" || die "no such folder: $1"
for f in policy.pt policy_io.json; do [[ -f "$EXPORTED/$f" ]] || die "missing $EXPORTED/$f"; done
for f in "${FILES[@]}" cartpole.service; do [[ -f "$SRC/$f" ]] || die "missing $SRC/$f (run from the repo's pi/ folder)"; done
[[ "$(uname -m)" == "aarch64" ]] || die "needs 64-bit Raspberry Pi OS (aarch64) for torch; this is $(uname -m)"

echo "==> packages"
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip

echo "==> user '$SERVICE' (group dialout for the serial port)"
if ! id "$SERVICE" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE"
fi
usermod -aG dialout "$SERVICE"

echo "==> files to $APP"
systemctl stop "$SERVICE" 2>/dev/null || true  # motor off (SIGINT -> PWM 0) before replacing files
install -d -m 755 "$APP"
for f in "${FILES[@]}"; do install -m 644 "$SRC/$f" "$APP/$f"; done
install -m 644 "$EXPORTED/policy.pt" "$APP/policy.pt"
install -m 644 "$EXPORTED/policy_io.json" "$APP/policy_io.json"

echo "==> python venv (torch is large: the first install takes a few minutes)"
[[ -x "$APP/venv/bin/python" ]] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q torch pyserial
"$APP/venv/bin/python" -c "import torch; print('    torch', torch.__version__)"

echo "==> service"
if [[ ! -f /etc/default/$SERVICE ]]; then
    cat > /etc/default/$SERVICE <<'EOF'
# Arguments for run_policy.py (cartpole.service). After editing: sudo systemctl restart cartpole
#   --web          live page + Start/Stop/E-STOP (keep this: without it the controller starts at once)
#   --fake         simulated rig instead of the Pico (for testing the Pi without hardware)
#   --port 8000    web page port
CARTPOLE_ARGS="--web"
EOF
fi
install -m 644 "$SRC/cartpole.service" /etc/systemd/system/$SERVICE.service
systemctl daemon-reload
systemctl enable --now "$SERVICE"
sleep 3
systemctl --no-pager --lines=5 status "$SERVICE" || true

ADDR="$(hostname -I | awk '{print $1}')"
echo
echo "Installed. Page: http://${ADDR:-<pi-address>}:8000  (motor stays off until you press Start)"
echo "Logs: journalctl -u $SERVICE -f    Stop: sudo systemctl stop $SERVICE    Options: /etc/default/$SERVICE"

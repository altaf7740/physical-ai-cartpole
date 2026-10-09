"""Train the rig balance policy with Isaac Lab's stock RSL-RL trainer.

Run from this project's folder (logs go to ./logs/rsl_rl/cartpole_rig):
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts\\train.py --headless [--num_envs N] [--max_iterations N]
All other arguments are passed through to scripts/reinforcement_learning/rsl_rl/train.py.
"""

import os
import runpy
import sys

ISAACLAB = os.environ.get("ISAACLAB_PATH", r"C:\isaac\IsaacLab")
STOCK = os.path.join(ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl", "train.py")

sys.argv = [STOCK, "--task", "Cartpole-Rig-Balance-v0", "--external_callback", "cartpole_rig.register", *sys.argv[1:]]
sys.path.insert(0, os.path.dirname(STOCK))  # for its local `import cli_args`
runpy.run_path(STOCK, run_name="__main__")

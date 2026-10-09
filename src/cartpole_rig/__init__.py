"""Balance-only cart-pole task for a real belt-driven rig (Isaac Lab 3.0, manager-based, RSL-RL PPO).

Importing this package registers the gym environment. With Isaac Lab's stock scripts pass
``--external_callback cartpole_rig.register``; the scripts in ../../scripts do that for you.
"""

import sys

import gymnasium as gym

from . import agents

TASK = "Cartpole-Rig-Balance-v0"

gym.register(
    id=TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CartpoleRigEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpoleRigPPORunnerCfg",
    },
)


def register() -> list[str]:
    """Hook for Isaac Lab's ``--external_callback``; returns the CLI args it did not consume (all of them)."""
    return sys.argv[1:]

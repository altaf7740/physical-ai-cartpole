"""Cart-pole rig articulation for Isaac Lab, built from rig_cfg.RIG (geometry and joints: see usd_builder.py).

The USD is rewritten every time the scene is spawned, so it always matches the config. (It is written at spawn
time, not on import, because importing ``pxr`` before Kit starts breaks Kit's own USD libraries.)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .rig_cfg import RIG

USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "cartpole_rig.usda")


def spawn_rig(prim_path, cfg, *args, **kwargs):
    """Write the USD from RIG, then spawn it like a normal UsdFileCfg."""
    from .usd_builder import write_usd

    write_usd(RIG, cfg.usd_path)
    return sim_utils.spawn_from_usd(prim_path, cfg, *args, **kwargs)


CARTPOLE_RIG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_rig,
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_angular_velocity=1000.0, enable_gyroscopic_forces=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3), joint_pos={"cart": 0.0, "pole": 0.0, ".*_pulley": 0.0}
    ),
    actuators={
        # the stepper: a stiff velocity drive, target = stepper speed (set by StepperAction every physics step)
        "cart": ImplicitActuatorCfg(
            joint_names_expr=["cart"], effort_limit_sim=200.0, stiffness=0.0, damping=RIG.cart_drive_damping
        ),
        # passive pivot: zero-velocity-target damping == viscous bearing friction; StepperAction adds the Coulomb
        # friction as an effort
        "pole": ImplicitActuatorCfg(
            joint_names_expr=["pole"], effort_limit_sim=100.0, stiffness=0.0, damping=RIG.pivot_viscous
        ),
        # visual pulleys: velocity-tracking drive, target = belt speed / r (set by DCMotorAction)
        "pulleys": ImplicitActuatorCfg(
            joint_names_expr=[".*_pulley"], effort_limit_sim=1.0, stiffness=0.0, damping=1e-4
        ),
    },
)

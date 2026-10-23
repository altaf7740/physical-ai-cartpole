"""Balance-only cart-pole env for the real rig. Structure follows Isaac Lab's Isaac-Cartpole-v0
(manager_based/classic/cartpole); all numbers come from rig_cfg.RIG."""

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.classic.cartpole.mdp import joint_pos_target_l2

from . import mdp
from .asset import CARTPOLE_RIG_CFG
from .rig_cfg import RIG

CART = SceneEntityCfg("robot", joint_names=["cart"])
POLE = SceneEntityCfg("robot", joint_names=["pole"])


@configclass
class RigSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = CARTPOLE_RIG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    )


@configclass
class ActionsCfg:
    motor = mdp.StepperActionCfg(
        asset_name="robot",
        joint_name="cart",
        max_speed=RIG.max_speed,
        max_accel=RIG.max_accel,
        pico_accel=RIG.pico_accel,
        pico_start_speed=RIG.pico_start_speed,
        pulley_radius=RIG.pulley_radius,
        pivot_coulomb_torque=RIG.pivot_coulomb_accel * RIG.pole_pivot_inertia,
        delay_steps=RIG.action_delay_steps,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [x, x_dot, theta, theta_dot] * obs_scale  (order matters: the Pi builds the same vector)
        state = ObsTerm(
            func=mdp.rig_state,
            params={
                "noise_std": RIG.obs_noise_std,
                "noise_scale_range": RIG.dr_obs_noise,
                "delay_steps": RIG.obs_delay_steps,
                "steps_per_meter": RIG.steps_per_meter,
                "encoder_cpr": RIG.encoder_cpr,
                "alpha": RIG.velocity_alpha,
                "angle_offset": RIG.angle_offset,
            },
            scale=RIG.obs_scale,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False  # quantization, filter and noise are applied inside rig_state
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # --- startup: body / joint properties (CPU-side writes, done once) ---
    pole_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": RIG.dr_pole_mass,
            "operation": "scale",
        },
    )
    pole_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pole"]), "com_range": {"z": RIG.dr_pole_com}},
    )
    pivot_friction = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={"asset_cfg": POLE, "damping_distribution_params": RIG.dr_pivot_friction, "operation": "scale"},
    )

    # --- reset: command integration + pivot Coulomb friction (latency, sensor noise and the angle offset are
    # re-sampled in the terms' reset) ---
    stepper = EventTerm(
        func=mdp.randomize_stepper,
        mode="reset",
        params={"action_name": "motor", "accel_range": RIG.dr_max_accel, "friction_range": RIG.dr_pivot_friction},
    )

    # --- interval: disturbance taps on the pole ---
    push = EventTerm(
        func=mdp.push_pole,
        mode="interval",
        interval_range_s=RIG.push_interval_s,
        params={"pole_vel": RIG.push_pole_vel},
    )

    # --- reset: initial state ---
    reset_cart = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": CART,
            "position_range": (-RIG.reset_cart_pos, RIG.reset_cart_pos),
            "velocity_range": (-RIG.reset_cart_vel, RIG.reset_cart_vel),
        },
    )
    reset_pole = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": POLE,
            "position_range": (-RIG.reset_pole_angle, RIG.reset_pole_angle),
            "velocity_range": (-RIG.reset_pole_vel, RIG.reset_pole_vel),
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=base_mdp.is_alive, weight=RIG.w_alive)
    pole_angle = RewTerm(func=joint_pos_target_l2, weight=-RIG.w_pole_angle, params={"asset_cfg": POLE, "target": 0.0})
    cart_pos = RewTerm(func=joint_pos_target_l2, weight=-RIG.w_cart_pos, params={"asset_cfg": CART, "target": 0.0})
    cart_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-RIG.w_cart_vel, params={"asset_cfg": CART})
    action = RewTerm(func=base_mdp.action_l2, weight=-RIG.w_action)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-RIG.w_action_rate)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=base_mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": CART, "bounds": (-RIG.cart_limit, RIG.cart_limit)},
    )
    pole_fallen = DoneTerm(
        func=base_mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": POLE, "bounds": (-RIG.max_pole_angle, RIG.max_pole_angle)},
    )


@configclass
class CartpoleRigEnvCfg(ManagerBasedRLEnvCfg):
    scene: RigSceneCfg = RigSceneCfg(num_envs=4096, env_spacing=1.5, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = RIG.decimation
        self.episode_length_s = RIG.episode_s
        # front view of env 0 from above the rail, so the belt and motor behind it are visible
        self.viewer.eye = (0.3, -1.25, 0.95)
        self.viewer.lookat = (0.0, 0.0, 0.55)
        self.viewer.origin_type = "env"
        self.sim.dt = RIG.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()

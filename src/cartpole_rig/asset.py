"""Cart-pole rig articulation, generated from rig_cfg.RIG.

The USD is rewritten every time the scene is spawned, so it always matches the config. (It is written at spawn
time, not on import, because importing ``pxr`` before Kit starts breaks Kit's own USD libraries.)

Front view (-y toward the viewer, so +x is to the right; +z up):
  - rail (fixed to the world): 2020 extrusion + MGN12 rail, olive end blocks, gear motor on the left block
  - joint ``cart`` (prismatic X): carriage, plate, belt clamp, encoder bracket + encoder housing
  - joint ``pole`` (revolute Y, on the encoder shaft): steel rod, built pointing +Z so angle 0 is upright
  - joints ``motor_pulley`` / ``idler_pulley`` (revolute Y): GT2 pulleys; the belt loop runs between them
    behind the rail and its top strand is clamped to the carriage

The belt is modelled as rigid: the motor torque reaches the carriage as force = torque / r (see DCMotorAction)
and the motor + pulley inertia sits on the ``cart`` joint as armature J / r^2. The two pulley links are visual
(near-zero inertia) and are spun at belt speed by DCMotorAction. No colliders: nothing touches anything.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .rig_cfg import RIG

USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "cartpole_rig.usda")

# visual geometry [m] (dynamics only use the masses / inertias / joint frames below)
CARRIAGE_LEN = 0.045  # MGN12H block length
BELT_Y = 0.025  # belt plane, behind the rail
PULLEY_Z = -0.004  # pulley centre height (top belt strand at PULLEY_Z + r, just under the cart plate)
SHAFT_Y, SHAFT_Z = -0.072, 0.04  # pendulum pivot on the cart: encoder shaft tip, in front of the carriage

STEEL, ALU, DARK, BLACK = (0.75, 0.75, 0.78), (0.82, 0.83, 0.85), (0.25, 0.25, 0.27), (0.08, 0.08, 0.08)
OLIVE, BLUE, RED = (0.5, 0.5, 0.0), (0.2, 0.45, 0.8), (0.85, 0.1, 0.1)


def write_usd(path: str = USD_PATH) -> str:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    def box(path, size, pos, rgb):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*pos))
        c.AddScaleOp().Set(Gf.Vec3f(*size))
        c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def cyl(path, radius, height, pos, rgb, axis="Y"):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(height)
        c.CreateAxisAttr(axis)
        c.AddTranslateOp().Set(Gf.Vec3d(*pos))
        c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def body(path, pos, mass, inertia, com=(0, 0, 0)):
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        return x

    def joint(kind, name, parent, child, pos0, axis):
        j = kind.Define(stage, f"/cartpole/joints/{name}")
        j.CreateBody0Rel().SetTargets([parent.GetPath()])
        j.CreateBody1Rel().SetTargets([child.GetPath()])
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        return j

    r = RIG.pulley_radius
    stop = RIG.cart_limit + RIG.end_stop_margin
    rail_len = 2 * stop + CARRIAGE_LEN + 0.02
    px = rail_len / 2 + 0.02  # pulley / end block x

    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/cartpole")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # --- rail assembly (static) ---
    rail = body("/cartpole/rail", (0, 0, 0), 5.0, (0.1, 0.1, 0.1))
    box("/cartpole/rail/extrusion", (rail_len, 0.02, 0.02), (0, 0, -0.024), ALU)
    box("/cartpole/rail/mgn12", (rail_len, 0.012, 0.008), (0, 0, -0.010), DARK)
    for name, sx in (("end_left", -1), ("end_right", 1)):
        box(f"/cartpole/rail/{name}", (0.04, 0.05, 0.07), (sx * px, -0.01, -0.02), OLIVE)
    box("/cartpole/rail/belt_top", (2 * px, 0.006, 0.0015), (0, BELT_Y, PULLEY_Z + r), BLACK)
    box("/cartpole/rail/belt_bottom", (2 * px, 0.006, 0.0015), (0, BELT_Y, PULLEY_Z - r), BLACK)
    cyl("/cartpole/rail/motor", 0.0125, 0.05, (-px, 0.06, PULLEY_Z), DARK)  # 25 mm gear motor behind the block
    cyl("/cartpole/rail/motor_shaft", 0.002, 0.03, (-px, 0.035, PULLEY_Z), STEEL)
    cyl("/cartpole/rail/idler_axle", 0.002, 0.03, (px, 0.02, PULLEY_Z), STEEL)

    # --- pulleys (visual links, spun at belt speed) ---
    pulleys = {}
    for name, sx in (("motor_pulley", -1), ("idler_pulley", 1)):
        p = body(f"/cartpole/{name}", (sx * px, BELT_Y, PULLEY_Z), 0.005, (1e-7, 1e-7, 1e-7))
        cyl(f"/cartpole/{name}/hub", r + 0.0008, 0.008, (0, 0, 0), STEEL)
        cyl(f"/cartpole/{name}/flange", r + 0.003, 0.001, (0, -0.0045, 0), STEEL)
        box(f"/cartpole/{name}/mark", (0.002, 0.0012, r + 0.003), (0, -0.0055, (r + 0.003) / 2), RED)
        pulleys[name] = p

    # --- carriage: all of RIG.cart_mass lives here (block + plate + clamp + encoder) ---
    cart = body("/cartpole/cart", (0, 0, 0), RIG.cart_mass, (1e-3, 1e-3, 1e-3))
    box("/cartpole/cart/block", (CARRIAGE_LEN, 0.027, 0.010), (0, 0, -0.001), BLACK)
    box("/cartpole/cart/plate", (0.08, 0.075, 0.005), (0, 0.0025, 0.0065), BLUE)
    box("/cartpole/cart/belt_clamp", (0.02, 0.01, 0.005), (0, BELT_Y, PULLEY_Z + r), DARK)
    box("/cartpole/cart/bracket", (0.05, 0.004, 0.05), (0, -0.032, 0.034), BLUE)
    cyl("/cartpole/cart/encoder", 0.019, 0.03, (0, -0.049, SHAFT_Z), DARK)
    cyl("/cartpole/cart/encoder_shaft", 0.003, 0.012, (0, -0.068, SHAFT_Z), STEEL)

    # --- pendulum on the encoder shaft, rod along +Z ---
    pole = body(
        "/cartpole/pole",
        (0, SHAFT_Y, SHAFT_Z),
        RIG.pole_mass,
        (RIG.pole_inertia, RIG.pole_inertia, 1e-6),
        com=(0, 0, RIG.pole_com),
    )
    cyl("/cartpole/pole/hub", 0.008, 0.012, (0, 0, 0), STEEL)
    cyl("/cartpole/pole/rod", RIG.pole_radius, RIG.pole_length, (0, 0, RIG.pole_length / 2), STEEL, axis="Z")

    # --- joints ---
    UsdPhysics.FixedJoint.Define(stage, "/cartpole/joints/world_to_rail").CreateBody1Rel().SetTargets([rail.GetPath()])
    slider = joint(UsdPhysics.PrismaticJoint, "cart", rail, cart, (0, 0, 0), "X")
    slider.CreateLowerLimitAttr(-stop)
    slider.CreateUpperLimitAttr(stop)
    revolutes = [joint(UsdPhysics.RevoluteJoint, "pole", cart, pole, (0, SHAFT_Y, SHAFT_Z), "Y")]
    for name, sx in (("motor_pulley", -1), ("idler_pulley", 1)):
        revolutes.append(joint(UsdPhysics.RevoluteJoint, name, rail, pulleys[name], (sx * px, BELT_Y, PULLEY_Z), "Y"))
    # a (zero-gain) drive so the actuator damping / velocity targets have something to act on
    for j in revolutes:
        UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular").CreateTypeAttr("force")

    stage.GetRootLayer().Export(path)
    return path


def spawn_rig(prim_path, cfg, *args, **kwargs):
    """Write the USD from RIG, then spawn it like a normal UsdFileCfg."""
    write_usd(cfg.usd_path)
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
        # effort-controlled: DCMotorAction writes the belt force. Armature = motor + pulley inertia through the belt.
        "cart": ImplicitActuatorCfg(
            joint_names_expr=["cart"], effort_limit_sim=100.0, stiffness=0.0, damping=0.0, armature=RIG.reflected_mass
        ),
        # passive pivot: zero-velocity-target damping == viscous bearing friction
        "pole": ImplicitActuatorCfg(
            joint_names_expr=["pole"], effort_limit_sim=100.0, stiffness=0.0, damping=RIG.pivot_viscous
        ),
        # visual pulleys: velocity-tracking drive, target = belt speed / r (set by DCMotorAction)
        "pulleys": ImplicitActuatorCfg(
            joint_names_expr=[".*_pulley"], effort_limit_sim=1.0, stiffness=0.0, damping=1e-4
        ),
    },
)

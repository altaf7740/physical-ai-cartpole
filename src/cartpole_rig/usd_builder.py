"""Builds the cart-pole rig USD from a RigCfg. Uses only ``pxr`` (no Isaac Lab), so plain Isaac Sim can use it too.

Front view (-y toward the viewer, so +x is to the right; +z up):
  - rail (fixed to the world): 2020 extrusion + MGN12 rail, olive end blocks, gear motor on the left block
  - joint ``cart`` (prismatic X): carriage, plate, belt clamp, encoder bracket + encoder housing
  - joint ``pole`` (revolute Y, on the encoder shaft): steel rod, built pointing +Z so angle 0 is upright
  - joints ``motor_pulley`` / ``idler_pulley`` (revolute Y): GT2 pulleys; the belt loop runs between them
    behind the rail and its top strand is clamped to the carriage

The belt is modelled as rigid: the motor torque reaches the carriage as force = torque / r and the motor + pulley
inertia sits on the ``cart`` joint as armature J / r^2. The pulley links are visual (near-zero inertia) and are
spun at belt speed by the controller. The armature and the pole / pulley drive damping are written into the USD
(Isaac Lab sets the same values again from its actuator config). No colliders: nothing touches anything.

Import this only after Kit has started (importing ``pxr`` before Kit breaks Kit's own USD libraries).
"""

import math
import os

# visual geometry [m] (dynamics only use the masses / inertias / joint frames below)
CARRIAGE_LEN = 0.045  # MGN12H block length
BELT_Y = 0.025  # belt plane, behind the rail
PULLEY_Z = -0.004  # pulley centre height (top belt strand at PULLEY_Z + r, just under the cart plate)
SHAFT_Y, SHAFT_Z = -0.072, 0.04  # pendulum pivot on the cart: encoder shaft tip, in front of the carriage
PULLEY_DAMPING = 1e-4  # [N*m*s/rad] velocity drive that spins the visual pulleys

STEEL, ALU, DARK, BLACK = (0.75, 0.75, 0.78), (0.82, 0.83, 0.85), (0.25, 0.25, 0.27), (0.08, 0.08, 0.08)
OLIVE, BLUE, RED = (0.5, 0.5, 0.0), (0.2, 0.45, 0.8), (0.85, 0.1, 0.1)


def write_usd(rig, path: str) -> str:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

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

    def damping_drive(j, damping_per_rad):
        # zero-stiffness force drive; USD angular drive gains are per degree
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(0.0)
        d.CreateDampingAttr(damping_per_rad * math.pi / 180.0)

    r = rig.pulley_radius
    stop = rig.cart_limit + rig.end_stop_margin
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

    # --- carriage: all of rig.cart_mass lives here (block + plate + clamp + encoder) ---
    cart = body("/cartpole/cart", (0, 0, 0), rig.cart_mass, (1e-3, 1e-3, 1e-3))
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
        rig.pole_mass,
        (rig.pole_inertia, rig.pole_inertia, 1e-6),
        com=(0, 0, rig.pole_com),
    )
    cyl("/cartpole/pole/hub", 0.008, 0.012, (0, 0, 0), STEEL)
    cyl("/cartpole/pole/rod", rig.pole_radius, rig.pole_length, (0, 0, rig.pole_length / 2), STEEL, axis="Z")

    # --- joints ---
    UsdPhysics.FixedJoint.Define(stage, "/cartpole/joints/world_to_rail").CreateBody1Rel().SetTargets([rail.GetPath()])
    slider = joint(UsdPhysics.PrismaticJoint, "cart", rail, cart, (0, 0, 0), "X")
    slider.CreateLowerLimitAttr(-stop)
    slider.CreateUpperLimitAttr(stop)
    PhysxSchema.PhysxJointAPI.Apply(slider.GetPrim()).CreateArmatureAttr(rig.reflected_mass)  # motor + pulley
    damping_drive(joint(UsdPhysics.RevoluteJoint, "pole", cart, pole, (0, SHAFT_Y, SHAFT_Z), "Y"), rig.pivot_viscous)
    for name, sx in (("motor_pulley", -1), ("idler_pulley", 1)):
        j = joint(UsdPhysics.RevoluteJoint, name, rail, pulleys[name], (sx * px, BELT_Y, PULLEY_Z), "Y")
        damping_drive(j, PULLEY_DAMPING)

    stage.GetRootLayer().Export(path)
    return path

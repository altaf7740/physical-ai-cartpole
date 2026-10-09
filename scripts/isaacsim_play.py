"""Plain Isaac Sim player (no Isaac Lab): the rig balanced by the exported policy, with keyboard taps.

Start it through Isaac Sim's own launcher (``make run-sim`` does this), from the project folder:
    C:\\isaac\\IsaacLab\\_isaac_sim\\isaac-sim.bat --exec scripts\\isaacsim_play.py

It builds the rig USD from rig_cfg.py, adds a physics scene (120 Hz, CPU), a light and a floor, loads policy.pt
(newest logs/rsl_rl/cartpole_rig/*/exported/policy.pt, or the CARTPOLE_POLICY environment variable) and presses
Play. On every physics step it runs the same DC-motor + rail-friction model as the Isaac Lab env, and every
second step (60 Hz) the policy, with the nominal one-step action delay. No noise and no domain randomization:
this is the nominal rig.

Keys (click the viewport first): LEFT / RIGHT tap the pole, A / D bump the cart, UP / DOWN tap strength,
R reset. The rig also resets by itself when the pole passes the termination angle or the cart leaves the rail.
"""

import asyncio
import glob
import importlib.util
import math
import os
import random
import sys
import time

import numpy as np
import torch

import omni.kit.app
import omni.physics.tensors as tensors
import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

ROOT = os.environ.get("CARTPOLE_ROOT") or os.getcwd()  # project folder
ROBOT = "/World/Robot"
ROBOT_Z = 0.3  # [m] rail height, as in the Isaac Lab env


def _load(name: str, path: str):
    """Import one file as a module (the cartpole_rig package itself imports Isaac Lab)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RIG = _load("rig_cfg", os.path.join(ROOT, "src", "cartpole_rig", "rig_cfg.py")).RIG
usd_builder = _load("usd_builder", os.path.join(ROOT, "src", "cartpole_rig", "usd_builder.py"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from viewer_ui import KeyTaps, SignalPanel  # noqa: E402


def find_policy() -> str:
    path = os.environ.get("CARTPOLE_POLICY")
    if path:
        return path
    found = sorted(glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "cartpole_rig", "*", "exported", "policy.pt")))
    if not found:
        raise FileNotFoundError("no exported policy.pt under logs/: run `make play` once to export one")
    return found[-1]


def build_stage(stage, rig_usd: str):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateTimeStepsPerSecondAttr(round(1 / RIG.sim_dt))
    physx.CreateEnableGPUDynamicsAttr(False)
    physx.CreateBroadphaseTypeAttr("MBP")
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(500.0)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")  # visual only
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.005))
    floor.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.01))
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.18, 0.2)])
    robot = stage.DefinePrim(ROBOT, "Xform")
    robot.GetReferences().AddReference(rig_usd)
    UsdGeom.Xformable(robot).AddTranslateOp().Set(Gf.Vec3d(0, 0, ROBOT_Z))


def set_camera():
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    cam = ViewportCameraState("/OmniverseKit_Persp")
    cam.set_position_world(Gf.Vec3d(0.3, -1.25, 0.95 + ROBOT_Z), True)  # same view as the Isaac Lab env
    cam.set_target_world(Gf.Vec3d(0.0, 0.0, 0.55 + ROBOT_Z), True)


class Player:
    """Policy + DC motor on Isaac Sim's physics steps."""

    def __init__(self, policy_path: str):
        self.policy = torch.jit.load(policy_path, map_location="cpu").eval()
        self.scale = np.array(RIG.obs_scale, dtype=np.float32)
        self.keys = KeyTaps(RIG)
        self.panel = SignalPanel(RIG)
        self.view = None
        self.steps = 0
        self.command = 0.0  # voltage fraction applied this step (policy output delayed one control step)
        self.next_command = 0.0
        self.action = 0.0
        self.force = 0.0
        self.falls = 0
        self.speed, self.t_last = 1.0, time.time()
        self._sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_step)
        print(f"[isaacsim_play] policy: {policy_path}", flush=True)

    def _init_view(self):
        sim_view = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
        sim_view.set_subspace_roots("/")
        self.view = sim_view.create_articulation_view(ROBOT)
        names = list(self.view.shared_metatype.dof_names)
        self.cart, self.pole = names.index("cart"), names.index("pole")
        self.pulleys = [names.index("motor_pulley"), names.index("idler_pulley")]
        self.idx = np.array([0], dtype=np.uint32)
        self._sim_view = sim_view  # keep alive
        self.reset()

    def reset(self):
        q = np.zeros((1, self.view.max_dofs), dtype=np.float32)
        q[0, self.pole] = random.uniform(-RIG.reset_pole_angle, RIG.reset_pole_angle)
        self.view.set_dof_positions(q, self.idx)
        self.view.set_dof_velocities(np.zeros_like(q), self.idx)
        self.command = self.next_command = self.action = 0.0

    def _on_step(self, dt: float):
        if self.view is None:
            self._init_view()
        q = self.view.get_dof_positions()
        qd = self.view.get_dof_velocities()
        x, x_dot = float(q[0, self.cart]), float(qd[0, self.cart])
        theta, theta_dot = float(q[0, self.pole]), float(qd[0, self.pole])

        if self.keys.reset_requested or abs(theta) > RIG.max_pole_angle or abs(x) > RIG.cart_limit:
            if not self.keys.reset_requested:
                self.falls += 1
                print(f"[isaacsim_play] fell (x={x:+.3f} m, theta={theta:+.3f} rad), reset #{self.falls}", flush=True)
            self.keys.reset_requested = False
            self.reset()
            return
        taps = self.keys.pop()
        if taps:
            qd = qd.copy()
            for d_cart, d_pole in taps:
                qd[0, self.cart] += d_cart
                qd[0, self.pole] += d_pole
            self.view.set_dof_velocities(qd, self.idx)

        # policy at 60 Hz; its output reaches the motor one control step later (nominal action delay)
        if self.steps % RIG.decimation == 0:
            raw = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
            with torch.inference_mode():
                self.action = float(self.policy(torch.from_numpy(raw * self.scale)[None])[0, 0].clamp(-1.0, 1.0))
            self.command, self.next_command = self.next_command, self.action
            now = time.time()
            self.speed = 0.97 * self.speed + 0.03 * (RIG.sim_dt * RIG.decimation) / max(now - self.t_last, 1e-6)
            self.t_last = now
            self.panel.update(raw.tolist(), raw.tolist(), self.action, self.command * RIG.supply_voltage, self.force, self.speed)
        if self.steps % round(5.0 / RIG.sim_dt) == 0:  # status line every 5 s of sim time
            print(
                f"[isaacsim_play] t={self.steps * RIG.sim_dt:6.1f} s  x={x:+.3f} m  theta={theta:+.3f} rad  "
                f"speed={self.speed:.2f}x  falls={self.falls}",
                flush=True,
            )
        self.steps += 1

        # DC motor through the belt + rail friction (same equations as mdp.DCMotorAction, nominal values)
        r = RIG.pulley_radius
        current = (self.command * RIG.supply_voltage - RIG.motor_ke * x_dot / r) / RIG.motor_resistance
        belt = max(-RIG.max_force, min(RIG.max_force, RIG.motor_kt * current / r))
        self.force = belt - RIG.cart_viscous * x_dot - RIG.cart_coulomb * math.tanh(x_dot / 0.01)
        f = np.zeros((1, self.view.max_dofs), dtype=np.float32)
        f[0, self.cart] = self.force
        self.view.set_dof_actuation_forces(f, self.idx)
        w = np.zeros_like(f)
        w[0, self.pulleys] = x_dot / r  # visual pulleys at belt speed
        self.view.set_dof_velocity_targets(w, self.idx)


async def main():
    for _ in range(10):  # let the UI finish starting
        await omni.kit.app.get_app().next_update_async()
    rig_usd = usd_builder.write_usd(RIG, os.path.join(ROOT, "src", "cartpole_rig", "assets", "cartpole_rig.usda"))
    ctx = omni.usd.get_context()
    await ctx.new_stage_async()
    build_stage(ctx.get_stage(), rig_usd)
    await omni.kit.app.get_app().next_update_async()
    set_camera()
    global PLAYER
    PLAYER = Player(find_policy())
    omni.timeline.get_timeline_interface().play()
    print("[isaacsim_play] playing", flush=True)


PLAYER = None
asyncio.ensure_future(main())

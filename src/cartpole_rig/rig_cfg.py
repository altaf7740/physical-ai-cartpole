"""Every physical parameter of the real cart-pole rig, in one place.

Plain Python (no Isaac Lab imports) so the asset generator, the env and the export script all read the same
numbers. Edit RIG below after measuring the rig; the USD asset is regenerated from it on the next run.

Conventions: cart position x = 0 at rail centre, +x is the direction the cart moves for a positive action.
Pole angle theta = 0 upright, theta > 0 when the pole leans toward +x.

The cart is driven by a NEMA17 stepper (TB6600 at 1/8 microstepping) through a GT2 belt, and the Pi runs the
policy as an *acceleration* command (hardware/src/interface.py, action_type = "acceleration"):
    v_cmd += clip(action, -1, 1) * max_accel * dt,  v_cmd clipped to +-max_speed,  step rate = v_cmd * steps/m
and the Pico ramps the real step rate toward that at pico_accel (hardware/src/pico_main.py). A stepper that does
not lose steps moves exactly as commanded, so cart mass and rail friction do not change the cart motion; the sim
drives the cart joint with a stiff velocity drive.
"""

import math
from dataclasses import dataclass


@dataclass
class RigCfg:
    # --- rail / cart travel ---
    cart_limit: float = 0.35  # [m] |x| beyond this ends the episode (limit switches at +-0.3966 m)
    end_stop_margin: float = 0.03  # [m] hard joint limit = cart_limit + margin (~ the limit switches)

    # --- masses / geometry ---
    cart_mass: float = 0.8  # [kg] plate + MGN12 carriage + pendulum encoder (placeholder; no effect on a stepper)
    pole_mass: float = 0.197  # [kg] steel rod
    # Measured 2026-10-23: hanging swing period 1.00 s -> equivalent length J_pivot / (m * l) = 0.250 m, i.e. a
    # uniform rod of 0.375 m (the encoder hub / clamp shorten the effective length of the 0.5 m rod).
    pole_length: float = 0.375  # [m] uniform rod with the measured swing period
    pole_com: float = 0.1875  # [m] pivot to pole centre of mass (uniform rod: length / 2)
    pole_radius: float = 0.004  # [m] rod radius (visual only; inertia uses the slender-rod formula)

    # --- drive: GT2 belt, 20T pulley, stepper ---
    pulley_radius: float = 0.040 / (2 * math.pi)  # [m] 40 mm travel per rev -> ~6.37 mm
    steps_per_meter: float = 40000.0  # [steps/m] TB6600 at 1/8 = 1600 steps/rev / 0.04 m
    max_speed: float = 0.25  # [m/s] command clip; the stepper loses steps above ~0.3 m/s at 1.0 A (2026-10-23)
    max_accel: float = 5.0  # [m/s^2] action +1 == +max_accel (Pi integrates it into the speed command)
    pico_accel: float = 5.0  # [m/s^2] Pico ramp (ACCEL = 200000 steps/s^2 at 40000 steps/m)
    pico_start_speed: float = 0.03  # [m/s] Pico START_HZ 1200: speeds below this are applied without a ramp
    cart_drive_damping: float = 2000.0  # [N*s/m] stiff velocity drive that makes the cart follow the stepper

    # --- friction ---
    # Measured: the hanging swing loses a constant 2.35 deg per cycle (Coulomb, no visible viscous part), i.e. a
    # friction deceleration of ~0.40 rad/s^2; torque = that * J_pivot.
    pivot_coulomb_accel: float = 0.40  # [rad/s^2]
    pivot_viscous: float = 1.0e-5  # [N*m*s/rad] pole bearing viscous friction (small)

    # --- timing ---
    sim_dt: float = 1.0 / 240.0  # [s] physics step (the Pico ramp is applied per physics step)
    control_hz: float = 60.0  # [Hz] policy rate (decimation = 1 / (sim_dt * control_hz))
    episode_s: float = 15.0  # [s] time-out
    obs_delay_steps: tuple = (0, 0)  # [control steps] the Pi acts on the reply it just got
    action_delay_steps: tuple = (1, 1)  # [control steps] a command goes out with the next tick's request

    # --- sensors / Pi state estimator (hardware/src/state.py) ---
    encoder_cpr: int = 2400  # [counts/rev] 600 PPR quadrature
    velocity_alpha: float = 0.3  # low-pass weight on new finite-difference velocity samples
    # the pole angle zero is set with the pole hanging, where pivot friction leaves it up to ~0.8 deg off
    angle_offset: float = math.radians(1.0)  # [rad] per-episode constant bias, uniform in +-this

    # --- reset (the policy takes over when a hand-held pole is within engage_angle of upright) ---
    reset_pole_angle: float = math.radians(5.0)  # [rad] pole starts within +/- this of upright
    reset_pole_vel: float = 0.3  # [rad/s]
    reset_cart_pos: float = 0.1  # [m]
    reset_cart_vel: float = 0.0  # [m/s] the cart is at rest when the policy engages

    # --- disturbances: random "taps" on the pole during the episode (instant velocity kicks) ---
    # at 0.25 m/s the rig can absorb ~0.65 rad/s from upright (speed_budget.py), so keep the taps below that
    push_interval_s: tuple = (2.0, 4.0)  # [s] time between taps, per env
    push_pole_vel: float = 0.4  # [rad/s] pole angular velocity kick, uniform in +/- this

    # --- termination ---
    max_pole_angle: float = math.radians(17.0)  # [rad]

    # --- observations: [x, x_dot, theta, theta_dot] ---
    # extra Gaussian noise on top of the modelled encoder / step quantization and velocity filter
    obs_noise_std: tuple = (0.0, 0.005, 0.001, 0.02)  # [m, m/s, rad, rad/s]
    obs_scale: tuple = (1 / 0.35, 1 / 0.25, 1 / 0.1, 1 / 1.0)  # [1/unit] policy input = obs * scale

    # --- rewards (weights per step) ---
    w_alive: float = 1.0
    w_pole_angle: float = 20.0  # * theta^2 [rad^2]
    w_cart_pos: float = 1.0  # * x^2 [m^2]
    w_cart_vel: float = 0.5  # * x_dot^2 [m^2/s^2]
    w_action: float = 0.01  # * a^2
    w_action_rate: float = 0.02  # * (a_t - a_{t-1})^2

    # --- domain randomization (scale factors on the nominal values above, uniform) ---
    dr_pole_mass: tuple = (0.9, 1.1)
    dr_pole_com: tuple = (-0.015, 0.015)  # [m] added to pole_com (changes the swing period ~+-4%)
    dr_pivot_friction: tuple = (0.5, 2.0)  # scales pivot_coulomb_accel and pivot_viscous
    dr_max_accel: tuple = (0.9, 1.1)  # scales the speed-command integration (dt jitter, ramp)
    dr_obs_noise: tuple = (0.5, 1.5)  # scales obs_noise_std

    @property
    def decimation(self) -> int:
        return round(1.0 / (self.sim_dt * self.control_hz))

    @property
    def pole_inertia(self) -> float:
        """[kg*m^2] uniform slender rod about its centre of mass."""
        return self.pole_mass * self.pole_length**2 / 12.0

    @property
    def pole_pivot_inertia(self) -> float:
        """[kg*m^2] about the pivot."""
        return self.pole_inertia + self.pole_mass * self.pole_com**2


RIG = RigCfg()

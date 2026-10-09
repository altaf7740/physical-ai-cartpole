"""Every physical parameter of the real cart-pole rig, in one place.

Plain Python (no Isaac Lab imports) so the asset generator, the env and the export script all read the same
numbers. Edit RIG below after measuring the rig; the USD asset is regenerated from it on the next run.

Conventions: cart position x = 0 at rail centre, +x is the direction the cart moves for a positive action.
Pole angle theta = 0 upright, theta > 0 when the pole leans toward +x.
"""

import math
from dataclasses import dataclass


@dataclass
class RigCfg:
    # --- rail / cart travel ---
    cart_limit: float = 0.35  # [m] |x| beyond this ends the episode (usable travel ~0.7 m)
    end_stop_margin: float = 0.01  # [m] hard joint limit = cart_limit + margin (end stops)

    # --- masses / geometry ---
    cart_mass: float = 0.8  # [kg] plate + MGN12 carriage + pendulum encoder (placeholder, measure)
    pole_mass: float = 0.197  # [kg] steel rod
    pole_length: float = 0.5  # [m] pivot to rod tip
    pole_com: float = 0.25  # [m] pivot to pole centre of mass (uniform rod: length / 2)
    pole_radius: float = 0.004  # [m] rod radius (visual only; inertia uses the slender-rod formula)

    # --- drive: GT2 belt, 20T pulley ---
    pulley_radius: float = 0.040 / (2 * math.pi)  # [m] 40 mm travel per rev -> ~6.37 mm

    # --- DC gear motor, constants referred to the gearbox output shaft ---
    supply_voltage: float = 12.0  # [V] action +1 == +supply_voltage
    motor_ke: float = 12.0 / (1000 * 2 * math.pi / 60)  # [V*s/rad] back-EMF, from 1000 RPM no-load at 12 V
    motor_kt: float = 12.0 / (1000 * 2 * math.pi / 60)  # [N*m/A] torque constant (= Ke in SI)
    motor_resistance: float = 20.0  # [ohm] chosen so stall force at 12 V ~= 11 N (0.7 kg*cm treated as stall)
    max_force: float = 11.0  # [N] cart force clip
    deadband: float = 0.0  # [-] |action| below this gives 0 V (driver deadband / belt slack), 0..1
    # rotor (x gear ratio^2) + gearbox + pulley inertia at the output shaft. PLACEHOLDER (rough estimate for a
    # ~1:10 geared 370-size motor): measure it, it matters. Seen by the cart as an extra mass J / r^2 (~0.37 kg).
    motor_inertia: float = 1.5e-5  # [kg*m^2]

    # --- friction ---
    cart_viscous: float = 0.5  # [N*s/m] rail + belt viscous friction
    cart_coulomb: float = 0.3  # [N] rail + belt Coulomb friction
    pivot_viscous: float = 1.0e-4  # [N*m*s/rad] pole bearing viscous friction

    # --- timing ---
    sim_dt: float = 1.0 / 120.0  # [s] physics step
    control_hz: float = 60.0  # [Hz] policy rate (decimation = 1 / (sim_dt * control_hz))
    episode_s: float = 15.0  # [s] time-out
    obs_delay_steps: tuple = (0, 1)  # [control steps] sensor -> policy latency, sampled per episode
    action_delay_steps: tuple = (1, 1)  # [control steps] policy -> PWM latency, sampled per episode

    # --- reset ---
    reset_pole_angle: float = math.radians(8.0)  # [rad] pole starts within +/- this of upright
    reset_pole_vel: float = 0.1  # [rad/s]
    reset_cart_pos: float = 0.05  # [m]
    reset_cart_vel: float = 0.05  # [m/s]

    # --- disturbances: random "taps" during the episode (instant velocity kicks) ---
    # a 0.01 N*s tap at the rod tip gives ~0.3 rad/s; 1 rad/s is a firm flick
    push_interval_s: tuple = (2.0, 4.0)  # [s] time between taps, per env
    push_pole_vel: float = 1.0  # [rad/s] pole angular velocity kick, uniform in +/- this
    push_cart_vel: float = 0.1  # [m/s] cart velocity kick (bump to the carriage), uniform in +/- this

    # --- termination ---
    max_pole_angle: float = math.radians(17.0)  # [rad]

    # --- observations: [x, x_dot, theta, theta_dot] ---
    obs_noise_std: tuple = (0.0005, 0.01, 0.002, 0.03)  # [m, m/s, rad, rad/s] Gaussian sensor noise
    obs_scale: tuple = (1 / 0.35, 1 / 0.7, 1 / 0.3, 1 / 3.0)  # [1/unit] policy input = obs * scale

    # --- rewards (weights per step) ---
    w_alive: float = 1.0
    w_pole_angle: float = 5.0  # * theta^2 [rad^2]
    w_cart_pos: float = 1.0  # * x^2 [m^2]
    w_cart_vel: float = 0.1  # * x_dot^2 [m^2/s^2]
    w_action: float = 0.01  # * a^2
    w_action_rate: float = 0.05  # * (a_t - a_{t-1})^2

    # --- domain randomization (scale factors on the nominal values above, uniform) ---
    dr_cart_mass: tuple = (0.8, 1.25)
    dr_pole_mass: tuple = (0.9, 1.1)
    dr_pole_com: tuple = (-0.025, 0.025)  # [m] added to pole_com (stands in for length error)
    dr_cart_friction: tuple = (0.5, 2.0)  # scales cart_viscous and cart_coulomb
    dr_pivot_friction: tuple = (0.5, 2.0)
    dr_motor_kt: tuple = (0.8, 1.2)
    dr_motor_ke: tuple = (0.8, 1.2)
    dr_motor_resistance: tuple = (0.8, 1.3)
    dr_motor_inertia: tuple = (0.5, 2.0)
    dr_obs_noise: tuple = (0.5, 1.5)  # scales obs_noise_std

    @property
    def decimation(self) -> int:
        return round(1.0 / (self.sim_dt * self.control_hz))

    @property
    def reflected_mass(self) -> float:
        """[kg] motor + pulley inertia as seen by the cart through a rigid belt."""
        return self.motor_inertia / self.pulley_radius**2

    @property
    def pole_inertia(self) -> float:
        """[kg*m^2] uniform slender rod about its centre of mass."""
        return self.pole_mass * self.pole_length**2 / 12.0


RIG = RigCfg()

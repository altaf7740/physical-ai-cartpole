"""Simulated rig for testing run_policy.py without hardware (pure Python, no Isaac).

It produces what the Pico would send: motor encoder counts and pendulum encoder counts (pendulum 0 = hanging
down), and takes the PWM duty. Physics: rigid cart-pole (pole = uniform rod on a free pivot) driven by the same
DC-motor + belt + rail-friction model as the Isaac Lab env, with end stops at the rail limits, integrated at
120 Hz. After reset() the pole is held upright, as your hand would, until release(); while running it gets a
random tap every few seconds. Parameters come from the "rig" section of policy_io.json (written by
scripts/play.py).

This is a test stand-in. It is not PhysX and not the real rig.
"""

import math
import random

G = 9.81


class FakeRig:
    def __init__(self, rig: dict, motor_cpr: int, travel_per_rev: float, pole_cpr: int, dt: float, taps: bool = True):
        self.p = rig
        self.motor_cpr, self.travel_per_rev, self.pole_cpr = motor_cpr, travel_per_rev, pole_cpr
        self.dt, self.substeps = dt, 2
        self.taps = taps
        self.duty = 0.0
        self.time_to_tap = random.uniform(3.0, 5.0)
        self.reset()

    def reset(self):
        """Stand the pole back up near the rail centre and hold it there (what you would do by hand)."""
        self.x, self.x_dot = 0.0, 0.0
        self.theta, self.theta_dot = random.uniform(-0.05, 0.05), 0.0  # from upright
        self.held = True

    def release(self):
        """Let go of the pole."""
        self.held = False

    def read_counts(self) -> tuple[int, int]:
        motor = round(self.x / self.travel_per_rev * self.motor_cpr)
        pole = round((self.theta + math.pi) / (2 * math.pi) * self.pole_cpr)  # encoder 0 = hanging
        return motor, pole

    def write_duty(self, duty: float):
        self.duty = max(-1.0, min(1.0, duty))

    def advance(self):
        """Simulate one control period."""
        p = self.p
        r = p["pulley_radius"]
        m_cart = p["cart_mass"] + p["motor_inertia"] / r**2  # reflected motor + pulley inertia
        m, l, i_com = p["pole_mass"], p["pole_com"], p["pole_mass"] * p["pole_length"] ** 2 / 12.0
        stop = p["cart_limit"] + p["end_stop_margin"]
        if self.held:
            self.theta_dot = 0.0  # hand on the pole
            return
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            v = self.x_dot
            current = (self.duty * p["supply_voltage"] - p["motor_ke"] * v / r) / p["motor_resistance"]
            belt = max(-p["max_force"], min(p["max_force"], p["motor_kt"] * current / r))
            force = belt - p["cart_viscous"] * v - p["cart_coulomb"] * math.tanh(v / 0.01)
            s, c = math.sin(self.theta), math.cos(self.theta)
            # [m_cart + m, m l c; m l c, I + m l^2] [x_dd; th_dd] = [F + m l s th_d^2; m g l s - b th_d]
            a11, a12, a22 = m_cart + m, m * l * c, i_com + m * l * l
            b1 = force + m * l * s * self.theta_dot**2
            b2 = m * G * l * s - p["pivot_viscous"] * self.theta_dot
            det = a11 * a22 - a12 * a12
            x_dd = (b1 * a22 - a12 * b2) / det
            th_dd = (a11 * b2 - a12 * b1) / det
            self.x_dot += x_dd * h
            self.x += self.x_dot * h
            self.theta_dot += th_dd * h
            self.theta += self.theta_dot * h
            if abs(self.x) > stop:  # end stop
                self.x, self.x_dot = math.copysign(stop, self.x), 0.0
        if self.taps:
            self.time_to_tap -= self.dt
            if self.time_to_tap <= 0:
                self.theta_dot += random.uniform(-0.5, 0.5) * p["push_pole_vel"]
                self.time_to_tap = random.uniform(3.0, 5.0)

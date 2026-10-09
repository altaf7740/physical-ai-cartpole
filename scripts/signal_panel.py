"""Live signal panel for the Kit viewer: env 0's policy inputs and outputs in physical units, with 5 s plots.

Inputs are shown as the policy sees them (noisy and delayed, divided back out of obs_scale), with the true
simulator value next to them. Plot ranges are the termination limits / physical limits from RIG.
"""

import math

import omni.ui as ui

HISTORY = 300  # samples per plot (5 s at 60 Hz)
BLUE, ORANGE, GREY = 0xFFFFCC66, 0xFF3399FF, 0xFFBBBBBB  # omni.ui colours are 0xAABBGGRR


class SignalPanel:
    def __init__(self, rig):
        v_max = rig.supply_voltage / rig.motor_ke * rig.pulley_radius  # [m/s] no-load cart speed
        # (key, section, title, unit, plot range +-, colour)
        self._rows = [
            ("x", "Policy inputs (measured: noisy + delayed)", "cart position", "m", rig.cart_limit, BLUE),
            ("x_dot", None, "cart velocity", "m/s", 1.2 * v_max, BLUE),
            ("theta", None, "pole angle", "rad", rig.max_pole_angle, BLUE),
            ("theta_dot", None, "pole angular velocity", "rad/s", 3.0, BLUE),
            ("action", "Policy output", "action (PWM duty)", "-1..1", 1.0, ORANGE),
            ("voltage", "Applied to the motor (after action delay)", "motor voltage", "V", rig.supply_voltage, ORANGE),
            ("force", None, "cart force (belt - friction)", "N", rig.max_force, ORANGE),
        ]
        self._data = {key: [0.0] * HISTORY for key, *_ in self._rows}
        self._labels, self._plots = {}, {}
        self._window = ui.Window("Rig signals (env 0)", width=460, height=720, position_x=40, position_y=60)
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=3):
                    self._speed = ui.Label("", height=18, style={"color": GREY})
                    for key, section, title, unit, lim, colour in self._rows:
                        if section:
                            ui.Spacer(height=6)
                            ui.Label(section, height=20, style={"font_size": 16, "color": GREY})
                        with ui.HStack(height=18):
                            ui.Label(f"{title} [{unit}]", width=200)
                            self._labels[key] = ui.Label("", style={"color": colour})
                        with ui.HStack(height=44):
                            ui.Label(f"±{lim:.2f}", width=40, style={"color": GREY, "font_size": 12})
                            self._plots[key] = ui.Plot(
                                ui.Type.LINE,
                                -lim,
                                lim,
                                *self._data[key],
                                height=44,
                                style={"color": colour, "background_color": 0xFF202020},
                            )

    def update(self, meas, true, action: float, voltage: float, force: float, speed: float):
        """meas / true: [x, x_dot, theta, theta_dot] in m, m/s, rad, rad/s; speed: sim time / wall time."""
        self._speed.text = f"sim speed: {speed:.2f}x real time"
        texts = {
            "x": f"{meas[0]:+.4f}   (true {true[0]:+.4f})",
            "x_dot": f"{meas[1]:+.3f}   (true {true[1]:+.3f})",
            "theta": f"{meas[2]:+.4f} = {math.degrees(meas[2]):+.1f}°   (true {true[2]:+.4f})",
            "theta_dot": f"{meas[3]:+.3f}   (true {true[3]:+.3f})",
            "action": f"{action:+.3f}",
            "voltage": f"{voltage:+.2f}",
            "force": f"{force:+.2f}",
        }
        values = dict(zip(("x", "x_dot", "theta", "theta_dot"), meas))
        values.update(action=action, voltage=voltage, force=force)
        for key, text in texts.items():
            d = self._data[key]
            d.pop(0)
            d.append(values[key])
            self._plots[key].set_data(*d)
            self._labels[key].text = text

    def destroy(self):
        self._window.destroy()

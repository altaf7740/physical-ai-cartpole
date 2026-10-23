"""Kit viewer helpers (omni.ui / carb input only, no Isaac Lab): a live signal panel and keyboard taps.

SignalPanel: env 0's policy inputs and outputs in physical units, with 5 s plots.

Inputs are shown as the policy sees them (noisy and delayed, divided back out of obs_scale), with the true
simulator value next to them. Plot ranges are the termination limits / physical limits from RIG.
"""

import math

import omni.ui as ui

HISTORY = 300  # samples per plot (5 s at 60 Hz)
BLUE, ORANGE, GREY = 0xFFFFCC66, 0xFF3399FF, 0xFFBBBBBB  # omni.ui colours are 0xAABBGGRR


class SignalPanel:
    def __init__(self, rig):
        v_max = rig.max_speed
        # (key, section, title, unit, plot range +-, colour)
        self._rows = [
            ("x", "Policy inputs (as the Pi estimates them)", "cart position", "m", rig.cart_limit, BLUE),
            ("x_dot", None, "cart velocity", "m/s", 1.2 * v_max, BLUE),
            ("theta", None, "pole angle", "rad", rig.max_pole_angle, BLUE),
            ("theta_dot", None, "pole angular velocity", "rad/s", 3.0, BLUE),
            ("action", "Policy output", "action (cart accel / max_accel)", "-1..1", 1.0, ORANGE),
            ("v_cmd", "Stepper", "speed command at the Pico (after delay)", "m/s", v_max, ORANGE),
            ("v_step", None, "stepper speed (Pico ramp)", "m/s", v_max, ORANGE),
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

    def update(self, meas, true, action: float, v_cmd: float, v_step: float, speed: float):
        """meas / true: [x, x_dot, theta, theta_dot] in m, m/s, rad, rad/s; speed: sim time / wall time."""
        self._speed.text = f"sim speed: {speed:.2f}x real time"
        texts = {
            "x": f"{meas[0]:+.4f}   (true {true[0]:+.4f})",
            "x_dot": f"{meas[1]:+.3f}   (true {true[1]:+.3f})",
            "theta": f"{meas[2]:+.4f} = {math.degrees(meas[2]):+.1f}°   (true {true[2]:+.4f})",
            "theta_dot": f"{meas[3]:+.3f}   (true {true[3]:+.3f})",
            "action": f"{action:+.3f}",
            "v_cmd": f"{v_cmd:+.3f}",
            "v_step": f"{v_step:+.3f}",
        }
        values = dict(zip(("x", "x_dot", "theta", "theta_dot"), meas))
        values.update(action=action, v_cmd=v_cmd, v_step=v_step)
        for key, text in texts.items():
            d = self._data[key]
            d.pop(0)
            d.append(values[key])
            self._plots[key].set_data(*d)
            self._labels[key].text = text

    def destroy(self):
        self._window.destroy()


class KeyTaps:
    """Viewer keyboard disturbances, applied to every env (click the viewer window first so it has focus).

    LEFT / RIGHT  tap the pole toward -x / +x     (pole angular velocity -/+ rig.push_pole_vel * strength)
    UP / DOWN     stronger / weaker (x1.25 per press)
    R             reset request (used by the plain Isaac Sim player)
    (numpad 4 / 6 / 8 / 2 work as the arrows)
    """

    # Kit reports arrow keys as LEFT/RIGHT/UP/DOWN or as NUMPAD_4/6/8/2 depending on how they were sent
    _ALIASES = {"NUMPAD_4": "LEFT", "NUMPAD_6": "RIGHT", "NUMPAD_8": "UP", "NUMPAD_2": "DOWN"}

    def __init__(self, rig):
        import carb
        import omni.appwindow

        self.rig = rig
        self.strength = 1.0
        self.reset_requested = False
        self._pending = []  # (d_cart_vel, d_pole_vel) to apply before the next step
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._press = carb.input.KeyboardEventType.KEY_PRESS
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        print("[INFO] keyboard taps:" + self.__doc__.split("\n", 1)[1].rstrip())

    def _on_key(self, event, *args) -> bool:
        if event.type != self._press:
            return True
        name = self._ALIASES.get(event.input.name, event.input.name)
        if name == "R":
            self.reset_requested = True
            return True
        if name in ("UP", "DOWN"):
            self.strength *= 1.25 if name == "UP" else 0.8
            print(f"[TAP] strength x{self.strength:.2f} (pole tap {self.rig.push_pole_vel * self.strength:.2f} rad/s)")
            return True
        pole = self.rig.push_pole_vel * self.strength
        tap = {"LEFT": (0.0, -pole), "RIGHT": (0.0, pole)}.get(name)
        if tap is not None:
            self._pending.append(tap)
            print(f"[TAP] {name}: cart {tap[0]:+.2f} m/s, pole {tap[1]:+.2f} rad/s")
        return True

    def pop(self) -> list:
        pending, self._pending = self._pending, []
        return pending

    def close(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)

"""Hardware-free tests: python -m unittest discover -s src/tests -t src"""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import time

from calibration import TrackCalibrator, load_calibration, save_calibration
from config import Config, ConfigError, _validate, effective_max_velocity, load_config
from fake_pico import FakePico
from hardware_service import CommandError, HardwareService
from interface import ActionConverter, build_observation
from pico_link import Telemetry, format_step_rate, parse_telemetry
from policy import NumpyMlpPolicy
from safety import Mode, SafetyMonitor
from state import CartPoleState, StateEstimator, ticks_to_angle, wrap_angle

TEMPLATE = Path(__file__).resolve().parents[2] / "config.toml.template"


def template_config() -> Config:
    return load_config(TEMPLATE)


class TestConfig(unittest.TestCase):
    def test_template_loads(self):
        cfg = template_config()
        self.assertTrue(Path(cfg.policy.model_path).is_absolute())
        self.assertAlmostEqual(effective_max_velocity(cfg), 0.4)  # max_cart_vel_mps < max_step_rate / steps_per_meter

    def test_unknown_key_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('[policy]\nrate_hz = 60\nobs_ordr = ["cart_pos"]\n')
        with self.assertRaises(ConfigError):
            load_config(f.name)


class TestPicoProtocol(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_telemetry("120,-35,0\r\n"), Telemetry(120, -35, False))
        self.assertEqual(parse_telemetry("1,2,1"), Telemetry(1, 2, True))
        right = parse_telemetry("1,2,2")
        self.assertEqual(right.limits, 2)
        self.assertTrue(right.limit_hit)
        for bad in ("", "1,2", "a,b,c", "1,2,3,4"):
            self.assertIsNone(parse_telemetry(bad))

    def test_format(self):
        self.assertEqual(format_step_rate(-1250), b"-1250\n")


class TestState(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(wrap_angle(3 * math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(wrap_angle(-3 * math.pi / 2), math.pi / 2)

    def test_angle_zero_down(self):
        # Reset while hanging down: half a turn later is upright (0 rad).
        self.assertAlmostEqual(ticks_to_angle(1200, 2400, angle_zero="down"), 0.0)
        self.assertAlmostEqual(abs(ticks_to_angle(0, 2400, angle_zero="down")), math.pi)

    def test_angle_zero_upright(self):
        self.assertAlmostEqual(ticks_to_angle(0, 2400, angle_zero="upright"), 0.0)
        self.assertAlmostEqual(ticks_to_angle(-600, 2400, sign=-1, angle_zero="upright"), math.pi / 2)

    def test_velocity_across_wrap(self):
        cfg = template_config()
        est = StateEstimator(cfg.hardware, replace(cfg.estimator, velocity_alpha=1.0))
        est.update(Telemetry(1199 + 1200, 0, False), 0.01)  # just below +pi
        s = est.update(Telemetry(1201 + 1200, 100, False), 0.01)  # just past it
        self.assertAlmostEqual(s.pole_vel, 2 * (2 * math.pi / 2400) / 0.01, places=6)
        self.assertAlmostEqual(s.cart_vel, (100 / cfg.hardware.steps_per_meter) / 0.01)


class TestInterface(unittest.TestCase):
    def test_observation_order(self):
        s = CartPoleState(cart_pos=1, pole_pos=2, cart_vel=3, pole_vel=4)
        np.testing.assert_array_equal(
            build_observation(s, ("pole_pos", "pole_vel", "cart_pos", "cart_vel")), [2, 4, 1, 3]
        )

    def test_observation_scale(self):
        s = CartPoleState(cart_pos=1, pole_pos=2, cart_vel=3, pole_vel=4)
        np.testing.assert_allclose(
            build_observation(s, ("cart_pos", "cart_vel", "pole_pos", "pole_vel"), (2.0, 0.5, 10.0, 1.0)), [2, 1.5, 20, 4]
        )

    def test_obs_scale_length_checked(self):
        cfg = template_config()
        with self.assertRaises(ConfigError):
            _validate(replace(cfg, policy=replace(cfg.policy, obs_scale=(1.0, 2.0))))

    def test_velocity_action_clamped(self):
        cfg = template_config()
        conv = ActionConverter(cfg, effective_max_velocity(cfg))
        spm = cfg.hardware.steps_per_meter
        cap = min(cfg.hardware.max_step_rate, round(effective_max_velocity(cfg) * spm))
        self.assertEqual(conv.command(0.1, 0.01), round(0.1 * spm))
        self.assertEqual(conv.command(5.0, 0.01), cap)  # clipped to 1.0 m/s, then capped by the speed limit
        self.assertEqual(conv.command(-5.0, 0.01), -cap)

    def test_acceleration_integrates(self):
        cfg = template_config()
        cfg = replace(cfg, policy=replace(cfg.policy, action_type="acceleration", action_scale=10.0))
        conv = ActionConverter(cfg, effective_max_velocity(cfg))
        conv.command(1.0, 0.01)
        self.assertAlmostEqual(conv.velocity, 0.1)
        conv.command(1.0, 0.01)
        self.assertAlmostEqual(conv.velocity, 0.2)

    def test_cart_sign_flips_steps(self):
        cfg = template_config()
        cfg = replace(cfg, hardware=replace(cfg.hardware, cart_sign=-1))
        self.assertEqual(ActionConverter(cfg, 0.4).command(0.1, 0.01), -round(0.1 * cfg.hardware.steps_per_meter))


class TestSafety(unittest.TestCase):
    def state(self, deg=0.0, x=0.0):
        return CartPoleState(x, math.radians(deg), 0.0, 0.0)

    def test_engage_hysteresis(self):
        mon = SafetyMonitor(template_config().safety)
        self.assertIs(mon.update(self.state(30), False), Mode.IDLE)
        self.assertIs(mon.update(self.state(10), False), Mode.BALANCING)
        self.assertIs(mon.update(self.state(20), False), Mode.BALANCING)
        self.assertIs(mon.update(self.state(30), False), Mode.IDLE)

    def test_faults_latch(self):
        mon = SafetyMonitor(template_config().safety)
        self.assertIsNot(mon.update(self.state(0, x=0.3), False), Mode.FAULT)  # no soft track limit
        self.assertIs(mon.update(self.state(0), True), Mode.FAULT)
        self.assertIs(mon.update(self.state(0), False), Mode.FAULT)  # stays latched after the switch opens

    def test_limit_switch_and_lost_link(self):
        mon = SafetyMonitor(template_config().safety)
        self.assertIs(mon.update(self.state(0), True), Mode.FAULT)
        mon.reset()
        for _ in range(5):
            self.assertIsNot(mon.missed_read(), Mode.FAULT)
        self.assertIs(mon.missed_read(), Mode.FAULT)


class TestNumpyPolicy(unittest.TestCase):
    def test_mlp_forward(self):
        rng = np.random.default_rng(0)
        w0, b0 = rng.normal(size=(8, 4)), rng.normal(size=8)
        w1, b1 = rng.normal(size=(1, 8)), rng.normal(size=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.npz"
            np.savez(path, W0=w0, b0=b0, W1=w1, b1=b1, activation="relu")
            policy = NumpyMlpPolicy(path)
            obs = np.array([0.1, -0.2, 0.3, 0.0])
            expected = w1 @ np.maximum(w0 @ obs + b0, 0) + b1
            np.testing.assert_allclose(policy.act(obs), expected, rtol=1e-5)
            self.assertEqual((policy.obs_dim, policy.action_dim), (4, 1))


class TestFakePico(unittest.TestCase):
    def test_reset_reads_pole_down(self):
        cfg = template_config()
        pico = FakePico(cfg)
        pico.open()
        pico.reset_counters()
        tel = pico.exchange(0)
        angle = ticks_to_angle(tel.encoder_ticks, cfg.hardware.encoder_cpr, angle_zero=cfg.hardware.angle_zero)
        self.assertAlmostEqual(abs(angle), math.pi, places=2)

    def test_step_rate_moves_cart(self):
        pico = FakePico(template_config())
        pico.open()
        pico.exchange(2000)
        time.sleep(0.05)
        self.assertGreater(pico.exchange(0).cart_steps, 50)


def wait_for(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestHardwareService(unittest.TestCase):
    def setUp(self):
        cfg = template_config()
        self.cfg = replace(cfg, ui=replace(cfg.ui, home_on_connect=False))  # no automatic homing run
        self.svc = HardwareService(self.cfg, lambda: FakePico(self.cfg), fake=True)

    def tearDown(self):
        self.svc.close()

    def test_jog_streams_samples_and_moves(self):
        self.svc.connect()
        self.assertTrue(wait_for(lambda: self.svc.status()["link"] == "connected"))
        self.svc.jog(0.1)
        self.assertTrue(wait_for(lambda: any(s["cmd"] == round(0.1 * self.cfg.hardware.steps_per_meter) for s in self.svc.samples_since(0, 1000)[0])))
        self.assertTrue(wait_for(lambda: self.svc.samples_since(0, 1000)[0][-1]["cmd"] == 0, timeout=1.0),
                        "jog must expire without repeats (dead-man)")
        self.assertGreater(self.svc.samples_since(0, 1000)[0][-1]["x"], 0.0)

    def test_commands_need_connection(self):
        with self.assertRaises(CommandError):
            self.svc.jog(0.1)

    def test_manual_jog_ignores_soft_limit(self):
        # The physical switches stop the cart (in the Pico); manual jog may use the whole track.
        self.svc._jog_velocity, self.svc._jog_deadline = 0.1, time.monotonic() + 1
        beyond = CartPoleState(0.3, 0.0, 0.0, 0.0)
        self.assertEqual(self.svc._manual_rate(beyond), round(0.1 * self.cfg.hardware.steps_per_meter))



def run_calibration(cal: TrackCalibrator, left: int, right: int, start: int = 0, dt: float = 1 / 60,
                    max_ticks: int = 100_000) -> int:
    """Simulated cart between two switches at raw steps `left` < `right`, moving at the commanded rate."""
    steps, rate = float(start), 0
    for _ in range(max_ticks):
        steps = min(max(steps + rate * dt, left), right)  # the switches stop it, like the Pico
        limits = (1 if steps <= left else 0) | (2 if steps >= right else 0)
        rate = cal.step(round(steps), limits, dt)
        if cal.finished:
            return round(steps)
    raise AssertionError("calibration did not finish")


class TestCalibration(unittest.TestCase):
    def test_measures_travel_and_parks_in_the_middle(self):
        cal = TrackCalibrator(rate=2000)
        end = run_calibration(cal, left=-3000, right=5000)
        self.assertEqual(cal.phase, "done")
        self.assertEqual(cal.result.travel_steps, 8000)
        self.assertEqual(cal.result.passes, (8000, 8000))
        self.assertEqual(cal.result.centre_steps, 1000)
        self.assertLessEqual(abs(end - 1000), 40)

    def test_starting_on_the_right_switch_is_not_a_left_hit(self):
        cal = TrackCalibrator(rate=2000)
        run_calibration(cal, left=-3000, right=5000, start=5000)
        self.assertEqual(cal.result.travel_steps, 8000)

    def test_home_uses_saved_travel(self):
        cal = TrackCalibrator(rate=2000, travel_steps=8000)
        end = run_calibration(cal, left=-3000, right=5000, start=4000)
        self.assertEqual(cal.phase, "done")
        self.assertEqual(cal.legs, [-1])  # left switch only
        self.assertEqual(cal.target, 1000)
        self.assertIsNone(cal.result)  # homing does not re-measure
        self.assertLessEqual(abs(end - 1000), 40)

    def test_missing_switch_fails(self):
        cal = TrackCalibrator(rate=2000, leg_timeout_s=1.0)
        run_calibration(cal, left=-10**9, right=5000)
        self.assertEqual(cal.phase, "failed")
        self.assertIn("left", cal.error)

    def test_save_and_load(self):
        cal = TrackCalibrator(rate=2000)
        run_calibration(cal, left=-3000, right=5000)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "track.json"
            save_calibration(path, cal.result)
            self.assertEqual(load_calibration(path), cal.result)
            self.assertIsNone(load_calibration(Path(d) / "missing.json"))

    def test_homing_keeps_x_when_zeroing_the_pole(self):
        cfg = template_config()
        est = StateEstimator(cfg.hardware, cfg.estimator)
        est.set_origins(cart_steps=1000)
        est.set_origins(encoder_ticks=50)
        s = est.update(Telemetry(50, 1000, 0), 0.01)
        self.assertAlmostEqual(s.cart_pos, 0.0)
        self.assertAlmostEqual(s.pole_pos, ticks_to_angle(0, cfg.hardware.encoder_cpr, angle_zero=cfg.hardware.angle_zero))


if __name__ == "__main__":
    unittest.main()

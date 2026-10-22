# Pico-side HAT mapping check (motor disconnected): lights M1A, M1B, M2A, M2B in turn, twice.
from machine import Pin
import utime
PWM1, DIR1 = Pin(14, Pin.OUT, value=0), Pin(15, Pin.OUT, value=0)
PWM2, DIR2 = Pin(18, Pin.OUT, value=0), Pin(19, Pin.OUT, value=0)
STEPS = (("M1A", PWM1, DIR1, 0), ("M1B", PWM1, DIR1, 1), ("M2A", PWM2, DIR2, 0), ("M2B", PWM2, DIR2, 1))
try:
    t0 = utime.ticks_ms()
    for rnd in range(2):
        for name, pwm, d, level in STEPS:
            d.value(level)
            pwm.value(1)
            print("t=%4.1fs  %s should be lit" % (utime.ticks_diff(utime.ticks_ms(), t0) / 1000, name))
            utime.sleep(4)
            pwm.value(0)
finally:
    for p in (PWM1, DIR1, PWM2, DIR2):
        p.value(0)
    print("all off")

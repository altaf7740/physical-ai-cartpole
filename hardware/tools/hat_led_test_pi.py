# Pi-side HAT mapping check (motor disconnected), Cytron HAT-MDD10 pinout:
# PWM1=GPIO12, DIR1=GPIO26, PWM2=GPIO13, DIR2=GPIO24. Lights M1A, M1B, M2A, M2B in turn, twice.
import time
from gpiozero import OutputDevice

pins = {n: OutputDevice(n) for n in (12, 26, 13, 24)}
SLOTS = (("M1A", 12, 26, 0), ("M1B", 12, 26, 1), ("M2A", 13, 24, 0), ("M2B", 13, 24, 1))
try:
    t0 = time.monotonic()
    for _ in range(2):
        for name, pwm, d, level in SLOTS:
            pins[d].value = level
            pins[pwm].on()
            print(f"t={time.monotonic() - t0:4.1f}s  {name} should be lit", flush=True)
            time.sleep(4)
            pins[pwm].off()
finally:
    for p in pins.values():
        p.off()
        p.close()
    print("all off")

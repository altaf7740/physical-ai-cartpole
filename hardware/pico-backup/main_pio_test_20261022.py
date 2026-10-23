from machine import Pin
import rp2
import utime


# RP2040 PIO HARDWARE DRIVER
@rp2.asm_pio(out_init=(rp2.PIO.OUT_LOW,) * 4, out_shiftdir=rp2.PIO.SHIFT_RIGHT)
def stepper_pio():
  pull(block)
  out(pins, 4)


sm = rp2.StateMachine(0, stepper_pio, freq=10_000_000, out_base=Pin(14))
sm.active(1)

# --- MECHANICAL CONFIGURATION ---
BELT_PITCH_MM = 2.0  # GT2 belt = 2mm pitch
PULLEY_TEETH = 20  # 20T pulley = 40mm per revolution
STEPS_PER_REV = 200  # 200 full-steps per revolution

MM_PER_REV = BELT_PITCH_MM * PULLEY_TEETH  # 40.0 mm/rev
STEPS_PER_MM = STEPS_PER_REV / MM_PER_REV  # 5.0 full-steps/mm

TARGET_MM = 300.0
TOTAL_STEPS = int(round(TARGET_MM * STEPS_PER_MM))  # 1,500 steps

# 2-phase full-step sequence masks
forward_masks = [0b0101, 0b0110, 0b1010, 0b1001]
backward_masks = list(reversed(forward_masks))


def move_distance_smooth(
    masks, total_steps, start_hz=100, max_hz=1000, ramp_steps=600
):
  """High-velocity smoothed acceleration profile with elevated frequency cap."""
  ramp_steps = min(ramp_steps, total_steps // 2)
  decel_start = total_steps - ramp_steps
  hz_span = max_hz - start_hz

  next_time = utime.ticks_us()
  for i in range(total_steps):
    if i < ramp_steps and ramp_steps > 0:
      hz = start_hz + (hz_span * i) // ramp_steps
    elif i >= decel_start and ramp_steps > 0:
      hz = start_hz + (hz_span * (total_steps - 1 - i)) // ramp_steps
    else:
      hz = max_hz

    sm.put(masks[i & 3])
    next_time = utime.ticks_add(next_time, 1_000_000 // hz)
    while utime.ticks_diff(next_time, utime.ticks_us()) > 0:
      pass


try:
  print(
      f"Running high-velocity move ({TARGET_MM}mm) up to 1000 Hz max speed..."
  )
  while True:
    print("Moving Forward...")
    move_distance_smooth(
        forward_masks,
        TOTAL_STEPS,
        start_hz=100,
        max_hz=1000,
        ramp_steps=600,
    )
    utime.sleep_ms(300)

    print("Moving Backward...")
    move_distance_smooth(
        backward_masks,
        TOTAL_STEPS,
        start_hz=100,
        max_hz=1000,
        ramp_steps=600,
    )
    utime.sleep_ms(300)

except KeyboardInterrupt:
  print("\nStopped by user.")

finally:
  sm.put(0b0000)
  sm.active(0)
  print("Motor stopped.")
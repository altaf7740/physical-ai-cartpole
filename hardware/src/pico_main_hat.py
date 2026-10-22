# Pico firmware (MicroPython): quadrature encoder, stepper on a dual H-bridge HAT, limit switches.
# Old HAT variant: copy to src/pico_main.py and `make flash`. Not what is on the Pico.
#
# Protocol, newline-terminated ASCII, one reply per command:
#   "R"          -> zero counters, coils off; reply "0,0,0"
#   "<steps/s>"  -> signed half-step rate;   reply "<enc_ticks>,<cart_steps>,<lim_hit>"
# Watchdog: stepping stops if no command arrives for WATCHDOG_MS (host crashed/unplugged).
#
# Wiring (driver: SmartElex / Cytron HAT-MDD10 style, sign-magnitude PWM + DIR per channel):
#   Pico GP14 -> PWM1   Pico GP15 -> DIR1      stepper coil A -> M1A / M1B
#   Pico GP18 -> PWM2   Pico GP19 -> DIR2      stepper coil B -> M2A / M2B
#   Pico GND  -> HAT logic GND                 motor supply   -> VM / GND

import sys
import uselect
import utime
from machine import Pin, PWM

# --- Encoder (GP2 = A, GP3 = B) ---
ENC_A = Pin(2, Pin.IN)
ENC_B = Pin(3, Pin.IN)

# --- Limit switches (GP16 left, GP17 right; active low) ---
LIM_L = Pin(16, Pin.IN, Pin.PULL_UP)
LIM_R = Pin(17, Pin.IN, Pin.PULL_UP)

# --- Motor driver: one H-bridge channel per coil ---
# PWM high -> output A/B driven by DIR; PWM low -> both outputs low, so the coil current
# keeps circulating (slow decay) and the average coil current is about duty * VM / R.
PWM1 = PWM(Pin(14))
DIR1 = Pin(15, Pin.OUT, value=0)
PWM2 = PWM(Pin(18))
DIR2 = Pin(19, Pin.OUT, value=0)
for _pwm in (PWM1, PWM2):
    _pwm.freq(20000)  # driver max is 20 kHz
    _pwm.duty_u16(0)

# The driver has no current regulation, so current is set by duty from these numbers.
# A lower COIL_OHMS or higher SUPPLY_V than reality gives *less* current, never more.
SUPPLY_V = 12.0     # VM on the driver
COIL_OHMS = 1.5     # stepper coil resistance (measure: Ohm between M1A and M1B, power off)
RUN_A = 1.0         # coil current while stepping
HOLD_A = 0.4        # coil current while standing still
MAX_DUTY = 0.5


def amps_to_duty(amps):
    return int(min(amps * COIL_OHMS / SUPPLY_V, MAX_DUTY) * 65535)


RUN_DUTY = amps_to_duty(RUN_A)
HOLD_DUTY = amps_to_duty(HOLD_A)
MIN_STEP_US = 250   # 4000 half-steps/s max
WATCHDOG_MS = 250

# Half-step sequence as (coil A, coil B) polarity: +1, -1, or 0 (off).
HALF_STEP_SEQ = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
INV_SQRT2_Q16 = 46341  # both-coil states get 1/sqrt(2) so torque stays even across steps

# Quadrature lookup indexed by (prev_state << 2) | state, with state = (A << 1) | B.
# 00 -> 10 -> 11 -> 01 -> 00 counts +1 per edge, the reverse -1. A double change
# (an edge we were too slow to see) counts 0 rather than guessing a direction.
QUAD = (0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0)

enc_ticks = 0
enc_state = (ENC_A.value() << 1) | ENC_B.value()


def enc_isr(pin):
    global enc_ticks, enc_state
    s = (ENC_A.value() << 1) | ENC_B.value()
    enc_ticks += QUAD[(enc_state << 2) | s]
    enc_state = s


coil_state = (0, 0)
coil_duty = 0


def set_coils(state, duty):
    global coil_state, coil_duty
    coil_state, coil_duty = state, duty
    a, b = state
    if a and b:
        duty = (duty * INV_SQRT2_Q16) >> 16
    DIR1.value(1 if a < 0 else 0)
    PWM1.duty_u16(duty if a else 0)
    DIR2.value(1 if b < 0 else 0)
    PWM2.duty_u16(duty if b else 0)


def set_duty(duty):
    set_coils(coil_state, duty)


def coils_off():
    set_coils((0, 0), 0)


def limit_hit():
    return 1 if (LIM_L.value() == 0 or LIM_R.value() == 0) else 0


try:
    # hard=True runs the handler immediately instead of queueing it, so fast swings don't drop edges.
    ENC_A.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=enc_isr, hard=True)
    ENC_B.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=enc_isr, hard=True)

    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)

    cart_steps = 0
    current_dir = 0
    step_delay_us = 0
    seq_idx = 0
    last_step_time = utime.ticks_us()
    last_cmd_ms = utime.ticks_ms()
    coils_off()

    while True:
        # 1. Step pulse when due
        now = utime.ticks_us()
        if current_dir != 0:
            if utime.ticks_diff(utime.ticks_ms(), last_cmd_ms) > WATCHDOG_MS:
                current_dir = 0
                step_delay_us = 0
                set_duty(HOLD_DUTY)
            elif (LIM_L.value() == 0 and current_dir == -1) or (LIM_R.value() == 0 and current_dir == 1):
                set_duty(HOLD_DUTY)
            elif utime.ticks_diff(now, last_step_time) >= step_delay_us:
                last_step_time = now
                seq_idx = (seq_idx + current_dir) & 7
                set_coils(HALF_STEP_SEQ[seq_idx], RUN_DUTY)
                cart_steps += current_dir

        # 2. Serial command
        if poll.poll(0):
            line = sys.stdin.readline().strip()
            if not line:
                continue
            if line == "R":
                enc_ticks = 0
                cart_steps = 0
                current_dir = 0
                step_delay_us = 0
                coils_off()
                last_cmd_ms = utime.ticks_ms()
                print("0,0,0")
                continue
            try:
                freq = int(line)
            except ValueError:
                continue
            last_cmd_ms = utime.ticks_ms()
            if abs(freq) < 5:
                current_dir = 0
                step_delay_us = 0
                if coil_duty:  # hold position after moving; stays unpowered after "R"
                    set_duty(HOLD_DUTY)
            else:
                current_dir = 1 if freq > 0 else -1
                step_delay_us = max(1_000_000 // abs(freq), MIN_STEP_US)
            print(f"{enc_ticks},{cart_steps},{limit_hit()}")

except Exception as e:
    coils_off()
    print("ERR," + repr(e))  # the host can't parse this, so it counts as missed replies and faults
    raise

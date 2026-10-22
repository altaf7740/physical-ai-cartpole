# Pico firmware (MicroPython): quadrature encoder + PIO step/dir pulses for a TB6600.
# Flash with `make flash` (copies this file to the Pico as :main.py).
#
# Protocol, newline-terminated ASCII, one reply per command:
#   "R"          -> zero counters, stop; reply "0,0,0"
#   "<steps/s>"  -> signed microstep rate;   reply "<enc_ticks>,<cart_steps>,<lim_hit>"
# Watchdog: stepping stops if no command arrives for WATCHDOG_MS (host crashed/unplugged).
#
# Wiring (TB6600 inputs common cathode: PUL-, DIR-, ENA- to Pico GND):
#   Pico GP16 -> PUL+   Pico GP17 -> DIR+   ENA+ open (driver always enabled, holds torque at rest)
#   Pico GP2 / GP3 -> encoder A / B
#   No limit switches are wired, so lim_hit is always 0.
# Microstepping and current are set on the TB6600 DIP switches (1/16 = 3200 steps/rev);
# [hardware] steps_per_meter in config.toml must match.
#
# The PIO does the step timing: each FIFO word carries the direction (bit 0) and the low
# time before the next pulse (upper 31 bits, in PIO cycles at 1 MHz). DIR is set first and
# then held for that low time, which covers the TB6600's DIR setup time; PUL is 5 us high.
# The CPU only keeps the 8-deep FIFO topped up, so serial I/O and encoder interrupts never
# delay or starve the stepping.

import sys
import uselect
import utime
import rp2
from machine import Pin


@rp2.asm_pio(
    set_init=rp2.PIO.OUT_LOW,
    out_init=rp2.PIO.OUT_LOW,
    out_shiftdir=rp2.PIO.SHIFT_RIGHT,
    fifo_join=rp2.PIO.JOIN_TX,
)
def step_dir_pio():
    pull(block)
    out(pins, 1)          # DIR
    out(x, 31)
    label("low")
    jmp(x_dec, "low")     # low time, also DIR setup time
    set(pins, 1) [4]      # PUL high for 5 cycles
    set(pins, 0)


# --- Encoder (GP2 = A, GP3 = B) ---
# Pull-ups for open-collector encoders; harmless on push-pull 3.3 V outputs.
ENC_A = Pin(2, Pin.IN, Pin.PULL_UP)
ENC_B = Pin(3, Pin.IN, Pin.PULL_UP)

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


# --- Stepper (TB6600) ---
PUL_PIN = 16
DIR_PIN = 17
PIO_HZ = 1_000_000   # 1 cycle = 1 us
PIO_OVERHEAD = 10    # pull + 2x out + final jmp + 5 high + 1 low
FIFO_DEPTH = 8
MAX_HZ = 20000       # TB6600 input limit; 375 RPM at 1/16
START_HZ = 2400      # start/stop rate, no ramp below it (0.75 rev/s at 1/16)
RAMP = 64            # Hz gained or lost per step
WATCHDOG_MS = 250


sm = rp2.StateMachine(0, step_dir_pio, freq=PIO_HZ, set_base=Pin(PUL_PIN), out_base=Pin(DIR_PIN))
try:
    # hard=True runs the handler immediately instead of queueing it, so fast swings don't drop edges.
    ENC_A.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=enc_isr, hard=True)
    ENC_B.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=enc_isr, hard=True)
    sm.active(1)

    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)
    rd = sys.stdin.buffer.read

    cart_steps = 0
    target = 0
    v = 0
    line = b""
    last_cmd_ms = utime.ticks_ms()

    while True:
        # 1. Queue steps until the FIFO is full
        while sm.tx_fifo() < FIFO_DEPTH:
            if v == 0:
                if target == 0:
                    break
                s = START_HZ if abs(target) > START_HZ else abs(target)
                v = s if target > 0 else -s

            cart_steps += 1 if v > 0 else -1
            s = v if v > 0 else -v
            sm.put((1 if v > 0 else 0) | ((PIO_HZ // s - PIO_OVERHEAD) << 1))

            # Ramp toward target; when stopping or reversing, decelerate first
            if target != 0 and (target > 0) == (v > 0):
                a = target if target > 0 else -target
                if s < a:
                    s = min(s + RAMP, a)
                elif s > a:
                    s = max(s - RAMP, a)
                v = s if v > 0 else -s
            else:
                s -= RAMP
                v = 0 if s <= START_HZ else (s if v > 0 else -s)

        # 2. Watchdog
        if target != 0 and utime.ticks_diff(utime.ticks_ms(), last_cmd_ms) > WATCHDOG_MS:
            target = 0

        # 3. Serial command, read without blocking so the FIFO never runs dry
        while poll.poll(0):
            c = rd(1)
            if c != b"\n":
                if len(line) < 16:
                    line += c
                continue
            cmd, line = line.strip(), b""
            if not cmd:
                continue
            last_cmd_ms = utime.ticks_ms()
            if cmd == b"R":
                enc_ticks = 0
                cart_steps = 0
                target = 0
                v = 0
                print("0,0,0")
                continue
            try:
                freq = int(cmd)
            except ValueError:
                continue
            target = 0 if abs(freq) < 5 else max(-MAX_HZ, min(MAX_HZ, freq))
            print("%d,%d,0" % (enc_ticks, cart_steps))

except Exception as e:
    print("ERR," + repr(e))  # the host can't parse this, so it counts as missed replies and faults
    raise
finally:
    sm.active(0)
    # Hand the pins back from PIO to plain GPIO, driven low (no pulses)
    Pin(PUL_PIN, Pin.OUT, value=0)
    Pin(DIR_PIN, Pin.OUT, value=0)

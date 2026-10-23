# Pico firmware (MicroPython): quadrature encoder + PIO step/dir pulses for a TB6600.
# Flash with `make flash` (copies this file to the Pico as :main.py).
#
# Protocol, newline-terminated ASCII, one reply per command:
#   "R"          -> zero counters, stop; reply "0,0,0"
#   "<steps/s>"  -> signed microstep rate;   reply "<enc_ticks>,<cart_steps>,<lim_hit>"
#   lim_hit is a bitmask: 1 = left switch pressed, 2 = right, 3 = both (the host treats non-zero as hit).
# Watchdog: stepping stops if no command arrives for WATCHDOG_MS (host crashed/unplugged).
# Limits: no new steps are queued toward a pressed switch (hard stop, no ramp; the at most QUEUE_CHUNKS
# chunks already queued, ~3 ms of travel at speed, still run); moving away is allowed.
# Negative steps move toward the left switch.
#
# Wiring (TB6600 inputs common cathode: PUL-, DIR-, ENA- to Pico GND):
#   Pico GP16 -> PUL+   Pico GP17 -> DIR+   ENA+ open (driver always enabled, holds torque at rest)
#   Pico GP14 / GP15 -> encoder A / B (index channel C/Z not connected)
#   Pico GP13 / GP1 -> left / right limit switch, C + NC to GND (internal pull-up): closed = low at rest,
#   pressed = open = high. A broken wire also reads high, so it stops the cart instead of going unnoticed.
# Microstepping and current are set on the TB6600 DIP switches (1/16 = 3200 steps/rev);
# [hardware] steps_per_meter in config.toml must match.
#
# The PIO does the step timing. Each FIFO word is a chunk of steps at one rate: direction
# (bit 0), steps - 1 (bits 1-7, up to 128 steps) and the per-step period minus PIO_OVERHEAD
# (bits 8-31, in PIO cycles at 1 MHz). DIR is set first and then held for the low time
# before the first pulse, which covers the TB6600's DIR setup time; PUL is 5 us high.
# A chunk lasts about CHUNK_US and the ramp advances once per chunk, so the CPU does one
# FIFO put per millisecond instead of one per step. Every pass of the main loop queues at
# most one chunk and then services the watchdog and serial, so no step rate can starve them
# (one put per step could not keep up above ~10 kHz and blocked serial until a limit switch).

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
    out(y, 7)             # steps in this chunk - 1
    out(isr, 24)          # per-step delay, reloaded into x for every step
    label("step")
    mov(x, isr)
    label("low")
    jmp(x_dec, "low")     # low time, also DIR setup time
    set(pins, 1) [4]      # PUL high for 5 cycles
    set(pins, 0)
    jmp(y_dec, "step")


# --- Encoder (GP14 = A, GP15 = B; C/Z unused) ---
# Pull-ups for open-collector encoders; harmless on push-pull 3.3 V outputs.
ENC_A = Pin(14, Pin.IN, Pin.PULL_UP)
ENC_B = Pin(15, Pin.IN, Pin.PULL_UP)

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


# --- Limit switches (GP13 left, GP1 right; normally closed to GND, so pressed = high) ---
LIM_L = Pin(13, Pin.IN, Pin.PULL_UP)
LIM_R = Pin(1, Pin.IN, Pin.PULL_UP)


def limit_hit():
    return LIM_L.value() | (LIM_R.value() << 1)


# --- Stepper (TB6600) ---
PUL_PIN = 16
DIR_PIN = 17
DIR_POSITIVE = 0     # DIR level for positive steps (away from the left switch); flip if they move toward it
PIO_HZ = 1_000_000   # 1 cycle = 1 us
PIO_OVERHEAD = 9     # cycles per step besides the delay loop: mov + 1 + 5 high + 1 low + jmp
MAX_HZ = 20000       # TB6600 input limit; 375 RPM at 1/16
START_HZ = 2400      # start/stop rate, no ramp below it (0.75 rev/s at 1/16)
ACCEL = 400_000      # steps/s^2 for ramping up, down and reversing (5 m/s^2 at 1/16, 80000 steps/m)
CHUNK_US = 1000      # target chunk duration; the ramp advances once per chunk
MAX_CHUNK = 128      # steps per chunk (7-bit field)
QUEUE_CHUNKS = 3     # chunks kept queued in the PIO FIFO (stop latency ~3 ms at speed)
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
        # 1. Queue at most one chunk of steps per pass
        if v == 0 and target != 0:
            s = START_HZ if abs(target) > START_HZ else abs(target)
            v = s if target > 0 else -s

        if v != 0 and sm.tx_fifo() < QUEUE_CHUNKS:
            if (v < 0 and LIM_L.value()) or (v > 0 and LIM_R.value()):
                v = 0  # at a switch: stop dead; the next pass may start away from it
            else:
                s = v if v > 0 else -v
                n = s * CHUNK_US // 1_000_000
                n = 1 if n < 1 else (MAX_CHUNK if n > MAX_CHUNK else n)
                sm.put((DIR_POSITIVE if v > 0 else 1 - DIR_POSITIVE) | ((n - 1) << 1)
                       | ((PIO_HZ // s - PIO_OVERHEAD) << 8))
                cart_steps += n if v > 0 else -n

                # Ramp toward target over this chunk's duration (n / s); when stopping or reversing,
                # decelerate first
                dv = ACCEL * n // s
                if dv < 1:
                    dv = 1
                if target != 0 and (target > 0) == (v > 0):
                    a = target if target > 0 else -target
                    if s < a:
                        s = min(s + dv, a)
                    elif s > a:
                        s = max(s - dv, a)
                    v = s if v > 0 else -s
                else:
                    s -= dv
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
                target = 0
                v = 0
                t0 = utime.ticks_ms()
                # let queued chunks finish so the zeroed count matches the cart; stays well under the
                # host's 50 ms wait after "R" (pico_link.reset_counters)
                while sm.tx_fifo() and utime.ticks_diff(utime.ticks_ms(), t0) < 20:
                    pass
                utime.sleep_ms(2)
                enc_ticks = 0
                cart_steps = 0
                print("0,0,%d" % limit_hit())
                continue
            try:
                freq = int(cmd)
            except ValueError:
                continue
            target = 0 if abs(freq) < 5 else max(-MAX_HZ, min(MAX_HZ, freq))
            print("%d,%d,%d" % (enc_ticks, cart_steps, limit_hit()))

except Exception as e:
    print("ERR," + repr(e))  # the host can't parse this, so it counts as missed replies and faults
    raise
finally:
    sm.active(0)
    # Hand the pins back from PIO to plain GPIO, driven low (no pulses)
    Pin(PUL_PIN, Pin.OUT, value=0)
    Pin(DIR_PIN, Pin.OUT, value=0)

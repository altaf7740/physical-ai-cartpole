# Pico firmware (MicroPython): quadrature encoder + PIO half-step stepper on an L298N.
# L298N variant: copy to src/pico_main.py and `make flash`. Not what is on the Pico.
#
# Protocol, newline-terminated ASCII, one reply per command:
#   "R"          -> zero counters, coils off; reply "0,0,0"
#   "<steps/s>"  -> signed half-step rate;   reply "<enc_ticks>,<cart_steps>,<lim_hit>"
# Watchdog: stepping stops if no command arrives for WATCHDOG_MS (host crashed/unplugged).
#
# Wiring:
#   Pico GP14..GP17 -> L298N IN1..IN4 (ENA/ENB jumpers on), coils on OUT1-2 / OUT3-4
#   Pico GP2 / GP3  -> encoder A / B
#   No limit switches are wired (GP16/17 are motor pins now), so lim_hit is always 0.
#
# Stepping is the PIO program from ~/Developer/nema-speed-ui/pico_stepper.py: each FIFO
# word carries the coil pattern (low 4 bits) and how long to hold it (upper 28 bits, in
# PIO cycles at 1 MHz). The CPU only keeps the 8-deep FIFO topped up, so serial I/O and
# encoder interrupts never delay or starve the stepping.

import sys
import uselect
import utime
import rp2
from machine import Pin


@rp2.asm_pio(
    out_init=(rp2.PIO.OUT_LOW,) * 4,
    out_shiftdir=rp2.PIO.SHIFT_RIGHT,
    fifo_join=rp2.PIO.JOIN_TX,
)
def stepper_pio():
    pull(block)
    out(pins, 4)
    out(x, 28)
    label("hold")
    jmp(x_dec, "hold")


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


# --- Stepper ---
# Half-step table as (IN1, IN2, IN3, IN4) = (A+, A-, B+, B-), from nemaaaaaa.py.
SEQ = (
    (1, 0, 0, 0),
    (1, 0, 1, 0),
    (0, 0, 1, 0),
    (0, 1, 1, 0),
    (0, 1, 0, 0),
    (0, 1, 0, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
)
MASKS = tuple(sum(b << i for i, b in enumerate(s)) for s in SEQ)

PIO_HZ = 1_000_000   # 1 cycle = 1 us
PIO_OVERHEAD = 4     # pull + 2x out + final jmp
FIFO_DEPTH = 8
MAX_HZ = 5000        # half-steps/s, tested on this rig with the speed UI
START_HZ = 300       # start/stop rate, no ramp below it
RAMP = 8             # Hz gained or lost per half-step
WATCHDOG_MS = 250


def coils_off(sm):
    sm.put(10 << 4)  # pattern 0 held for a few us, then the PIO waits with all IN pins low


sm = rp2.StateMachine(0, stepper_pio, freq=PIO_HZ, out_base=Pin(14))
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
    idx = 1
    energized = False
    line = b""
    last_cmd_ms = utime.ticks_ms()

    while True:
        # 1. Queue steps until the FIFO is full
        while sm.tx_fifo() < FIFO_DEPTH:
            if v == 0:
                if target == 0:
                    if energized:
                        coils_off(sm)
                        energized = False
                    break
                s = START_HZ if abs(target) > START_HZ else abs(target)
                v = s if target > 0 else -s

            step = 1 if v > 0 else -1
            idx = (idx + step) & 7
            cart_steps += step
            s = v if v > 0 else -v
            sm.put(MASKS[idx] | ((PIO_HZ // s - PIO_OVERHEAD) << 4))
            energized = True

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
    # Hand the pins back from PIO to plain GPIO, driven low (coils off)
    for p in (14, 15, 16, 17):
        Pin(p, Pin.OUT, value=0)

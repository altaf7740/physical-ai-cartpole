import sys
import uselect
from machine import Pin, PWM
import utime

try:
    # --- 1. Encoder Setup (GP2, GP3) ---
    ENC_A = Pin(2, Pin.IN)
    ENC_B = Pin(3, Pin.IN)

    # --- 2. Limit Switches (GP16, GP17) ---
    LIM_L = Pin(16, Pin.IN, Pin.PULL_UP)
    LIM_R = Pin(17, Pin.IN, Pin.PULL_UP)

    # --- 3. L298N Pins ---
    IN1 = Pin(14, Pin.OUT)
    IN2 = Pin(15, Pin.OUT)
    IN3 = Pin(18, Pin.OUT)
    IN4 = Pin(19, Pin.OUT)

    ENA = PWM(Pin(13))
    ENB = PWM(Pin(20))
    ENA.freq(20000)
    ENB.freq(20000)

    RUN_DUTY  = 14000
    HOLD_DUTY = 6500

    HALF_STEP_SEQ = (
        (1, 0, 0, 0),
        (1, 0, 1, 0),
        (0, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
    )

    enc_ticks = 0
    cart_steps = 0
    current_dir = 0
    step_delay_us = 0
    seq_idx = 0

    last_a = ENC_A.value()
    last_b = ENC_B.value()

    def set_coils(state):
        IN1.value(state[0])
        IN2.value(state[1])
        IN3.value(state[2])
        IN4.value(state[3])

    def enc_isr(pin):
        global enc_ticks, last_a, last_b
        a = ENC_A.value()
        b = ENC_B.value()
        if a != last_a or b != last_b:
            if a == b:
                enc_ticks += 1 if a != last_b else -1
            else:
                enc_ticks += 1 if a == last_b else -1
            last_a = a
            last_b = b

    ENC_A.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=enc_isr)
    ENC_B.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=enc_isr)

    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)

    last_step_time = utime.ticks_us()
    ENA.duty_u16(HOLD_DUTY)
    ENB.duty_u16(HOLD_DUTY)

    while True:
        # 1. Non-blocking Stepper Pulse Check
        now = utime.ticks_us()
        if step_delay_us > 0 and current_dir != 0:
            if (LIM_L.value() == 0 and current_dir == -1) or (LIM_R.value() == 0 and current_dir == 1):
                ENA.duty_u16(HOLD_DUTY)
                ENB.duty_u16(HOLD_DUTY)
            elif utime.ticks_diff(now, last_step_time) >= step_delay_us:
                last_step_time = now
                ENA.duty_u16(RUN_DUTY)
                ENB.duty_u16(RUN_DUTY)
                seq_idx = (seq_idx + current_dir) & 7
                set_coils(HALF_STEP_SEQ[seq_idx])
                cart_steps += current_dir

        # 2. Non-blocking Serial Check
        if poll.poll(0):
            line = sys.stdin.readline().strip()
            if not line:
                continue
            if line == "R":
                enc_ticks = 0
                cart_steps = 0
                current_dir = 0
                step_delay_us = 0
                ENA.duty_u16(0)
                ENB.duty_u16(0)
                set_coils((0, 0, 0, 0))
                print("0,0,0")
            else:
                try:
                    freq = int(line)
                    if abs(freq) < 5:
                        current_dir = 0
                        step_delay_us = 0
                        ENA.duty_u16(HOLD_DUTY)
                        ENB.duty_u16(HOLD_DUTY)
                    else:
                        current_dir = 1 if freq > 0 else -1
                        step_delay_us = max(int(1_000_000 / abs(freq)), 250)
                    lim_hit = 1 if (LIM_L.value() == 0 or LIM_R.value() == 0) else 0
                    print(f"{enc_ticks},{cart_steps},{lim_hit}")
                except ValueError:
                    pass

except Exception as e:
    # Fallback error print so serial always gets valid debug info instead of crashing silently
    print(f"0,0,0")

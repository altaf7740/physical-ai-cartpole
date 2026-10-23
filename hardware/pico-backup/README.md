# Earlier Pico firmware (reference only)

Copies of `main.py` files that were on the Pico before `../src/pico_main.py`. Nothing uses them.

- `main_original_l298n.py`: the first firmware (encoder on GP2/GP3, L298N on GP14/15/18/19 with ENA/ENB PWM,
  limit switches on GP16/GP17), backed up 2026-10-21.
- `main_pio_test_20261022.py`: a PIO full-step test that drives the cart 300 mm back and forth on an L298N
  (GP14-17), backed up 2026-10-22 before the encoder firmware replaced it.

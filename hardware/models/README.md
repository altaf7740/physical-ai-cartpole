# models/

Placeholder for the trained policy. Model files are git-ignored.

Put **one** of these here and point `[policy] model_path` in `config.toml` at it:

| File | Backend | Where it comes from |
|---|---|---|
| `policy.onnx` | `onnx` (onnxruntime) | Isaac Lab `play.py` writes `logs/<lib>/<task>/<run>/exported/policy.onnx` |
| `policy.npz` | `numpy` | Arrays `W0, b0, W1, b1, ...` (weights shaped `(out, in)`), optional `obs_mean`, `obs_std`, `activation` |

The model must take `len(obs_order)` inputs and give 1 output (the cart action).

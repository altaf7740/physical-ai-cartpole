"""Load the trained policy and run inference.

Two backends:
  * onnx  — the `exported/policy.onnx` written by Isaac Lab's play.py (needs onnxruntime)
  * numpy — an `.npz` MLP (W0, b0, W1, b1, ...; torch layout, weights shaped
            (out, in)), optional `obs_mean` / `obs_std`, optional `activation`
            ("elu" default, matching rsl_rl). No dependencies beyond numpy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from config import PolicyConfig


class Policy(Protocol):
    obs_dim: int
    action_dim: int

    def act(self, obs: np.ndarray) -> np.ndarray: ...


class OnnxPolicy:
    def __init__(self, path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # tiny MLP: threading only adds jitter
        self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        self._input_name = inp.name
        self.obs_dim = int(inp.shape[-1])
        self.action_dim = int(out.shape[-1])

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self._session.run(None, {self._input_name: x})[0].reshape(-1)


_ACTIVATIONS = {
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))),
    "relu": lambda x: np.maximum(x, 0),
    "tanh": np.tanh,
}


class NumpyMlpPolicy:
    def __init__(self, path: Path):
        data = np.load(path)
        n_layers = sum(1 for k in data.files if k.startswith("W"))
        if n_layers == 0:
            raise ValueError(f"{path}: no W0/b0... arrays found")
        self._weights = [data[f"W{i}"].astype(np.float32) for i in range(n_layers)]
        self._biases = [data[f"b{i}"].astype(np.float32) for i in range(n_layers)]
        activation = str(data["activation"]) if "activation" in data.files else "elu"
        self._act = _ACTIVATIONS[activation]
        self._mean = data["obs_mean"].astype(np.float32) if "obs_mean" in data.files else None
        self._std = data["obs_std"].astype(np.float32) if "obs_std" in data.files else None
        self.obs_dim = self._weights[0].shape[1]
        self.action_dim = self._weights[-1].shape[0]

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._mean is not None:
            x = (x - self._mean) / (self._std + 1e-8)
        last = len(self._weights) - 1
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = w @ x + b
            if i < last:
                x = self._act(x)
        return x


def load_policy(cfg: PolicyConfig) -> Policy:
    path = Path(cfg.model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"model not found at {path} — copy your exported policy there (see models/README.md)"
        )
    backend = cfg.backend
    if backend == "auto":
        backend = "numpy" if path.suffix == ".npz" else "onnx"
    policy: Policy = NumpyMlpPolicy(path) if backend == "numpy" else OnnxPolicy(path)

    if policy.obs_dim != len(cfg.obs_order):
        raise ValueError(
            f"model expects {policy.obs_dim} observations but obs_order has {len(cfg.obs_order)}"
        )
    if policy.action_dim != 1:
        raise ValueError(f"expected a single cart action, model outputs {policy.action_dim}")
    return policy

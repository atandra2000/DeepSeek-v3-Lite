"""Centralize the opt-in policy for Triton model backends.

Unset or disabled environments force Triton configuration keys to their
portable PyTorch defaults and emit one startup warning.
"""
from __future__ import annotations

import os
from typing import Callable


# Each entry maps a Triton config value to its portable default.
_DISPATCH = {
    ("attn_impl",    "triton"):         "sdpa",
    ("moe_dispatch", "triton_grouped"): "stacked",
}


def enforce_triton_env_var(model_cfg: dict, log: Callable[[str], None]) -> None:
    """Force any 'triton' dispatch back to its PyTorch default unless the
    master env-var is set. No-op when `ENABLE_TRITON_KERNELS=1`.
    """
    if os.environ.get("ENABLE_TRITON_KERNELS", "0") == "1":
        return
    forced = []
    for (key, triton_val), pytorch_val in _DISPATCH.items():
        if model_cfg.get(key) == triton_val:
            model_cfg[key] = pytorch_val
            forced.append(f"{key}='{triton_val}' -> '{pytorch_val}'")
    if forced:
        log(
            "[warn] Triton dispatch keys set without ENABLE_TRITON_KERNELS=1; "
            f"forcing {', '.join(forced)}. "
            "Set ENABLE_TRITON_KERNELS=1 to enable the fused Triton paths."
        )


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (Triton needs pow2 tl.arange lengths)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

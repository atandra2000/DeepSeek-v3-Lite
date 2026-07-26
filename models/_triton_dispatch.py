"""Triton dispatch contract: force-back guard for `ENABLE_TRITON_KERNELS`.

Without `ENABLE_TRITON_KERNELS=1`, `enforce_triton_env_var` rewrites any
Triton dispatch key (`attn_impl='triton'`, `moe_dispatch='triton_grouped'`)
to its PyTorch default with a single warning — never per-layer.
"""
from __future__ import annotations

import os
from typing import Callable


# Triton dispatch keys + their PyTorch defaults. A config with the
# Triton value for any of these is force-backed to the PyTorch default
# when ENABLE_TRITON_KERNELS != '1'.
# Single tuple table — earlier _TRITON_DISPATCH_KEYS + _PYTORCH_DEFAULTS pair
# was enforced in lockstep via a dedicated test; one source of truth now.
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

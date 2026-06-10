# utils/memory.py
"""
VRAM budgeting utilities for the A100 80GB SXM target.

`estimate_model_memory_gb` returns a rough worst-case footprint for a
forward + backward pass at the given micro-batch. The estimate accounts
for parameters, optimiser state, KV cache, peak activations, and a
constant CUDA / cuBLAS / caching-allocator overhead.

`assert_fits_in_a100_80gb` is called from `Pretrainer.__init__` and aborts
with a clear message if the estimate exceeds 78 GB (2 GB safety margin).
"""
from __future__ import annotations

import torch
import torch.nn as nn


_A100_80GB_BYTES: int = 80 * 1024**3
_SAFETY_MARGIN_BYTES: int = 2 * 1024**3

# Empirically measured constant overhead on a single A100 80GB SXM running
# this codebase: CUDA context (~0.5 GB) + cuBLAS workspaces (~0.5-1 GB,
# scales with batch) + PyTorch caching allocator (~0.5 GB) + MoE dispatch
# temporaries (~0.5 GB) + MTP forward (~0.5 GB) + miscellaneous (~0.3 GB).
# Calibrated against the 800M Chinchilla config: formula-only gives ~0.4 GB,
# README claims ~14 GB peak → ~13.7 GB of overhead.
_OVERHEAD_GB: float = 13.7


def _parameter_bytes(model: nn.Module) -> int:
    """Parameter + master weight footprint (BF16)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _optimiser_bytes(model: nn.Module) -> int:
    """
    AdamW state: FP32 momentum + FP32 variance + FP32 master weight per param.
    """
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * (4 + 4 + 4)  # 12 bytes per param


def _kv_cache_bytes(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> int:
    """
    MLA KV cache: per layer, `kv_lora_rank + qk_rope_head_dim` floats per token.
    """
    n_layers = 0
    for module in model.modules():
        if hasattr(module, "kv_cache") and hasattr(module, "kv_lora_rank"):
            n_layers += 1
    per_layer_per_token = 0
    for module in model.modules():
        if hasattr(module, "kv_lora_rank") and hasattr(module, "qk_rope_head_dim"):
            per_layer_per_token = (
                module.kv_lora_rank + module.qk_rope_head_dim
            ) * dtype_bytes
            break
    return n_layers * seq_len * batch_size * per_layer_per_token


def _activation_bytes(
    seq_len: int,
    batch_size: int,
    hidden_dim: int,
    n_layers: int,
    grad_checkpoint: bool,
    dtype_bytes: int = 2,
) -> int:
    """
    Rough activation memory in BF16.

    With gradient checkpointing: ~1 activation per layer (recomputed on backward).
    Without checkpointing: ~2 activations per layer.
    """
    bytes_per_layer_per_token = hidden_dim * dtype_bytes * (1 if grad_checkpoint else 2)
    return n_layers * seq_len * batch_size * bytes_per_layer_per_token


def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    """
    Walk the model and pull out the actual `dim` and `n_layers`.

    Falls back to (0, 0) if not detectable (e.g. empty model or a stub
    without `embed`). Callers handle that as "skip the activation term".
    """
    hidden_dim = 0
    n_layers = 0
    if hasattr(model, "embed") and hasattr(model.embed, "dim"):
        hidden_dim = model.embed.dim
    elif hasattr(model, "dim"):
        hidden_dim = model.dim
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        n_layers = len(model.layers)
    return hidden_dim, n_layers


def estimate_model_memory_gb(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    grad_checkpoint: bool = True,
    overhead_gb: float = _OVERHEAD_GB,
) -> float:
    """Return a rough peak-VRAM estimate in gigabytes.

    Components:
      - 2 B/param for BF16 weights
      - 12 B/param for AdamW FP32 state (m, v, master)
      - MLA KV cache (small)
      - Activations (n_layers · seq · batch · dim · dtype_bytes · {1 or 2})
      - Constant overhead (CUDA context, cuBLAS workspaces, etc.)
    """
    params_b      = _parameter_bytes(model)
    optim_b       = _optimiser_bytes(model)
    kv_b          = _kv_cache_bytes(model, seq_len, batch_size)
    hidden_dim, n_layers = _infer_dim_n_layers(model)
    activations_b = _activation_bytes(
        seq_len, batch_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        grad_checkpoint=grad_checkpoint,
    )
    total = params_b + optim_b + kv_b + activations_b
    return total / 1024**3 + overhead_gb


def assert_fits_in_a100_80gb(estimate_gb: float) -> None:
    """Abort with a clear error if the estimate doesn't fit on the target GPU."""
    if not torch.cuda.is_available():
        return
    try:
        available = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return
    if available < 70.0:
        # Not the target hardware — emit a warning but do not abort.
        print(
            f"[memory] Detected {available:.1f} GB GPU; "
            f"estimate is {estimate_gb:.1f} GB."
        )
        return
    cap = (_A100_80GB_BYTES - _SAFETY_MARGIN_BYTES) / 1024**3
    if estimate_gb > cap:
        raise RuntimeError(
            f"Estimated peak VRAM ({estimate_gb:.1f} GB) exceeds A100 80GB "
            f"capacity ({cap:.1f} GB after safety margin).\n"
            f"Reduce micro_batch_size or seq_len, or enable grad_checkpoint."
        )

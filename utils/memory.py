# utils/memory.py
"""
VRAM budgeting utilities.

`estimate_model_memory_gb` returns a rough worst-case footprint for a
forward + backward pass at the given micro-batch. The estimate accounts
for parameters, optimiser state, KV cache, peak activations, and a
GPU-dependent CUDA / cuBLAS / caching-allocator overhead.

`assert_fits_in_available_gpu` is called from `Pretrainer.__init__` and aborts
if the estimate exceeds available GPU memory minus safety margin.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


def _detect_overhead_gb() -> float:
    """
    Return a GPU-appropriate overhead estimate based on available VRAM.
    Calibrated for A100 80GB (~13.7 GB overhead), scales down for consumer GPUs.
    """
    if not torch.cuda.is_available():
        return 2.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Overhead scales with GPU memory: workspaces, allocator, MoE buffers, etc.
    return min(13.7, max(2.0, total_gb * 0.15))


def estimate_model_memory_gb(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    grad_checkpoint: bool = True,
    overhead_gb: float | None = None,
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
    if overhead_gb is None:
        overhead_gb = _detect_overhead_gb()
    total = params_b + optim_b + kv_b + activations_b
    return total / 1024**3 + overhead_gb


def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 2.0) -> None:
    """Abort with a clear error if the estimate doesn't fit on the available GPU."""
    if not torch.cuda.is_available():
        return
    try:
        available = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return
    if estimate_gb > available - safety_margin_gb:
        raise RuntimeError(
            f"Estimated peak VRAM ({estimate_gb:.1f} GB) exceeds available "
            f"GPU memory ({available:.1f} GB, {safety_margin_gb:.1f} GB margin).\n"
            f"Reduce micro_batch_size or seq_len, or enable grad_checkpoint."
        )
    print(
        f"[memory] Estimated peak VRAM: {estimate_gb:.1f} GB / "
        f"{available:.1f} GB available — OK."
    )




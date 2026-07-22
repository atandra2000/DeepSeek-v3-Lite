"""Memory-budget estimation for DeepSeek-V3-Lite training (CPU-friendly arithmetic helpers).

Component bytes (BF16, AdamW FP32): params = 2×N, optim = 12×N, KV = batch·seq·(kv_lora+qk_rope)·2,
activations = 24×B·S·D·L (with grad-ckpt) or 36× (without). Used by `scripts/microbench_a100.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Approx peak overhead from CUDA context + NCCL + caching allocator
# (A100 80GB, PyTorch 2.x). Empirically <= 17% of device total.
STATIC_PYTORCH_OVERHEAD_GB = 13.7


def _deduped_numel(model: nn.Module) -> int:
    """Total parameter count with shared tensors (weight tying) counted once."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        total += p.numel()
    return total


def _parameter_bytes(model: nn.Module) -> int:
    """Model weights in BF16 (2 bytes per param)."""
    return _deduped_numel(model) * 2


def _optimiser_bytes(model: nn.Module) -> int:
    """AdamW state: FP32 master copy (4) + first moment (4) + second moment (4) = 12 bytes/param."""
    return _deduped_numel(model) * 12


def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2) -> int:
    """Total MLA KV-cache bytes: `batch·seq·(kv_lora_rank + qk_rope_head_dim)·dtype_bytes` summed over layers."""
    total = 0
    layers = list(model.layers) if hasattr(model, "layers") else []
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        kv_lora_rank = getattr(attn, "kv_lora_rank", 0)
        qk_rope_head_dim = getattr(attn, "qk_rope_head_dim", 0)
        per_layer = batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
        total += per_layer
    return total


def _activation_bytes(
    seq_len: int, batch_size: int, hidden_dim: int, n_layers: int,
    grad_checkpoint: bool, dtype_bytes: int = 2,
) -> int:
    """Approx peak activation memory.

    Standard transformer: ~24 * B * S * D * L * dtype_bytes with grad
    checkpointing, ~36 * B * S * D * L * dtype_bytes without. The 24/36
    constants mirror the PaLM formula (PaLM Appendix A).
    """
    factor = 24 if grad_checkpoint else 36
    return factor * batch_size * seq_len * hidden_dim * n_layers * dtype_bytes


def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    """(hidden_dim, n_layers) for a `Transformer`-shaped model, (0, 0) for stubs."""
    if hasattr(model, "embed") and hasattr(model.embed, "embedding_dim"):
        dim = int(model.embed.embedding_dim)
    else:
        dim = 0
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        n_layers = len(model.layers)
    else:
        n_layers = 0
    return dim, n_layers


def _detect_overhead_gb() -> float:
    """PyTorch + CUDA context overhead in GB. CPU: 2 GB. CUDA: min(13.7, 0.17*total)."""
    if not torch.cuda.is_available():
        return 2.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return min(STATIC_PYTORCH_OVERHEAD_GB, max(2.0, total_gb * 0.17))


def estimate_model_memory_gb(
    model: nn.Module, seq_len: int, batch_size: int,
    grad_checkpoint: bool = True, overhead_gb: float | None = None,
    dtype_bytes: int = 2, inference: bool = False,
) -> float:
    """Sum of params + optim + kv + activations + overhead, in GB.

    `overhead_gb=None` autodetects via `_detect_overhead_gb()`. `inference=True`
    drops the optimiser and activation bytes — at inference we don't carry
    the AdamW state, and the forward-only activations are dominated by
    the KV cache anyway.
    """
    if overhead_gb is None:
        overhead_gb = _detect_overhead_gb()
    hidden_dim, n_layers = _infer_dim_n_layers(model)
    bytes_total = (
        _parameter_bytes(model)
        + _kv_cache_bytes(model, seq_len, batch_size, dtype_bytes)
    )
    if not inference:
        bytes_total += (
            _optimiser_bytes(model)
            + _activation_bytes(seq_len, batch_size, hidden_dim, n_layers,
                                grad_checkpoint, dtype_bytes)
        )
    return (bytes_total / 1024**3) + overhead_gb


def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 0.0) -> None:
    """No-op on CPU. On CUDA, raise RuntimeError if estimate > available - margin."""
    if not torch.cuda.is_available():
        return
    available_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if estimate_gb > available_gb - safety_margin_gb:
        raise RuntimeError(
            f"estimate {estimate_gb:.2f} GB exceeds available {available_gb:.2f} GB "
            f"minus safety margin {safety_margin_gb:.2f} GB"
        )

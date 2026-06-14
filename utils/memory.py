"""VRAM budgeting: estimate peak memory for forward+backward and assert fit on GPU."""
from __future__ import annotations
import torch
import torch.nn as nn


def _parameter_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _optimiser_bytes(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) * 12  # FP32 m,v,master


def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2) -> int:
    n_layers = sum(1 for m in model.modules() if hasattr(m, "kv_cache") and hasattr(m, "kv_lora_rank"))
    for m in model.modules():
        if hasattr(m, "kv_lora_rank") and hasattr(m, "qk_rope_head_dim"):
            per_token = (m.kv_lora_rank + m.qk_rope_head_dim) * dtype_bytes
            break
    else:
        per_token = 0
    return n_layers * seq_len * batch_size * per_token


def _activation_bytes(seq_len: int, batch_size: int, hidden_dim: int, n_layers: int, grad_checkpoint: bool, dtype_bytes: int = 2) -> int:
    factor = 1 if grad_checkpoint else 2
    return n_layers * seq_len * batch_size * hidden_dim * dtype_bytes * factor


def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    hd = getattr(model, "dim", 0)
    if hasattr(model, "embed") and hasattr(model.embed, "dim"):
        hd = model.embed.dim
    nl = len(model.layers) if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList) else 0
    return hd, nl


def _detect_overhead_gb() -> float:
    if not torch.cuda.is_available():
        return 2.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return min(13.7, max(2.0, total_gb * 0.17))


def estimate_model_memory_gb(model: nn.Module, seq_len: int, batch_size: int, grad_checkpoint: bool = True, overhead_gb: float | None = None) -> float:
    params_b = _parameter_bytes(model)
    optim_b = _optimiser_bytes(model)
    kv_b = _kv_cache_bytes(model, seq_len, batch_size)
    hd, nl = _infer_dim_n_layers(model)
    act_b = _activation_bytes(seq_len, batch_size, hidden_dim=hd, n_layers=nl, grad_checkpoint=grad_checkpoint)
    total = params_b + optim_b + kv_b + act_b
    return total / 1024**3 + (overhead_gb if overhead_gb is not None else _detect_overhead_gb())


def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 2.0) -> None:
    if not torch.cuda.is_available():
        return
    try:
        available = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return
    if estimate_gb > available - safety_margin_gb:
        raise RuntimeError(f"Estimated peak VRAM ({estimate_gb:.1f} GB) exceeds available GPU memory ({available:.1f} GB, {safety_margin_gb:.1f} GB margin).")
    print(f"[memory] Estimated peak VRAM: {estimate_gb:.1f} GB / {available:.1f} GB — OK.")

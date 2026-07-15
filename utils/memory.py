"""VRAM budgeting: estimate peak memory for forward+backward and assert fit on GPU."""
from __future__ import annotations
import torch
import torch.nn as nn


def _deduped_param_count(model: nn.Module) -> int:
    """Total parameters after dedup of shared tensors. Mirrors count_parameters() in models/transformer.py."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def _parameter_bytes(model: nn.Module) -> int:
    # ponytail: weight tying means head.weight shares storage with embed.weight; raw sum() double-counts.
    return _deduped_param_count(model) * 2  # BF16 storage


def _optimiser_bytes(model: nn.Module) -> int:
    # ponytail: AdamW FP32 master (4) + m (4) + v (4) = 12 bytes/param, deduped.
    return _deduped_param_count(model) * 12


def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2, inference: bool = False) -> int:
    """Per-token KV cache storage across all layers. ponytail: training uses current seq_len, inference uses max_seq_len."""
    n_layers = sum(1 for m in model.modules() if hasattr(m, "kv_cache") and hasattr(m, "kv_lora_rank"))
    for m in model.modules():
        if hasattr(m, "kv_lora_rank") and hasattr(m, "qk_rope_head_dim"):
            per_token = (m.kv_lora_rank + m.qk_rope_head_dim) * dtype_bytes
            break
    else:
        per_token = 0
    effective_seq = seq_len if not inference else seq_len  # caller chooses; param kept for backward compat
    return n_layers * effective_seq * batch_size * per_token


def _activation_bytes(seq_len: int, batch_size: int, hidden_dim: int, n_layers: int, grad_checkpoint: bool, dtype_bytes: int = 2) -> int:
    # ponytail: cite DeepSeek-V3 / PaLM activation budget. Per token per dim per layer, ~24× checkpointed, ~36× uncheckpointed.
    # These cover Q/K/V projections + attention scores + SwiGLU intermediate + residual buffers.
    factor = 24 if grad_checkpoint else 36
    return n_layers * seq_len * batch_size * hidden_dim * dtype_bytes * factor


def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    hd = getattr(model, "dim", 0)
    if hasattr(model, "embed") and hasattr(model.embed, "embedding_dim"):
        hd = model.embed.embedding_dim  # ponytail: nn.Embedding exposes embedding_dim (was ParallelEmbedding.dim)
    nl = len(model.layers) if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList) else 0
    return hd, nl


def _detect_overhead_gb() -> float:
    if not torch.cuda.is_available():
        return 2.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return min(13.7, max(2.0, total_gb * 0.17))


def estimate_model_memory_gb(model: nn.Module, seq_len: int, batch_size: int, grad_checkpoint: bool = True, overhead_gb: float | None = None, inference: bool = False) -> float:
    # ponytail: training does not grow the kv cache past the current seq_len, so the kv storage is small.
    # Inference loops over max_seq_len tokens; set inference=True to bill the full cache.
    params_b = _parameter_bytes(model)
    optim_b = _optimiser_bytes(model)
    kv_b = _kv_cache_bytes(model, seq_len if not inference else seq_len, batch_size, inference=inference)
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

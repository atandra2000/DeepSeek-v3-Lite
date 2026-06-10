# scripts/microbench_82m.py
"""
Microbench — measure peak VRAM of the 82M model.

Run on a single RTX 4090 24GB (or any CUDA GPU).

Usage:
    python scripts/microbench_82m.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from models.transformer import Transformer
from utils.memory import estimate_model_memory_gb, assert_fits_in_available_gpu


def main() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_82m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    print(f"Building 82M model from {cfg_path} ...")
    print(f"  micro_batch_size = {bs}")
    print(f"  max_seq_len      = {seq}")

    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    print(f"  parameters       = {n_p:,}  ({n_p/1e6:.1f} M)")

    # Estimated peak
    est = estimate_model_memory_gb(m, seq_len=seq, batch_size=bs, grad_checkpoint=True)
    print(f"  estimated peak   = {est:.2f} GB")
    assert_fits_in_available_gpu(est, safety_margin_gb=2.0)

    # Measured peak
    print("Running forward + backward ...")
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, cfg["model"]["vocab_size"], (bs, seq), device="cuda")
    y = m(x)
    y.sum().backward()
    measured = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  measured peak    = {measured:.2f} GB")

    delta = abs(measured - est) / est * 100
    print(f"  delta vs estimate = {delta:.1f}%")

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = measured / total_gb * 100
    print(f"  measured / total = {pct:.1f}% of {total_gb:.0f} GB")

    if measured > total_gb - 4.0:
        print("\n*** WARNING: peak within 4 GB of capacity. Consider:")
        print("  - halve micro_batch_size (8 -> 4)")
        print("  - reduce seq_len (1024 -> 768)")
    elif measured > total_gb * 0.75:
        print("\n*** NOTICE: peak > 75% of VRAM. Comfortable but tight.")
    else:
        print("\n✓ Peak comfortably under GPU capacity.")


if __name__ == "__main__":
    main()

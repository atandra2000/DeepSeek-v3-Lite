# scripts/microbench_800m.py
"""
Microbench — measure peak VRAM of the 757M model.

Run on a single A100 80GB SXM. Run BEFORE the 1k-step dry-run.

Usage:
    python scripts/microbench_800m.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from models.transformer import Transformer
from utils.memory import estimate_model_memory_gb, assert_fits_in_a100_80gb


def main() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_800m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    print(f"Building 757M model from {cfg_path} ...")
    print(f"  micro_batch_size = {bs}")
    print(f"  max_seq_len      = {seq}")

    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    print(f"  parameters       = {n_p:,}  ({n_p/1e6:.1f} M)")

    # Estimated peak
    est = estimate_model_memory_gb(m, seq_len=seq, batch_size=bs, grad_checkpoint=True)
    print(f"  estimated peak   = {est:.2f} GB")
    assert_fits_in_a100_80gb(est)

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

    if measured > 70.0:
        print("\n*** WARNING: peak > 70 GB — only 8 GB headroom. Consider:")
        print("  - halve micro_batch_size (4 -> 2)")
        print("  - reduce seq_len (1024 -> 768)")
        print("  - reduce dim (1024 -> 896, ~600M params)")
    elif measured > 60.0:
        print("\n*** NOTICE: peak > 60 GB. Comfortable but tight.")
    else:
        print("\n✓ Peak comfortably under the 80 GB cap.")


if __name__ == "__main__":
    main()

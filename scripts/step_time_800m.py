# scripts/step_time_800m.py
"""
Step-time microbench — measure ms/step on a single A100.

Validates architecture delivers expected ~40% MFU.

Usage:
    python scripts/step_time_800m.py [--steps 20] [--warmup 5]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from models.transformer import Transformer


A100_BF16_PEAK_TFLOPS = 312.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",  type=int, default=20, help="Number of timed steps")
    p.add_argument("--warmup", type=int, default=5,  help="Number of warmup steps (not timed)")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile for the benchmark")
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_800m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    print(f"Building 757M model: bs={bs}, seq={seq}")

    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    n_nonembed = n_p - 2 * cfg["model"]["vocab_size"] * cfg["model"]["dim"]
    print(f"  non-embed params  = {n_nonembed/1e6:.1f} M")

    opt = torch.optim.AdamW(m.parameters(), lr=8.4e-5, betas=(0.9, 0.95), fused=True)

    if not args.no_compile:
        try:
            m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: FAILED ({e}); continuing without")

    def step() -> None:
        x = torch.randint(0, cfg["model"]["vocab_size"], (bs, seq), device="cuda")
        y = m(x)
        y.sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Warmup
    print(f"Warmup: {args.warmup} steps ...")
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    # Timed
    print(f"Timing: {args.steps} steps ...")
    t0 = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    ms = dt * 1000

    # FLOP accounting: 6 * P_nonembed * seq * bs per step (forward + backward)
    flops = 6 * n_nonembed * seq * bs
    tflops_per_s = flops / dt / 1e12
    mfu = tflops_per_s / A100_BF16_PEAK_TFLOPS * 100

    tok_per_s = bs * seq / dt
    print()
    print(f"Step time:        {ms:.1f} ms")
    print(f"Throughput:       {tok_per_s:,.0f} tok/s")
    print(f"Achieved TFLOPS:  {tflops_per_s:.1f}")
    print(f"MFU (A100 peak):  {mfu:.1f}%")
    print()
    if mfu < 25:
        print("*** MFU < 25% — investigate. Common causes:")
        print("  - MoE Python loop overhead (set use_grouped: 'stacked' in YAML)")
        print("  - SDPA path materialising K (set attn_impl: 'absorption' once Phase B2 lands)")
        print("  - cuBLAS workspace contention (check CUDA_VISIBLE_DEVICES)")
    elif mfu < 35:
        print("MFU in the 25-35% range — workable. Look for any non-fused ops.")
    else:
        print("✓ MFU in the expected 35-45% range for MoE-on-A100 BF16.")


if __name__ == "__main__":
    main()

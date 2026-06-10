# scripts/step_time_82m.py
"""
Step-time microbench — measure ms/step on a single GPU.

Validates architecture delivers expected ~20-30% MFU on consumer GPUs.

Usage:
    python scripts/step_time_82m.py [--steps 20] [--warmup 5]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from models.transformer import Transformer


# RTX 4090 BF16 peak. Adjust for your GPU:
#   RTX 3090:  142
#   RTX 4070:  118
#   A100 SXM:  312
GPU_BF16_PEAK_TFLOPS = 165.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",  type=int, default=20, help="Number of timed steps")
    p.add_argument("--warmup", type=int, default=5,  help="Number of warmup steps (not timed)")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile for the benchmark")
    p.add_argument("--peak-tflops", type=float, default=GPU_BF16_PEAK_TFLOPS,
                   help="GPU BF16 peak TFLOPS for MFU calculation")
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_82m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    print(f"Building 82M model: bs={bs}, seq={seq}")

    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    n_nonembed = n_p - (1 if cfg["model"].get("weight_tying", False) else 2) * cfg["model"]["vocab_size"] * cfg["model"]["dim"]
    print(f"  total params     = {n_p/1e6:.1f} M")
    print(f"  non-embed params = {n_nonembed/1e6:.1f} M")

    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95), fused=True)

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
    mfu = tflops_per_s / args.peak_tflops * 100

    tok_per_s = bs * seq / dt
    print()
    print(f"Step time:        {ms:.1f} ms")
    print(f"Throughput:       {tok_per_s:,.0f} tok/s")
    print(f"Achieved TFLOPS:  {tflops_per_s:.1f}")
    print(f"MFU (GPU peak):   {mfu:.1f}%")
    print()
    if mfu < 15:
        print("*** MFU < 15% — investigate. Common causes:")
        print("  - MoE Python loop overhead (set use_grouped: 'stacked' in YAML)")
        print("  - torch.compile not enabled or failed")
        print("  - cuBLAS workspace contention")
    elif mfu < 25:
        print("MFU in the 15-25% range — workable for consumer GPUs.")
    else:
        print("✓ MFU in the expected 20-30% range for MoE-on-GPU BF16.")


if __name__ == "__main__":
    main()

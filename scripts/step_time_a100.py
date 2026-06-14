"""Step-time microbench -- measure ms/step on A100 80GB SXM. Validates ~30-45% MFU."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, yaml
from models.transformer import Transformer

GPU_BF16_PEAK_TFLOPS = 312.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default="max-autotune")
    p.add_argument("--peak-tflops", type=float, default=GPU_BF16_PEAK_TFLOPS)
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_a100_422m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    print(f"Building 422M model: bs={bs}, seq={seq}")

    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    n_nonembed = n_p - (1 if cfg["model"].get("weight_tying", False) else 2) * cfg["model"]["vocab_size"] * cfg["model"]["dim"]
    print(f"  total params     = {n_p/1e6:.1f} M\n  non-embed params = {n_nonembed/1e6:.1f} M")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95), fused=True)

    if not args.no_compile:
        try:
            m = torch.compile(m, mode=args.compile_mode, fullgraph=False)
            print(f"  torch.compile: enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"  torch.compile: FAILED ({e}); continuing without")
    else:
        print("  torch.compile: disabled")

    def step():
        x = torch.randint(0, cfg["model"]["vocab_size"], (bs, seq), device="cuda")
        y = m(x)
        y.sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    print(f"Warmup: {args.warmup} steps ...")
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    print(f"Timing: {args.steps} steps ...")
    t0 = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    ms = dt * 1000
    flops = 6 * n_nonembed * seq * bs
    tflops_per_s = flops / dt / 1e12
    mfu = tflops_per_s / args.peak_tflops * 100
    tok_per_s = bs * seq / dt
    print(f"\nStep time:        {ms:.1f} ms\nThroughput:       {tok_per_s:,.0f} tok/s\nAchieved TFLOPS:  {tflops_per_s:.1f}\nMFU (GPU peak):   {mfu:.1f}%\n")
    if mfu < 25:
        print("*** MFU < 25% -- investigate. Common: MoE Python loop overhead, torch.compile not enabled, TF32 not set.")
    elif mfu < 35:
        print("MFU in 25-35% range -- workable but room for improvement on A100.")
    else:
        print("MFU in expected 30-45% range for MoE-on-A100 BF16.")


if __name__ == "__main__":
    main()

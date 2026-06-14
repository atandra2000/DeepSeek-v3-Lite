"""Microbench -- measure peak VRAM of the 422M model on A100 80GB."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, yaml
from models.transformer import Transformer
from utils.memory import estimate_model_memory_gb, assert_fits_in_available_gpu


def main() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_a100_422m.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    bs = cfg["training"]["micro_batch_size"]
    seq = cfg["model"]["max_seq_len"]
    print(f"Building 422M model from {cfg_path} ...")
    print(f"  micro_batch_size = {bs}\n  max_seq_len      = {seq}")
    m = Transformer(cfg, use_checkpoint=True).cuda()
    n_p = sum(p.numel() for p in m.parameters())
    print(f"  parameters       = {n_p:,}  ({n_p/1e6:.1f} M)")
    est = estimate_model_memory_gb(m, seq_len=seq, batch_size=bs, grad_checkpoint=True)
    print(f"  estimated peak   = {est:.2f} GB")
    assert_fits_in_available_gpu(est, safety_margin_gb=2.0)
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
    if measured > total_gb - 8.0:
        print("\n*** WARNING: peak within 8 GB of capacity. Consider halving micro_batch_size or seq_len.")
    elif measured > total_gb * 0.7:
        print("\n*** NOTICE: peak > 70% of VRAM. Comfortable.")
    else:
        print("\nPeak comfortably under GPU capacity -- plenty of headroom.")


if __name__ == "__main__":
    main()

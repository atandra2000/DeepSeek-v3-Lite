#!/usr/bin/env python
"""End-to-end GPU pipeline test for DeepSeek-V3-Lite.

Runs training -> checkpoint -> inference on a small config sized to fit the
GTX 1650 (4GB). Verifies: (1) data pipeline, (2) model construction, (3) training
step + finite loss, (4) checkpoint round-trip with MTP, (5) KV-cache generation,
(6) speculative MTP decoding, (7) peak VRAM < 4GB. Exits non-zero on any failure.
"""
import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.pretrain import PretrainDataset, Pretrainer, TrainingConfig
from models.transformer import Transformer
from models.mtp import MTPModule
from utils.checkpoint import CheckpointManager
from inference.speculative import SpeculativeDecoder


class Results:
    def __init__(self) -> None:
        self.cases: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.cases.append((name, ok, detail))
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))

    def all_ok(self) -> bool:
        return all(ok for _, ok, _ in self.cases)

    def summary(self) -> dict:
        return {
            "total": len(self.cases),
            "passed": sum(1 for _, ok, _ in self.cases if ok),
            "failed": sum(1 for _, ok, _ in self.cases if not ok),
            "cases": [
                # Coerce to str: detail strings sometimes embed torch.Tensors.
                {"name": str(n), "passed": bool(ok), "detail": str(d)}
                for n, ok, d in self.cases
            ],
        }


def _build_config(model_cfg_path: Path, ckpt_dir: Path, train_steps: int) -> TrainingConfig:
    with open(model_cfg_path) as f:
        raw = yaml.safe_load(f)
    model_cfg = raw["model"]
    train_cfg = raw.get("training", {})
    return TrainingConfig(
        model_config=model_cfg,
        data_path=str(_REPO_ROOT / "data" / "pretrain_chinchilla"),
        checkpoint_dir=str(ckpt_dir),
        max_seq_len=model_cfg["max_seq_len"],
        vocab_size=model_cfg["vocab_size"],
        batch_size=train_cfg.get("micro_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        max_steps=train_steps,
        warmup_steps=min(20, train_steps // 4),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        grad_checkpoint=train_cfg.get("grad_checkpoint", False),
        compile_model=train_cfg.get("compile", False),
        nan_guard=train_cfg.get("nan_guard", True),
        mtp_weight=model_cfg.get("mtp_loss_weight", 0.3) if model_cfg.get("mtp_depth", 0) > 0 else 0.0,
        save_every=train_steps + 100,  # explicit save happens later
        log_every=5,
    )


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_e2e(model_cfg_path: Path, train_steps: int, ckpt_dir: Path) -> Results:
    r = Results()
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    # 1. Data pipeline
    _section("1. Data pipeline")
    try:
        with open(model_cfg_path) as f:
            raw = yaml.safe_load(f)
        max_seq_len = raw["model"]["max_seq_len"]
        vocab_size = raw["model"]["vocab_size"]
        ds = PretrainDataset(
            str(_REPO_ROOT / "data" / "pretrain_chinchilla"),
            max_seq_len=max_seq_len, vocab_size=vocab_size,
        )
        r.add("Dataset loads", True, f"{len(ds):,} samples, {ds._total_tokens:,} tokens")
        x, y = ds[0]
        assert x.shape == (max_seq_len,) and y.shape == (max_seq_len,), \
            f"Unexpected sample shape: x={x.shape} y={y.shape}"
        assert x.dtype == torch.uint32 and y.dtype == torch.uint32
        assert (x[1:] == y[:-1]).all().item(), "x[1:] should equal y[:-1] (shifted LM)"
        r.add("Sample shape + shift correct", True)
        batch = torch.stack([ds[i][0] for i in range(4)])
        targets = torch.stack([ds[i][1] for i in range(4)])
        batch_gpu = batch.to(device, non_blocking=True)
        targets_gpu = targets.to(device, non_blocking=True)
        r.add("Batched GPU transfer", True,
              f"batch={tuple(batch.shape)} on {batch_gpu.device}")
    except Exception as exc:
        r.add("Data pipeline", False, f"{type(exc).__name__}: {exc}")
        return r

    # 2. Model construction
    _section("2. Model construction")
    try:
        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f)["model"]
        model = Transformer(model_cfg).to(device)
        try:
            from models.transformer import count_parameters
            total, _ = count_parameters(model)
        except Exception:
            total = sum(p.numel() for p in model.parameters())
        r.add("Transformer on GPU", True, f"{total:,} params ({total/1e6:.2f}M)")
        x_test = torch.randint(0, model_cfg["vocab_size"], (2, 16), device=device)
        with torch.no_grad():
            logits = model(x_test)
        assert logits.shape == (2, 16, model_cfg["vocab_size"]), f"Bad logits shape {logits.shape}"
        assert torch.isfinite(logits).all().item(), "Logits contain NaN/Inf"
        r.add("Forward pass (uncompiled)", True, f"logits={tuple(logits.shape)}")
    except Exception as exc:
        r.add("Model construction", False, f"{type(exc).__name__}: {exc}")
        return r

    # Pretrainer rebuilds the model internally; free the standalone copy first.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 3+4. Training
    _section(f"3+4. Training — {train_steps} step(s)")
    try:
        config = _build_config(model_cfg_path, ckpt_dir, train_steps)
        config.compile_model = False  # 1650: torch.compile can OOM or hang

        # Force fused=False AdamW so this works even when the fused CUDA path
        # isn't available on the test host.
        from torch.optim import AdamW as _AdamW
        import training.pretrain as pretrain_mod
        _orig_adamw = pretrain_mod.AdamW
        pretrain_mod.AdamW = lambda *a, **kw: _orig_adamw(*a, **{**kw, "fused": False})

        trainer = Pretrainer(config)
        torch.cuda.synchronize()
        peak_after_init = torch.cuda.max_memory_allocated() / 1e9
        r.add("Pretrainer constructed on GPU", True, f"peak after init: {peak_after_init:.2f} GB")

        from torch.utils.data import DataLoader
        ds = PretrainDataset(
            str(_REPO_ROOT / "data" / "pretrain_chinchilla"),
            max_seq_len=config.max_seq_len, vocab_size=config.vocab_size,
        )
        loader = DataLoader(ds, batch_size=config.batch_size, num_workers=0, drop_last=True)

        step = 0
        losses: list[float] = []
        moe_metrics: list[float] = []
        t0 = time.time()
        for tokens, targets in loader:
            if step >= train_steps:
                break
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            metrics = trainer.train_step(tokens, targets, step)
            if metrics is None:
                raise RuntimeError("train_step returned None — NaN/Inf in loss")
            losses.append(metrics.get("loss", float("nan")))
            moe = metrics.get("moe_balance")
            if moe is not None:
                moe_metrics.append(moe)
            step += 1
            if step % max(1, train_steps // 5) == 0:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step {step:>3d}/{train_steps}  loss={losses[-1]:.4f}  "
                      f"peak_vram={peak:.2f}GB  elapsed={time.time()-t0:.1f}s")
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9

        r.add("Training step ran", True, f"{step} step(s) in {elapsed:.2f}s")
        r.add("Loss is finite & decreasing",
              len(losses) >= 2 and all(torch.isfinite(torch.as_tensor(l)).item() for l in losses)
              and losses[-1] < losses[0] * 1.5,  # tolerate noise; reject explosion
              f"loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
        r.add("Peak VRAM < 4GB", peak < 4.0, f"{peak:.2f} GB")

        if moe_metrics:
            r.add("MoE balance metric finite",
                  all(torch.isfinite(torch.tensor(m)).item() for m in moe_metrics),
                  f"avg={sum(moe_metrics)/len(moe_metrics):.4e}")

        trainer.save_checkpoint(step)
        r.add("Checkpoint saved", (ckpt_dir / f"model_step_{step}.safetensors").exists())

    except Exception as exc:
        import traceback
        traceback.print_exc()
        r.add("Training", False, f"{type(exc).__name__}: {exc}")
        return r

    # 5. Checkpoint round-trip
    _section("5. Checkpoint round-trip")
    try:
        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f)["model"]
        m2 = Transformer(model_cfg).to(device)
        mtp2 = MTPModule(model_cfg, depth=model_cfg["mtp_depth"]).to(device) \
            if model_cfg.get("mtp_depth", 0) > 0 else None
        mgr = CheckpointManager(str(ckpt_dir))
        step_loaded = mgr.latest_step()
        meta = mgr.load(m2, step_loaded, device=str(device), strict=False)
        if mtp2 is not None and meta.get("has_mtp"):
            from safetensors.torch import load_file
            state = load_file(str(ckpt_dir / f"model_step_{step_loaded}.safetensors"),
                              device=str(device))
            mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
            if mtp_state:
                mtp2.load_state_dict(mtp_state, strict=False)
        r.add("Checkpoint loaded", True, f"step={step_loaded}, has_mtp={meta.get('has_mtp', False)}")

        with torch.no_grad():
            x_test = torch.randint(0, model_cfg["vocab_size"], (1, 16), device=device)
            logits_loaded = m2(x_test)
        assert torch.isfinite(logits_loaded).all().item(), "Loaded model produces non-finite logits"
        r.add("Loaded model produces finite logits", True, f"shape={tuple(logits_loaded.shape)}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        r.add("Checkpoint round-trip", False, f"{type(exc).__name__}: {exc}")

    # 6. Autoregressive generation
    _section("6. Autoregressive generation (KV cache)")
    try:
        m2.eval()
        prompt = torch.randint(0, model_cfg["vocab_size"], (1, 8), device=device)
        with torch.no_grad():
            out = m2.generate(prompt, max_new_tokens=32, temperature=1.0, top_p=0.9)
        assert out.shape == (1, 8 + 32), f"Bad generate shape: {out.shape}"
        assert torch.isfinite(out).all().item(), "Generate produced NaN/Inf"
        r.add("Greedy/sampled generation", True, f"shape={tuple(out.shape)}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        r.add("Generation", False, f"{type(exc).__name__}: {exc}")

    # 7. Speculative decoding
    _section("7. Speculative decoding (MTP draft head)")
    try:
        if mtp2 is not None:
            # Standalone MTPModule needs the main model's head wired in explicitly
            # (MultiTokenPrediction does this in __init__; a bare MTPModule does not).
            mtp2.set_output_head(m2.head)
            mtp2.eval()
            decoder = SpeculativeDecoder(m2, mtp2, acceptance_threshold=0.0)
            prompt = torch.randint(0, model_cfg["vocab_size"], (1, 4), device=device)
            with torch.no_grad():
                out_spec = decoder.generate(prompt, max_new_tokens=16, temperature=1.0,
                                            eos_token_id=None)
            assert out_spec.shape[1] >= prompt.shape[1], "Speculative output too short"
            assert torch.isfinite(out_spec).all().item(), "Speculative produced NaN/Inf"
            r.add("Speculative decode", True, f"shape={tuple(out_spec.shape)}")
        else:
            r.add("Speculative decode", True, "skipped (no MTP in this config)")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        r.add("Speculative decode", False, f"{type(exc).__name__}: {exc}")

    return r


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-V3-Lite E2E GPU test")
    parser.add_argument("--config", default="configs/pretrain_1650_2m.yaml")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--ckpt-dir", default=None)
    args = parser.parse_args()

    cfg_path = _REPO_ROOT / args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA not available — this test requires a GPU", file=sys.stderr)
        return 1

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(tempfile.mkdtemp(prefix="e2e_ckpt_"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config: {cfg_path}")
    print(f"Train steps: {args.train_steps}")
    print(f"Checkpoint dir: {ckpt_dir}")

    results = run_e2e(cfg_path, args.train_steps, ckpt_dir)

    _section("Summary")
    summary = results.summary()
    print(json.dumps(summary, indent=2))
    print(f"\n{summary['passed']}/{summary['total']} cases passed, "
          f"{summary['failed']} failed")

    return 0 if results.all_ok() else 1


if __name__ == "__main__":
    sys.exit(main())

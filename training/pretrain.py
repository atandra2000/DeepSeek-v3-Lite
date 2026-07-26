import argparse, math, os, sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import Transformer, count_parameters
from models.mtp import MultiTokenPrediction
from utils.checkpoint import CheckpointManager
from utils.logging import init_logging, get_logger


def make_warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@dataclass
class TrainingConfig:
    model_config: dict = field(default_factory=dict)
    data_path: str = "data/pretrain_data.bin"
    checkpoint_dir: str = "checkpoints/pretrain"
    vocab_size: int = 100018
    max_seq_len: int = 4096
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 20000
    warmup_steps: int = 2000
    lr: float = 2.2e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    mtp_weight: float = 0.0
    bias_update_speed: float = 0.001
    bias_update_every: int = 10
    grad_checkpoint: bool = True
    compile_model: bool = True
    save_every: int = 1000
    log_every: int = 100
    nan_guard: bool = False
    nan_guard_max_consecutive: int = 5
    mup_lr: bool = False
    mup_lr_reference: float = 6.0e-4
    mup_lr_reference_params: int = 757226496
    log_per_component_params: bool = True


class PretrainDataset(Dataset):
    """Packed pre-training dataset backed by flat token tensors (single-file or sharded)."""
    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Pre-training data not found: {data_path}\nRun `python data/prepare_data.py` first.")
        self.layout = "sharded" if os.path.isdir(data_path) else "single"
        if self.layout == "sharded":
            self.shard_paths = [str(p) for p in sorted(Path(data_path).glob("shard_*.bin"))]
            if not self.shard_paths:
                raise FileNotFoundError(f"No `shard_*.bin` files in {data_path}")
            self.shards = [torch.load(p, weights_only=True, map_location="cpu", mmap=True) for p in self.shard_paths]
            self.shard_sizes = [s.numel() for s in self.shards]
            self.shard_offsets = []
            running = 0
            for s in self.shard_sizes:
                self.shard_offsets.append(running)
                running += s
            self._total_tokens = sum(self.shard_sizes)
        else:
            self.data = torch.load(data_path, weights_only=True, map_location="cpu", mmap=True)
            self._total_tokens = self.data.numel()
        self._n_samples = (self._total_tokens - 1) // self.max_seq_len

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        """Map a global token index to (shard_idx, offset_within_shard).

        Out-of-range indices raise IndexError. Negative indices are
        rejected explicitly; indices >= _total_tokens are rejected because
        the last token position is _total_tokens - 1.
        """
        if global_idx < 0 or global_idx >= self._total_tokens:
            raise IndexError(
                f"global_idx {global_idx} out of range [0, {self._total_tokens})"
            )
        import bisect
        lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
        return lo, global_idx - self.shard_offsets[lo]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        needed = self.max_seq_len + 1
        if self.layout == "single":
            chunk = self.data[start: start + needed].clone()
        else:
            pieces = []
            cursor = start
            cursor_pos = 0
            while cursor_pos < needed:
                shard_idx, offset_in_shard = self._locate(cursor)
                shard = self.shards[shard_idx]
                take = min(needed - cursor_pos, self.shard_sizes[shard_idx] - offset_in_shard)
                pieces.append(shard[offset_in_shard: offset_in_shard + take])
                cursor += take
                cursor_pos += take
            chunk = torch.cat(pieces) if len(pieces) > 1 else pieces[0].clone()
        return chunk[:-1], chunk[1:]


class Pretrainer:
    """BF16 pre-training loop for single GPU."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("[warn] CUDA not available — running on CPU (smoke-testing only).")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        init_logging(config.log_every, seq_len=config.max_seq_len)
        self.logger = get_logger()

        self._log("Initialising model...")
        # AGENTS rule #7: default-config run must never silently switch to a Triton path.
        # Guard already ran in Transformer.__init__; this is a no-op.
        raw_model = Transformer(config.model_config, use_checkpoint=config.grad_checkpoint).to(self.device)
        total, trainable = count_parameters(raw_model)
        self._log(f"Parameters: {total:,} total / {trainable:,} trainable")
        if config.log_per_component_params:
            self._log_per_component_params(raw_model)

        self.mtp_wrapper: Optional[MultiTokenPrediction] = None
        mtp_depth = config.model_config.get("model", config.model_config).get("mtp_depth", 0)
        if mtp_depth > 0 and config.mtp_weight > 0.0:
            mtp_model = MultiTokenPrediction(config.model_config, raw_model).to(self.device)
            mtp_total, mtp_trainable = count_parameters(mtp_model)
            self._log(f"MTP enabled (depth={mtp_depth}, weight={config.mtp_weight}): {mtp_total:,} total / {mtp_trainable:,} trainable")
            total = mtp_total
            training_model: nn.Module = mtp_model
            self.mtp_wrapper = mtp_model
        else:
            training_model = raw_model

        if config.compile_model and hasattr(torch, "compile"):
            compile_mode = os.environ.get("TORCH_COMPILE_MODE", "max-autotune")
            self._log(f"Compiling model with torch.compile (mode={compile_mode})...")
            training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)

        self.model = training_model
        self.raw_model: Transformer = raw_model

        if config.mup_lr:
            new_lr = config.mup_lr_reference * (config.mup_lr_reference_params / total) ** 0.5
            self._log(f"µP LR scaling: {config.lr:.2e} → {new_lr:.2e} (ref {config.mup_lr_reference:.2e} @ {config.mup_lr_reference_params:,} params)")
            config.lr = new_lr

        seen = set()
        all_params = []
        for p in self.model.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                all_params.append(p)
        decay_params = [p for p in all_params if p.dim() >= 2]
        no_decay_params = [p for p in all_params if p.dim() < 2]
        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.lr, betas=(config.beta1, config.beta2), fused=True)

        lr_lambda = make_warmup_cosine_lambda(warmup_steps=config.warmup_steps, total_steps=config.max_steps, min_lr_ratio=config.min_lr_ratio)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.amp_dtype = torch.bfloat16
        self.ckpt_manager = CheckpointManager(config.checkpoint_dir)
        self._opt_steps: int = 0

    @staticmethod
    def _log(msg: str) -> None:
        print(msg)

    def _amp_context(self):
        return autocast("cuda", dtype=self.amp_dtype)

    def _update_moe_bias(self) -> None:
        for moe in self.raw_model.moe_layers():
            moe.update_gate_bias(speed=self.config.bias_update_speed)

    def _moe_balance_metric(self) -> torch.Tensor:
        """Return the on-device balance loss tensor — .item() is deferred to the logger path (avoid per-step sync)."""
        losses = [moe.get_load_balance_loss() for moe in self.raw_model.moe_layers()]
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).sum()

    def _log_per_component_params(self, model) -> None:
        from collections import defaultdict
        comps: defaultdict[str, int] = defaultdict(int)
        for name, p in model.named_parameters():
            if "embed" in name:
                comps["embedding"] += p.numel()
            elif "head" in name:
                comps["lm_head"] += p.numel()
            elif ".attn." in name and ("wq" in name or "wkv_a" in name or "wkv_b" in name or "wo" in name or "q_norm" in name or "kv_norm" in name):
                comps["mla_attn"] += p.numel()
            elif "attn_norm" in name or "ffn_norm" in name or name.endswith(".norm.weight"):
                comps["rmsnorm"] += p.numel()
            elif ".experts." in name and ("w1" in name or "w2" in name or "w3" in name):
                comps["moe_routed_experts"] += p.numel()
            elif "shared_experts" in name:
                comps["moe_shared_experts"] += p.numel()
            elif ".ffn.w" in name:
                comps["dense_swiglu"] += p.numel()
            elif ".gate." in name:
                comps["moe_gate"] += p.numel()
            else:
                comps["other"] += p.numel()
        total = sum(comps.values())
        self._log("  Per-component parameter breakdown:")
        for name_, n in sorted(comps.items(), key=lambda x: -x[1]):
            self._log(f"    {name_:25s}: {n:>12,}  ({n / total * 100 if total else 0.0:5.2f}%)")
        self._log(f"    {'TOTAL':25s}: {total:>12,}  ({total / 1e6:.2f} M)")

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, micro_step: int) -> Optional[Dict[str, Optional[float]]]:
        is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0
        # Cast uint32 → int64 at the boundary. PretrainDataset stores tokens as
        # uint32 for memory efficiency (4 bytes vs 8), but `nn.Embedding` and
        # `F.cross_entropy` both require Long indices. Doing the cast here keeps
        # the dataset's storage compact and the training path dtype-correct.
        if tokens.dtype != torch.long:
            tokens = tokens.to(torch.long)
        if targets.dtype != torch.long:
            targets = targets.to(torch.long)
        with self._amp_context():
            if self.mtp_wrapper is not None:
                main_logits, mtp_pairs = self.model(tokens, start_pos=0)
                total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)
                # ponytail: .detach() instead of .item() — logger does the host round-trip at log_every cadence.
                _ce_loss_val = main_loss.detach()
                _mtp_loss_val = mtp_loss.detach() if mtp_pairs else None
                loss = total_loss / self.config.gradient_accumulation_steps
            else:
                logits = self.model(tokens, start_pos=0, use_cache=False)
                main_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
                _ce_loss_val = main_loss.detach()
                _mtp_loss_val = None
                loss = main_loss / self.config.gradient_accumulation_steps
            balance_loss = self._moe_balance_metric()

        if self.config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
            self._log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={self._opt_steps}. Skipping backward.")
            self.optimizer.zero_grad(set_to_none=True)
            return None

        loss.backward()
        if is_opt_step:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._opt_steps += 1
            if self._opt_steps % self.config.bias_update_every == 0:
                self._update_moe_bias()

        return {"loss": _ce_loss_val, "mtp_loss": _mtp_loss_val, "balance_loss": balance_loss}

    def save_checkpoint(self, step: int, tag: str = "") -> None:
        model_to_save = self.raw_model
        state = model_to_save.state_dict()
        if self.mtp_wrapper is not None:
            mtp_mod = self.mtp_wrapper
            orig = getattr(mtp_mod, "_orig_mod", mtp_mod)
            mtp_state = {f"mtp.{k}": v for k, v in orig.state_dict().items() if k.startswith("mtp_modules.")}
            state.update(mtp_state)
        extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
                      "tag": tag or f"step_{step}", "config": asdict(self.config), "has_mtp": self.mtp_wrapper is not None}
        self.ckpt_manager.save(model_to_save, self.optimizer, step, extra_meta=extra_meta, state_dict=state)
        self._log(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, step: int) -> int:
        from safetensors.torch import load_file
        meta = self.ckpt_manager.load(self.raw_model, step, device=str(self.device), optimizer=self.optimizer, strict=False)
        if self.mtp_wrapper is not None and meta.get("has_mtp", False):
            weight_path = self.ckpt_manager.save_dir / f"model_step_{step}.safetensors"
            if weight_path.exists():
                state = load_file(str(weight_path), device=str(self.device))
                mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
                if mtp_state:
                    mtp_orig = getattr(self.mtp_wrapper, "_orig_mod", self.mtp_wrapper)
                    mtp_orig.load_state_dict(mtp_state, strict=False)
                    self._log(f"MTP weights restored ({len(mtp_state)} keys)")
        if "scheduler" in meta:
            self.scheduler.load_state_dict(meta["scheduler"])
        if "opt_steps" in meta:
            self._opt_steps = meta["opt_steps"]
        resumed_step = meta.get("step", step)
        self._log(f"Resumed from step {resumed_step}")
        return resumed_step

    def _find_latest_checkpoint(self) -> Optional[int]:
        return self.ckpt_manager.latest_step()

    def train(self) -> None:
        dataset = PretrainDataset(self.config.data_path, self.config.max_seq_len, self.config.vocab_size)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=8, pin_memory=True,
                            persistent_workers=True, prefetch_factor=8, drop_last=True)

        global_step = 0
        latest = self._find_latest_checkpoint()
        if latest is not None:
            try:
                global_step = self.load_checkpoint(latest)
            except Exception as exc:
                self._log(f"[warn] Could not load checkpoint: {exc}")

        self._log(f"Training from step {global_step} to {self.config.max_steps}")
        self.raw_model.train()
        epoch = 0
        nan_guard_streak = 0
        while global_step < self.config.max_steps:
            for tokens, targets in tqdm(loader):
                if global_step >= self.config.max_steps:
                    break
                tokens = tokens.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                metrics = self.train_step(tokens, targets, global_step)
                if metrics is None:
                    nan_guard_streak += 1
                    if nan_guard_streak >= self.config.nan_guard_max_consecutive:
                        latest = self._find_latest_checkpoint()
                        if latest is not None:
                            self._log(f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf — restoring checkpoint step {latest}.")
                            global_step = self.load_checkpoint(latest)
                        else:
                            self._log("[nan-guard] No checkpoint to restore from. Aborting.")
                            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
                        nan_guard_streak = 0
                    continue
                nan_guard_streak = 0
                if global_step % self.config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    # ponytail: single .item() per log step (not per micro-step) — avoids 3-4 forced GPU syncs per step.
                    log_metrics = {"balance_loss": float(metrics["balance_loss"].item()) if isinstance(metrics["balance_loss"], torch.Tensor) else float(metrics["balance_loss"])}
                    if metrics.get("mtp_loss") is not None:
                        log_metrics["mtp_loss"] = float(metrics["mtp_loss"].item()) if isinstance(metrics["mtp_loss"], torch.Tensor) else float(metrics["mtp_loss"])
                    ce = metrics["loss"]
                    self.logger.log(global_step, float(ce.item()) if isinstance(ce, torch.Tensor) else float(ce), lr=lr, metrics=log_metrics)
                if global_step % self.config.save_every == 0 and global_step > 0:
                    self.save_checkpoint(global_step)
                global_step += 1
        self.save_checkpoint(global_step, tag="final")
        self._log("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V3-Lite pre-training (single GPU)")
    parser.add_argument("--config", type=str, default="configs/pretrain_a100_422m.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint step number to resume from")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    t = yaml_cfg.get("training", {})
    d = yaml_cfg.get("data", {})

    config = TrainingConfig(
        model_config=yaml_cfg,
        data_path=args.data_path or d.get("train_data_path", "data/pretrain_data.bin"),
        checkpoint_dir=args.checkpoint_dir or t.get("save_dir", "checkpoints/pretrain"),
        max_seq_len=yaml_cfg.get("model", yaml_cfg).get("max_seq_len", 4096),
        vocab_size=yaml_cfg.get("model", yaml_cfg).get("vocab_size", 100018),
        batch_size=t.get("micro_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        max_steps=t.get("total_steps", 20000),
        warmup_steps=t.get("warmup_steps", 2000),
        lr=t.get("lr", 2.2e-4),
        min_lr_ratio=t.get("min_lr_ratio", 0.1),
        weight_decay=t.get("weight_decay", 0.1),
        max_grad_norm=t.get("grad_clip", 1.0),
        grad_checkpoint=t.get("grad_checkpoint", True) and not args.no_checkpoint,
        compile_model=t.get("compile", True) and not args.no_compile,
        mtp_weight=t.get("mtp_loss_weight", 0.0),
        bias_update_speed=t.get("bias_update_speed", 0.001),
        bias_update_every=t.get("bias_update_every", 10),
        save_every=t.get("save_interval", 1000),
        log_every=t.get("log_interval", 100),
        nan_guard=t.get("nan_guard", False),
        nan_guard_max_consecutive=t.get("nan_guard_max_consecutive", 5),
        mup_lr=t.get("mup_lr", False),
        mup_lr_reference=t.get("mup_lr_reference", 6.0e-4),
        mup_lr_reference_params=t.get("mup_lr_reference_params", 757226496),
        log_per_component_params=t.get("log_per_component_params", True),
    )

    trainer = Pretrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(int(args.resume))
    trainer.train()


if __name__ == "__main__":
    main()

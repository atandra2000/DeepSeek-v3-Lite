# training/pretrain.py
import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import Transformer, count_parameters
from utils.checkpoint import CheckpointManager
from utils.distributed import device
from utils.logging import init_logging, get_logger
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb


# ── Scheduler ─────────────────────────────────────────────────────────────────

def make_warmup_cosine_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> callable:
    """Build LambdaLR schedule: linear warmup → cosine decay → flat min."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All training hyperparameters. `model_config` holds the full parsed YAML dict."""

    model_config: dict = field(default_factory=dict)

    # Paths
    data_path:       str = "data/pretrain_data.bin"
    checkpoint_dir:  str = "checkpoints/pretrain"

    # Data
    vocab_size:  int = 102400
    max_seq_len: int = 4096

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 20_000
    warmup_steps: int = 2_000

    # Optimisation
    lr: float = 2.2e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # MoE load balancing (aux-loss-free gate handles balancing; kept for logging).
    balance_loss_alpha: float = 0.0
    bias_update_speed: float = 0.001
    bias_update_every: int = 10

    # Memory / compile
    grad_checkpoint: bool = True
    compile_model: bool = True

    # Logging / checkpointing
    save_every: int = 1_000
    log_every:  int = 100

    nan_guard: bool = False
    nan_guard_max_consecutive: int = 5
    mup_lr: bool = False
    mup_lr_reference: float = 6.0e-4
    mup_lr_reference_params: int = 757_226_496
    log_per_component_params: bool = True


# ── Dataset ────────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Packed pre-training dataset backed by flat token tensors.

    Two storage layouts are supported (auto-detected):

      Single-file: a single ``data_path`` pointing to a
        ``.bin`` produced by ``data/prepare_data.py`` with the default
        settings. The whole tensor is loaded into RAM once at construction.

      Sharded: a ``data_path`` that is a directory
        containing ``shard_NNNNN.bin`` files. The dataset builds an
        in-memory ``ShardIndex`` (offsets only) and lazily memory-maps
        the requested shard on each ``__getitem__`` call. RAM usage is
        bounded by the largest single shard (~1 GB).

    Each sample is ``(input, target)`` where ``input = tokens[i:i+L]`` and
    ``target = tokens[i+1:i+L+1]`` — the standard LM shift.
    """

    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Pre-training data not found: {data_path}\n"
                f"Run `python data/prepare_data.py --stage pretrain --tokenizer <name>` first."
            )

        if os.path.isdir(data_path):
            self._init_sharded(data_path)
        else:
            self._init_single(data_path)

    # ── Single-file layout ─────────────────────────────────────────────────────

    def _init_single(self, data_path: str) -> None:
        self.layout = "single"
        self.data = torch.load(data_path, weights_only=True)
        # Serves (len(data) - 1) // L samples; final sample truncated.
        self._n_samples = (len(self.data) - 1) // self.max_seq_len

    def _get_window_single(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        chunk = self.data[start : start + self.max_seq_len + 1]
        return chunk[:-1], chunk[1:]

    # ── Sharded layout ─────────────────────────────────────────────────────────

    def _init_sharded(self, data_dir: str) -> None:
        shard_paths = sorted(Path(data_dir).glob("shard_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No `shard_*.bin` files found in directory: {data_dir}\n"
                f"Re-run `data/prepare_data.py --stage pretrain "
                f"--shard-size-tokens 1000000000 ...`."
            )
        self.layout = "sharded"
        self.shard_paths = [str(p) for p in shard_paths]
        # Build an offset table: shard_offsets[i] = global token index
        # at which shard i begins. We read each shard's length once at
        # startup; reading the actual tensors is lazy (memory-mapped).
        self.shard_sizes: list[int] = []
        self.shard_offsets: list[int] = []
        running = 0
        for p in self.shard_paths:
            # We only need the .numel() — load with weights_only and
            # immediately drop the reference to keep RAM bounded.
            t = torch.load(p, weights_only=True, map_location="cpu")
            n = t.numel()
            del t
            self.shard_sizes.append(n)
            self.shard_offsets.append(running)
            running += n
        self._total_tokens = running
        self._n_samples = (self._total_tokens - 1) // self.max_seq_len
        # Per-shard tensor cache: most workloads are sequential, so the
        # last-touched shard is the next one we want. Simple FIFO eviction
        # at size 2 keeps memory bounded and lets a 1 GB shard live
        # alongside the ~16 MB being indexed.
        self._shard_cache: dict[int, torch.Tensor] = {}
        self._shard_cache_order: list[int] = []

    def _get_window_sharded(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        end   = start + self.max_seq_len + 1
        # Simple path: window does not cross a shard boundary.
        shard_idx, offset_in_shard = self._locate(start)
        if offset_in_shard + (self.max_seq_len + 1) <= self.shard_sizes[shard_idx]:
            shard = self._load_shard(shard_idx)
            chunk = shard[offset_in_shard : offset_in_shard + self.max_seq_len + 1]
            return chunk[:-1], chunk[1:]
        # Cross-shard path: concatenate from up to two shards. This is
        # rare (only at shard boundaries) and we accept the copy cost.
        return self._get_window_cross_shard(start)

    def _get_window_cross_shard(
        self, start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.max_seq_len + 1
        collected: list[int] = []
        cursor = start
        while len(collected) < needed:
            shard_idx, offset_in_shard = self._locate(cursor)
            shard = self._load_shard(shard_idx)
            take = min(
                needed - len(collected),
                self.shard_sizes[shard_idx] - offset_in_shard,
            )
            collected.extend(shard[offset_in_shard : offset_in_shard + take].tolist())
            cursor += take
        chunk = torch.tensor(collected[:needed], dtype=torch.long)
        return chunk[:-1], chunk[1:]

    def _locate(self, global_idx: int) -> tuple[int, int]:
        """Locate the shard containing `global_idx` via binary search.

        `global_idx` must be strictly less than `_total_tokens`. The
        last (max_seq_len+1)-token window in the corpus can extend right
        up to the final token but not past it — this is enforced by
        the n_samples computation in `__len__` and `__init__`.
        """
        if global_idx < 0 or global_idx >= self._total_tokens:
            raise IndexError(
                f"global token index {global_idx} out of range "
                f"(total={self._total_tokens})"
            )
        # Binary search over shard_offsets. O(log n_shards) — fine for
        # any realistic shard count, and clearer than the linear scan it
        # replaces.
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, global_idx - self.shard_offsets[lo]

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]
        t = torch.load(self.shard_paths[shard_idx], weights_only=True,
                       map_location="cpu")
        self._shard_cache[shard_idx] = t
        self._shard_cache_order.append(shard_idx)
        # Evict to size 2 (current + previous shard).
        while len(self._shard_cache_order) > 2:
            evict = self._shard_cache_order.pop(0)
            self._shard_cache.pop(evict, None)
        return t

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layout == "single":
            return self._get_window_single(idx)
        return self._get_window_sharded(idx)


# ── Trainer ────────────────────────────────────────────────────────────────────

class Pretrainer:
    """BF16 pre-training loop for single GPU."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = device()

        if not torch.cuda.is_available():
            print("[warn] CUDA not available — running on CPU "
                  "(smoke-testing only; training requires a CUDA GPU).")

        init_logging(config.log_every, seq_len=config.max_seq_len)
        self.logger = get_logger()

        # ── Model ──────────────────────────────────────────────────────────
        self._log("Initialising model...")
        raw_model = Transformer(
            config.model_config,
            use_checkpoint=config.grad_checkpoint,
        ).to(self.device)

        total, trainable = count_parameters(raw_model)
        self._log(f"Parameters: {total:,} total / {trainable:,} trainable")

        if config.log_per_component_params:
            self._log_per_component_params(raw_model)

        # ── µP LR scaling ─────────────────────────────────────────────────────
        # If enabled, scale LR by lr_ref · (P_ref / P)^0.5.
        if config.mup_lr:
            new_lr = config.mup_lr_reference * (
                config.mup_lr_reference_params / total
            ) ** 0.5
            self._log(
                f"µP LR scaling: {config.lr:.2e} → {new_lr:.2e} "
                f"(ref {config.mup_lr_reference:.2e} @ {config.mup_lr_reference_params:,} params)"
            )
            config.lr = new_lr

        if config.compile_model and hasattr(torch, "compile"):
            self._log("Compiling model with torch.compile...")
            raw_model = torch.compile(raw_model, mode="reduce-overhead", fullgraph=False)

        self.model = raw_model
        self.raw_model: Transformer = raw_model

        # ── Optimiser ──────────────────────────────────────────────────────
        # Deduplicate parameters by tensor id (needed for weight tying where
        # head.weight shares storage with embed.weight).
        seen = set()
        all_params = []
        for p in self.raw_model.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                all_params.append(p)
        decay_params    = [p for p in all_params if p.dim() >= 2]
        no_decay_params = [p for p in all_params if p.dim() < 2]
        self.optimizer = AdamW(
            [
                {"params": decay_params,    "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            fused=True,
        )

        # ── Scheduler ──────────────────────────────────────────────────────
        lr_lambda = make_warmup_cosine_lambda(
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # ── AMP ────────────────────────────────────────────────────────────
        self.amp_dtype = torch.bfloat16

        # ── Checkpoint manager ─────────────────────────────────────────────
        self.ckpt_manager = CheckpointManager(config.checkpoint_dir)

        # ── Optimiser step counter ─────────────────────────────────────────
        self._opt_steps: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _log(msg: str) -> None:
        print(msg)

    def _amp_context(self):
        """Return the BF16 autocast context."""
        return autocast("cuda", dtype=self.amp_dtype)

    def _update_moe_bias(self) -> None:
        """
        Update per-expert load-balancing biases using routing cached during the most recent forward pass.
        """
        for moe in self.raw_model.moe_layers():
            moe.update_gate_bias(speed=self.config.bias_update_speed)

    def _moe_balance_metric(self) -> float:
        """Sum of MoE load-balance auxiliary losses (for logging)."""
        losses = [
            moe.get_load_balance_loss()
            for moe in self.raw_model.moe_layers()
        ]
        if not losses:
            return 0.0
        return float(torch.stack(losses).sum().item())

    def _log_per_component_params(self, model) -> None:
        """
        Print per-component parameter breakdown at startup.
        Buckets: embedding, lm_head, mla_attn, dense_swiglu, moe_routed/shared,
        moe_gate, rmsnorm, other. Sorted by size descending.
        """
        from collections import defaultdict
        comps: defaultdict[str, int] = defaultdict(int)
        for name, p in model.named_parameters():
            if "embed" in name:
                comps["embedding"] += p.numel()
            elif "head" in name:
                comps["lm_head"]   += p.numel()
            elif ".attn." in name and (
                "wq" in name or "wkv_a" in name or "wkv_b" in name
                or "wo" in name or "q_norm" in name or "kv_norm" in name
            ):
                comps["mla_attn"]  += p.numel()
            elif "attn_norm" in name or "ffn_norm" in name or name.endswith(".norm.weight"):
                comps["rmsnorm"]   += p.numel()
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
            pct = n / total * 100 if total else 0.0
            self._log(f"    {name_:25s}: {n:>12,}  ({pct:5.2f}%)")
        self._log(f"    {'TOTAL':25s}: {total:>12,}  ({total/1e6:.2f} M)")

    # ──────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        micro_step: int,
    ) -> Optional[Dict[str, float]]:
        """
        Run one micro-step (forward + backward). Optimiser steps every
        `gradient_accumulation_steps` micro-steps.

        Returns metrics dict, or None if NaN/Inf guard fired.
        """
        is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0

        with self._amp_context():
            # use_cache=False: teacher-forced training must not use the KV cache.
            logits = self.model(tokens, start_pos=0, use_cache=False)

            ce_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100)

            balance_loss = self._moe_balance_metric()
            # balance_loss is intentionally NOT added to the optimisation loss —
            # aux-loss-free bias updates already balance the routing.
            loss = ce_loss / self.config.gradient_accumulation_steps

        # NaN/Inf guard: a single bad expert can push loss to inf.
        # Catching it costs one .item() per micro-step.
        if self.config.nan_guard and (
            torch.isnan(loss).any().item() or torch.isinf(loss).any().item()
        ):
            self._log(
                f"[nan-guard] NaN/Inf detected at micro_step={micro_step}, "
                f"opt_steps={self._opt_steps}. Skipping backward."
            )
            # Discard the in-flight computation so the next forward is clean.
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

        return {
            "loss":         float(ce_loss.item()),
            "balance_loss": balance_loss,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, step: int, tag: str = "") -> None:
        """Save model, optimiser, and metadata to checkpoint."""
        extra_meta = {
            "scheduler": self.scheduler.state_dict(),
            "opt_steps": self._opt_steps,
            "tag":       tag or f"step_{step}",
            "config":    asdict(self.config),
        }
        # Unwrap torch.compile if necessary
        model_to_save = self.raw_model
        self.ckpt_manager.save(model_to_save, self.optimizer, step, extra_meta=extra_meta)
        self._log(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, step: int) -> int:
        """Load model, optimiser, and metadata from checkpoint."""
        meta = self.ckpt_manager.load(
            self.raw_model,
            step,
            device=str(self.device),
            optimizer=self.optimizer,
        )
        if "scheduler" in meta:
            self.scheduler.load_state_dict(meta["scheduler"])
        if "opt_steps" in meta:
            self._opt_steps = meta["opt_steps"]
        resumed_step = meta.get("step", step)
        self._log(f"Resumed from step {resumed_step}")
        return resumed_step

    def _find_latest_checkpoint(self) -> Optional[int]:
        return self.ckpt_manager.latest_step()

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        dataset = PretrainDataset(
            self.config.data_path,
            self.config.max_seq_len,
            self.config.vocab_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,
        )

        # Estimate VRAM and abort if it doesn't fit on the available GPU
        estimate = estimate_model_memory_gb(
            self.raw_model,
            seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
        )
        assert_fits_in_available_gpu(estimate)
        self._log(f"Estimated peak VRAM: {estimate:.1f} GB")

        # Resume from latest checkpoint if available
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
        # NaN/Inf recovery: restore from checkpoint after consecutive detections.
        nan_guard_streak = 0
        while global_step < self.config.max_steps:
            for tokens, targets in tqdm(loader):
                if global_step >= self.config.max_steps:
                    break

                tokens  = tokens.to(self.device,  non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                metrics = self.train_step(tokens, targets, global_step)

                # Handle NaN/Inf recovery
                if metrics is None:
                    nan_guard_streak += 1
                    if nan_guard_streak >= self.config.nan_guard_max_consecutive:
                        latest = self._find_latest_checkpoint()
                        if latest is not None:
                            self._log(
                                f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf "
                                f"steps — restoring from checkpoint step {latest}."
                            )
                            global_step = self.load_checkpoint(latest)
                        else:
                            self._log(
                                "[nan-guard] No checkpoint to restore from. "
                                "Aborting."
                            )
                            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
                        nan_guard_streak = 0
                    # Re-fetch the next batch and continue without advancing
                    # global_step (so the schedule stays correct).
                    continue
                nan_guard_streak = 0

                if global_step % self.config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.log(
                        global_step,
                        metrics["loss"],
                        lr=lr,
                        metrics={"balance_loss": metrics["balance_loss"]},
                    )

                if (
                    global_step % self.config.save_every == 0
                    and global_step > 0
                ):
                    self.save_checkpoint(global_step)

                global_step += 1

        self.save_checkpoint(global_step, tag="final")
        self._log("Training complete.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V3-Lite pre-training (single GPU)")
    parser.add_argument("--config",         type=str, default="configs/pretrain_82m.yaml")
    parser.add_argument("--data-path",      type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume",         type=str, default=None,
                        help="Checkpoint step number to resume from")
    parser.add_argument("--no-checkpoint",  action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no-compile",     action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    t = yaml_cfg.get("training", {})
    d = yaml_cfg.get("data",     {})

    config = TrainingConfig(
        model_config=yaml_cfg,

        data_path=args.data_path or d.get("train_data_path", "data/pretrain_data.bin"),
        checkpoint_dir=args.checkpoint_dir or t.get("save_dir", "checkpoints/pretrain"),

        # Read architecture params from model config (not training defaults).
        max_seq_len=yaml_cfg.get("model", yaml_cfg).get("max_seq_len", 4096),
        vocab_size=yaml_cfg.get("model", yaml_cfg).get("vocab_size", 102400),

        batch_size=t.get("micro_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        max_steps=t.get("total_steps", 20_000),
        warmup_steps=t.get("warmup_steps", 2_000),

        lr=t.get("lr", 2.2e-4),
        min_lr_ratio=t.get("min_lr_ratio", 0.1),
        weight_decay=t.get("weight_decay", 0.1),
        max_grad_norm=t.get("grad_clip", 1.0),

        grad_checkpoint=t.get("grad_checkpoint", True) and not args.no_checkpoint,
        compile_model=t.get("compile", True) and not args.no_compile,

        balance_loss_alpha=t.get("balance_loss_alpha", 0.0),
        bias_update_speed=t.get("bias_update_speed", 0.001),
        bias_update_every=t.get("bias_update_every", 10),

        save_every=t.get("save_interval", 1_000),
        log_every=t.get("log_interval", 100),

        # Passed through from the YAML if present
        nan_guard=t.get("nan_guard", False),
        nan_guard_max_consecutive=t.get("nan_guard_max_consecutive", 5),
        mup_lr=t.get("mup_lr", False),
        mup_lr_reference=t.get("mup_lr_reference", 6.0e-4),
        mup_lr_reference_params=t.get("mup_lr_reference_params", 757_226_496),
        log_per_component_params=t.get("log_per_component_params", True),
    )

    trainer = Pretrainer(config)

    if args.resume is not None:
        trainer.load_checkpoint(int(args.resume))

    trainer.train()


if __name__ == "__main__":
    main()

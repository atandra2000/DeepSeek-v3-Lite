# training/sft.py
import sys
from pathlib import Path
import json
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
import math
import yaml

sys.path.append(str(Path(__file__).parent.parent))


def make_cosine_lambda(total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return min_lr_ratio
        progress = min(1.0, step / total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


# ── Dataset ────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset with sample-isolation loss masking.
    Masks all tokens before the last `<|assistant|>` or `[/INST]` marker.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 8192) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        messages = item["messages"]

        # Build the full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Build the prompt-only template to find the assistant offset
        prompt_text = self.tokenizer.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)

        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]
        prompt_len = min(len(prompt_ids), len(full_ids))

        tokens = torch.tensor(full_ids, dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]

        # Sample-isolation: loss on assistant tokens only.
        # Positions [prompt_len-1 .. end-1] in `y` correspond to assistant tokens in `x`.
        loss_mask = torch.zeros_like(y, dtype=torch.float)
        loss_mask[max(0, prompt_len - 1):] = 1.0

        return x, y, loss_mask


def sft_collate_fn(batch):
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    xs_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    ys_pad = torch.full((len(ys), max_len), -100, dtype=torch.long)
    masks_pad = torch.zeros(len(masks), max_len, dtype=torch.float)

    for i, (x, y, m) in enumerate(zip(xs, ys, masks)):
        n = x.size(0)
        xs_pad[i, :n] = x
        ys_pad[i, :n] = y
        masks_pad[i, :n] = m

    return xs_pad, ys_pad, masks_pad


# ── Trainer ────────────────────────────────────────────────────────────────────

class SFTTrainer:
    """Supervised Fine-Tuning trainer with BF16 AMP."""

    def __init__(self, model: nn.Module, config: dict) -> None:
        self.model = model
        self.config = config

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("lr", 5e-6),
            weight_decay=0.1,
        )

        max_steps = config.get("max_steps", 1000)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=make_cosine_lambda(max_steps),
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        n_batches = 0

        for x, y, loss_mask in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x)
                per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                loss = (per_token * loss_mask.reshape(-1)).sum() / loss_mask.sum().clamp(min=1e-10)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True,
                        help="YAML model config path (e.g. configs/pretrain_800m.yaml)")
    parser.add_argument("--model-path",  type=str, default=None,
                        help="Checkpoint directory to resume from (optional)")
    parser.add_argument("--data-path",   type=str, default="data/sft_data.json")
    parser.add_argument("--output-path", type=str, default="checkpoints/sft")
    parser.add_argument("--tokenizer",   type=str, default="deepseek-ai/deepseek-coder-v2-lite")
    parser.add_argument("--epochs",      type=int, default=2)
    parser.add_argument("--batch-size",  type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--lr",          type=float, default=5e-6)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(
            f"SFT data not found: {args.data_path}\n"
            f"Run `python data/prepare_data.py --stage sft` first."
        )

    from models.transformer import Transformer
    from utils.checkpoint import CheckpointManager
    from transformers import AutoTokenizer

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    print("Initialising model from config...")
    model = Transformer(yaml_cfg).cuda()

    if args.model_path is not None:
        ckpt_mgr = CheckpointManager(args.model_path)
        step = ckpt_mgr.latest_step()
        if step is not None:
            print(f"Loading checkpoint step {step} from {args.model_path}")
            ckpt_mgr.load(model, step, device="cuda", strict=False)
        else:
            print(f"[warn] No checkpoints found in {args.model_path}; starting from scratch.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = SFTDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    config = {
        "lr": args.lr,
        "max_steps": args.epochs * len(dataloader),
        "output_path": args.output_path,
    }
    trainer = SFTTrainer(model, config)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} — Loss: {loss:.4f}")
        ckpt = Path(args.output_path) / f"sft_epoch_{epoch+1}.pt"
        trainer.save_checkpoint(str(ckpt))

    print("SFT complete.")


if __name__ == "__main__":
    main()

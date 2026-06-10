# training/distillation.py
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import yaml

sys.path.append(str(Path(__file__).parent.parent))


def make_cosine_lambda(total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return min_lr_ratio
        progress = min(1.0, step / total_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ── Dataset ────────────────────────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """
    Distillation dataset: tokenises (prompt, teacher_response) pairs.
    Labels mask the prompt so loss is on teacher response only.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        prompt = ex.get("prompt", "")
        teacher = ex.get("teacher_response", "")
        if not prompt or not teacher:
            # Return an empty sample that the collate function will drop.
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        teacher_ids = self.tokenizer.encode(teacher, add_special_tokens=False)

        full = prompt_ids + teacher_ids
        if len(full) > self.max_seq_len:
            full = full[: self.max_seq_len]

        ids = torch.tensor(full, dtype=torch.long)
        labels = ids.clone()
        labels[: len(prompt_ids)] = -100  # mask prompt tokens
        return ids, labels


def distill_collate_fn(batch):
    batch = [(i, l) for i, l in batch if i.numel() > 0]
    if not batch:
        return None
    ids_list, labels_list = zip(*batch)
    max_len = max(i.size(0) for i in ids_list)
    bsz = len(ids_list)
    ids_pad = torch.zeros(bsz, max_len, dtype=torch.long)
    labels_pad = torch.full((bsz, max_len), -100, dtype=torch.long)
    for k, (i, l) in enumerate(zip(ids_list, labels_list)):
        n = i.size(0)
        ids_pad[k, :n] = i
        labels_pad[k, :n] = l
    return ids_pad, labels_pad


# ── Trainer ────────────────────────────────────────────────────────────────────

class ReasoningDistillation:
    """
    Knowledge Distillation from a reasoning teacher. Single-GPU, BF16.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: dict,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.get("lr", 1e-5),
            weight_decay=0.01,
        )
        max_steps = config.get("max_steps", 1000)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=make_cosine_lambda(max_steps),
        )
        self.temperature = config.get("temperature", 2.0)
        self.alpha = config.get("distill_alpha", 0.7)

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between softened distributions."""
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if batch is None:
            return {"total_loss": 0.0, "distill_loss": 0.0, "task_loss": 0.0}
        input_ids, labels = batch
        input_ids = input_ids.cuda()
        labels = labels.cuda()

        self.student.train()
        with autocast("cuda", dtype=torch.bfloat16):
            student_logits = self.student(input_ids)
            with torch.no_grad():
                teacher_logits = self.teacher(input_ids)

            s_flat = student_logits.reshape(-1, student_logits.size(-1))
            t_flat = teacher_logits.reshape(-1, teacher_logits.size(-1))

            distill_loss = self.compute_distillation_loss(s_flat, t_flat)
            task_loss = F.cross_entropy(s_flat, labels.reshape(-1), ignore_index=-100)
            total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total_loss":   float(total_loss.item()),
            "distill_loss": float(distill_loss.item()),
            "task_loss":    float(task_loss.item()),
        }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, required=True,
                        help="YAML model config path (shared architecture for student and teacher)")
    parser.add_argument("--student-path", type=str, required=True,
                        help="Checkpoint directory for the student model")
    parser.add_argument("--teacher-path", type=str, required=True,
                        help="Checkpoint directory for the teacher model")
    parser.add_argument("--tokenizer",    type=str, default="deepseek-r1-distill")
    parser.add_argument("--data-path",    type=str, default="data/distill_data.json")
    parser.add_argument("--output-path",  type=str, default="checkpoints/distill")
    parser.add_argument("--epochs",       type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=2)
    parser.add_argument("--max-seq-len",  type=int, default=4096)
    parser.add_argument("--lr",           type=float, default=1e-5)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(
            f"Distillation data not found: {args.data_path}\n"
            f"Run `python data/prepare_data.py --stage distill` (or provide a JSONL of {{prompt, teacher_response}})."
        )

    from models.transformer import Transformer
    from utils.checkpoint import CheckpointManager
    from transformers import AutoTokenizer

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    def load_from(path: str, label: str) -> Transformer:
        print(f"Initialising {label} model...")
        m = Transformer(yaml_cfg).cuda()
        ckpt = CheckpointManager(path)
        step = ckpt.latest_step()
        if step is not None:
            print(f"  Loading {label} checkpoint step {step}")
            ckpt.load(m, step, device="cuda", strict=False)
        return m

    student = load_from(args.student_path, "student")
    teacher = load_from(args.teacher_path, "teacher")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = DistillationDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=distill_collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    config = {
        "lr": args.lr,
        "distill_alpha": 0.7,
        "temperature": 2.0,
        "max_steps": args.epochs * max(len(dataloader), 1),
    }
    trainer = ReasoningDistillation(student, teacher, config)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        for batch in dataloader:
            losses = trainer.train_step(batch)
            epoch_loss += losses["total_loss"]
            n += 1

        avg = epoch_loss / max(n, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — Loss: {avg:.4f}")

        ckpt = Path(args.output_path) / f"distill_epoch_{epoch+1}.pt"
        torch.save(student.state_dict(), ckpt)
        print(f"Checkpoint → {ckpt}")

    print("Distillation complete.")

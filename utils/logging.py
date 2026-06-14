"""Single-GPU training logger with optional WandB integration (enable with WANDB_PROJECT env var)."""
import json, os, time
from typing import Dict, Optional
import torch


class TrainingLogger:
    """Step-driven logger: prints a rolling-window summary every log_interval steps; optionally forwards to WandB."""

    def __init__(self, log_interval: int = 10, seq_len: int = 1024):
        self.log_interval = log_interval
        self.seq_len = seq_len
        self._start = time.time()
        self._step_start = time.time()
        self._loss_window: list[float] = []
        self._wandb = None
        wandb_project = os.environ.get("WANDB_PROJECT")
        if wandb_project:
            try:
                import wandb
                wandb.init(project=wandb_project, name=os.environ.get("WANDB_RUN_NAME"), reinit=True)
                self._wandb = wandb
            except ImportError:
                print("[logging] wandb not installed -- skipping WandB integration")

    def log(self, step: int, loss: float, metrics: Optional[Dict[str, float]] = None, lr: float = 0.0) -> None:
        self._loss_window.append(loss)
        if step % self.log_interval != 0 or not self._loss_window:
            return
        avg_loss = sum(self._loss_window) / len(self._loss_window)
        elapsed = max(time.time() - self._step_start, 1e-6)
        tokens_per_sec = (self.log_interval * self.seq_len) / elapsed
        ppl = torch.tensor(avg_loss).exp().item()
        parts = [f"step={step:>7}", f"loss={avg_loss:.4f}", f"ppl={ppl:.2f}", f"lr={lr:.2e}", f"tps={tokens_per_sec:,.0f}"]
        if metrics:
            for k, v in metrics.items():
                parts.append(f"{k}={v:.4f}")
        print(" | ".join(parts))
        if self._wandb is not None:
            log_dict = {"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}
            if metrics:
                log_dict.update({f"train/{k}": v for k, v in metrics.items()})
            self._wandb.log(log_dict, step=step)
        self._loss_window = []
        self._step_start = time.time()

    def save_log(self, filename: str, data: Dict) -> None:
        with open(filename, "a") as f:
            f.write(json.dumps(data) + "\n")

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


_logger: Optional[TrainingLogger] = None


def init_logging(log_interval: int = 10, seq_len: int = 1024) -> None:
    global _logger
    _logger = TrainingLogger(log_interval=log_interval, seq_len=seq_len)


def get_logger() -> TrainingLogger:
    global _logger
    if _logger is None:
        _logger = TrainingLogger()
    return _logger

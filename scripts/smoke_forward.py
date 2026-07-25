#!/usr/bin/env python
"""Forward-pass smoke test for the 422M config.

Used by CI to verify the model can be constructed from configs/pretrain_a100_422m.yaml
and run a forward pass on tiny input. Stays CPU-only; no checkpointing or autograd.
"""
import sys
from pathlib import Path

# Allow `python scripts/smoke_forward.py` from repo root — add repo root to sys.path
# so `from models.transformer import Transformer` works without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from models.transformer import Transformer


def main() -> None:
    cfg_path = Path("configs/pretrain_a100_422m.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["model"]
    # Shrink seqlen for fast CI smoke — full 4096 would be wasteful for a smoke test.
    cfg["max_seq_len"] = 16

    model = Transformer(cfg)
    x = torch.randint(0, cfg["vocab_size"], (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg["vocab_size"]), (
        f"unexpected shape {logits.shape}, expected (2, 16, {cfg['vocab_size']})"
    )
    print(f"OK — forward pass succeeded, shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()

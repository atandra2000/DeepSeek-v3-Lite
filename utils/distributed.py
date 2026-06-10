# utils/distributed.py
"""
Single-GPU device helpers.
"""
import torch


DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def device() -> torch.device:
    """Return the single training device."""
    return DEVICE

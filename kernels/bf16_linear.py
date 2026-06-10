# kernels/bf16_linear.py
"""
BF16 linear layer.

Drop-in replacement for `nn.Linear` used by single-GPU BF16 training.
GEMMs go through PyTorch's native cuBLAS path (BF16 SGEMM on A100).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BF16Linear(nn.Linear):
    """Standard `nn.Linear` stored in BF16."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "BF16Linear":
        """Construct a `BF16Linear` from an existing `nn.Linear`."""
        out_f, in_f = linear.weight.shape
        layer = cls(in_f, out_f, bias=linear.bias is not None)
        layer.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.clone()
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def replace_linear_with_bf16(
    model: torch.nn.Module,
    skip_modules: tuple = (),
) -> torch.nn.Module:
    """Recursively replace every `nn.Linear` with `BF16Linear` in-place."""
    for name, module in list(model.named_children()):
        if any(skip in name for skip in skip_modules):
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, BF16Linear.from_linear(module))
        else:
            replace_linear_with_bf16(module, skip_modules)
    return model

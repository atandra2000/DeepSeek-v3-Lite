"""Tests for the MoE Triton grouped-GEMM SwiGLU kernel.

Two layers:
  1. CPU tests: pure-PyTorch reference, import surface, dispatch wiring.
  2. GPU tests (gated by `pytest.mark.skipif`): numerics, gradcheck, stress.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.moe_triton import (  # noqa: E402
    HAS_TRITON,
    grouped_moe_pytorch,
    triton_grouped_moe_dispatch,
)


# -----------------------------------------------------------------------------
# CPU tests
# -----------------------------------------------------------------------------
class TestGroupedMoePytorchReference:
    def test_reference_matches_naive_loop(self):
        torch.manual_seed(0)
        T, D, E, I = 24, 16, 5, 12
        x = torch.randn(T, D, dtype=torch.float32)
        w1 = torch.randn(E, I, D, dtype=torch.float32) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.float32) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.float32) * 0.1
        counts = torch.tensor([6, 4, 5, 3, 6], dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
        gw = torch.rand(T, dtype=torch.float32) * 0.2

        y_ref = grouped_moe_pytorch(x, w1, w2, w3, offsets, gw)
        y_naive = torch.zeros_like(x)
        for e in range(E):
            s, t = offsets[e].item(), offsets[e + 1].item()
            if s == t:
                continue
            chunk = x[s:t]
            gate_pre = chunk @ w1[e].t()
            up = chunk @ w3[e].t()
            h = F.silu(gate_pre) * up
            out = h @ w2[e].t()
            y_naive[s:t] = out * gw[s:t].unsqueeze(-1)
        assert torch.allclose(y_ref, y_naive, atol=1e-5), \
            f"reference diverges (max diff {(y_ref - y_naive).abs().max().item():.2e})"

    def test_reference_with_empty_experts(self):
        torch.manual_seed(1)
        T, D, E, I = 12, 8, 4, 6
        x = torch.randn(T, D, dtype=torch.float32)
        w1 = torch.randn(E, I, D, dtype=torch.float32) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.float32) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.float32) * 0.1
        counts = torch.tensor([3, 0, 5, 4], dtype=torch.int64)  # expert 1 empty
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
        gw = torch.rand(T, dtype=torch.float32) * 0.2
        y = grouped_moe_pytorch(x, w1, w2, w3, offsets, gw)
        assert y.shape == (T, D)
        assert torch.isfinite(y).all()

    def test_reference_matches_unrolled_einsum(self):
        torch.manual_seed(2)
        T, D, E, I = 16, 8, 3, 4
        x = torch.randn(T, D, dtype=torch.float32)
        w1 = torch.randn(E, I, D, dtype=torch.float32) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.float32) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.float32) * 0.1
        counts = torch.tensor([5, 6, 5], dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
        gw = torch.rand(T, dtype=torch.float32) * 0.2
        y_ref = grouped_moe_pytorch(x, w1, w2, w3, offsets, gw)
        y_alt = torch.zeros_like(x)
        for e in range(E):
            s, t = offsets[e].item(), offsets[e + 1].item()
            if s == t:
                continue
            chunk = x[s:t]
            gate_pre = chunk @ w1[e].t()
            up = chunk @ w3[e].t()
            h = F.silu(gate_pre) * up
            out = h @ w2[e].t()
            y_alt[s:t] = out * gw[s:t].unsqueeze(-1)
        assert torch.allclose(y_ref, y_alt, atol=1e-5)


class TestMoeTritonImportSurface:
    def test_module_imports_without_triton(self):
        from models import moe_triton
        expected = "triton" in sys.modules
        assert moe_triton.HAS_TRITON == expected

    def test_kernel_call_raises_clean_import_error(self):
        from models import moe_triton
        if moe_triton.HAS_TRITON:
            pytest.skip("triton installed; import path covered by other test")
        x = torch.randn(4, 8, dtype=torch.float32)
        w1 = torch.randn(2, 4, 8, dtype=torch.float32)
        w2 = torch.randn(2, 8, 4, dtype=torch.float32)
        w3 = torch.randn(2, 4, 8, dtype=torch.float32)
        offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        gw = torch.ones(4, dtype=torch.float32)
        with pytest.raises(ImportError, match="triton"):
            triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)

    def test_kernel_call_raises_value_error_on_too_large_dim(self):
        from models import moe_triton
        if not moe_triton.HAS_TRITON:
            pytest.skip("triton not installed; covered by other test")
        T, D, E, I = 4, 8, 2, 300  # I > 256
        x = torch.randn(T, D, dtype=torch.bfloat16)
        w1 = torch.randn(E, I, D, dtype=torch.bfloat16) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.bfloat16) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.bfloat16) * 0.1
        offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        gw = torch.ones(T, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="256"):
            triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)


class TestMoeTritonDispatchWiring:
    def test_default_dispatch_is_stacked(self):
        from models.moe import DeepSeekMoE
        cfg = {
            "dim": 16, "n_routed_experts": 4, "n_shared_experts": 1,
            "moe_inter_dim": 8, "n_activated_experts": 2,
            "route_scale": 1.0, "bias_upper_threshold": 0.1,
            "bias_lower_threshold": 0.1,
        }
        moe = DeepSeekMoE(cfg)
        assert moe.moe_dispatch == "stacked"

    def test_explicit_stacked_dispatch(self):
        from models.moe import DeepSeekMoE
        cfg = {
            "dim": 16, "n_routed_experts": 4, "n_shared_experts": 1,
            "moe_inter_dim": 8, "n_activated_experts": 2,
            "route_scale": 1.0, "bias_upper_threshold": 0.1,
            "bias_lower_threshold": 0.1, "moe_dispatch": "stacked",
        }
        moe = DeepSeekMoE(cfg)
        assert moe.moe_dispatch == "stacked"

    def test_explicit_triton_dispatch(self):
        from models.moe import DeepSeekMoE
        cfg = {
            "dim": 16, "n_routed_experts": 4, "n_shared_experts": 1,
            "moe_inter_dim": 8, "n_activated_experts": 2,
            "route_scale": 1.0, "bias_upper_threshold": 0.1,
            "bias_lower_threshold": 0.1, "moe_dispatch": "triton_grouped",
        }
        moe = DeepSeekMoE(cfg)
        assert moe.moe_dispatch == "triton_grouped"

    def test_triton_path_falls_back_cleanly_on_cpu(self):
        """`moe_dispatch='triton_grouped'` on a CPU/Mac box must auto-fall-back to 'stacked' with a one-time warning."""
        from models.moe import DeepSeekMoE
        from models import moe_triton
        if moe_triton.HAS_TRITON:
            pytest.skip("triton installed; fallback not exercised")
        cfg = {
            "dim": 16, "n_routed_experts": 4, "n_shared_experts": 1,
            "moe_inter_dim": 8, "n_activated_experts": 2,
            "route_scale": 1.0, "bias_upper_threshold": 0.1,
            "bias_lower_threshold": 0.1,
        }
        m_stacked = DeepSeekMoE(dict(cfg, moe_dispatch="stacked"))
        m_triton = DeepSeekMoE(dict(cfg, moe_dispatch="triton_grouped"))
        m_triton.load_state_dict(m_stacked.state_dict())
        x = torch.randn(8, 16)
        y_stacked = m_stacked(x)
        y_triton = m_triton(x)
        assert torch.allclose(y_stacked, y_triton, atol=1e-5)


# -----------------------------------------------------------------------------
# GPU tests
# -----------------------------------------------------------------------------
gpu_required = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires triton + CUDA",
)


@gpu_required
class TestMoeTritonKernelGPU:
    def test_forward_matches_pytorch_tiny(self):
        torch.manual_seed(42)
        T, D, E, I = 32, 32, 4, 16
        device = torch.device("cuda")
        x = torch.randn(T, D, dtype=torch.float32, device=device) * 0.1
        w1 = torch.randn(E, I, D, dtype=torch.float32, device=device) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.float32, device=device) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.float32, device=device) * 0.1
        counts = torch.tensor([8, 6, 10, 8], dtype=torch.int64, device=device)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)]
        )
        gw = torch.rand(T, dtype=torch.float32, device=device) * 0.1
        y_ref = grouped_moe_pytorch(x, w1, w2, w3, offsets, gw)
        y_tri = triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)
        assert torch.allclose(y_ref, y_tri, atol=1e-3, rtol=1e-3)

    def test_forward_matches_pytorch_bf16(self):
        torch.manual_seed(43)
        T, D, E, I = 64, 48, 4, 24
        device = torch.device("cuda")
        x = torch.randn(T, D, dtype=torch.bfloat16, device=device) * 0.1
        w1 = torch.randn(E, I, D, dtype=torch.bfloat16, device=device) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.bfloat16, device=device) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.bfloat16, device=device) * 0.1
        counts = torch.tensor([16, 14, 18, 16], dtype=torch.int64, device=device)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)]
        )
        gw = torch.rand(T, dtype=torch.bfloat16, device=device) * 0.1
        y_ref = grouped_moe_pytorch(x, w1, w2, w3, offsets, gw)
        y_tri = triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)
        assert torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2)

    def test_forward_with_empty_experts(self):
        torch.manual_seed(44)
        T, D, E, I = 24, 16, 4, 12
        device = torch.device("cuda")
        x = torch.randn(T, D, dtype=torch.bfloat16, device=device) * 0.1
        w1 = torch.randn(E, I, D, dtype=torch.bfloat16, device=device) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.bfloat16, device=device) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.bfloat16, device=device) * 0.1
        counts = torch.tensor([6, 0, 8, 10], dtype=torch.int64, device=device)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)]
        )
        gw = torch.rand(T, dtype=torch.bfloat16, device=device) * 0.1
        y = triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)
        assert torch.isfinite(y).all()

    def test_forward_422m_config_shape(self):
        """422M config has moe_inter_dim=384; kernel must hard-fail."""
        device = torch.device("cuda")
        T, D, E, I = 8, 768, 20, 384
        x = torch.randn(T, D, dtype=torch.bfloat16, device=device) * 0.1
        w1 = torch.randn(E, I, D, dtype=torch.bfloat16, device=device) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.bfloat16, device=device) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.bfloat16, device=device) * 0.1
        counts = torch.full((E,), T // E, dtype=torch.int64, device=device)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)]
        )
        gw = torch.rand(T, dtype=torch.bfloat16, device=device) * 0.1
        with pytest.raises(ValueError, match="256"):
            triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)

    def test_autograd_gradcheck_tiny(self):
        """Run forward + backward, verify every input got a gradient."""
        torch.manual_seed(45)
        T, D, E, I = 32, 8, 2, 8
        device = torch.device("cuda")
        x = torch.randn(T, D, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        w1 = torch.randn(E, I, D, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        counts = torch.tensor([16, 16], dtype=torch.int64, device=device)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)]
        )
        gw = torch.rand(T, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        y = triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)
        y.sum().backward()
        for name, p in [("x", x), ("w1", w1), ("w2", w2), ("w3", w3), ("gw", gw)]:
            assert p.grad is not None, f"{name} has no grad after backward"

    def test_backward_matches_pytorch_tiny(self):
        torch.manual_seed(46)
        T, D, E, I = 16, 8, 2, 8
        device = torch.device("cuda")
        x = torch.randn(T, D, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        w1 = torch.randn(E, I, D, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        w2 = torch.randn(E, D, I, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        w3 = torch.randn(E, I, D, dtype=torch.float32, device=device, requires_grad=True) * 0.1
        counts = torch.tensor([8, 8], dtype=torch.int64, device=device)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)]
        )
        gw = torch.rand(T, dtype=torch.float32, device=device, requires_grad=True) * 0.1

        y_ref = grouped_moe_pytorch(x, w1, w2, w3, offsets, gw)
        g = torch.randn_like(y_ref)
        y_ref.backward(g)
        assert x.grad is not None and w1.grad is not None and w2.grad is not None and w3.grad is not None
        x_ref_grad = x.grad.clone()
        w1_ref_grad = w1.grad.clone()
        w2_ref_grad = w2.grad.clone()
        w3_ref_grad = w3.grad.clone()
        x.grad = None; w1.grad = None; w2.grad = None; w3.grad = None

        y_tri = triton_grouped_moe_dispatch(x, w1, w2, w3, gw, offsets)
        y_tri.backward(g)
        for name, ref, tri in [
            ("dx", x_ref_grad, x.grad),
            ("dw1", w1_ref_grad, w1.grad),
            ("dw2", w2_ref_grad, w2.grad),
            ("dw3", w3_ref_grad, w3.grad),
        ]:
            assert torch.allclose(ref, tri, atol=1e-3, rtol=1e-3), \
                f"{name} diverges (max diff {(ref - tri).abs().max().item():.2e})"

"""Tests for the MLA Triton fused-attention kernel.

CPU: pure-PyTorch reference and import surface.
GPU: optional numerics vs reference (gated on CUDA + triton).
"""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mla_triton import (  # noqa: E402
    HAS_TRITON,
    mla_attention_reference,
    triton_mla_attention,
)


def _tiny_mla_tensors(
    *,
    B: int = 2,
    H: int = 4,
    S: int = 8,
    R: int = 16,
    D_nope: int = 12,
    D_rope: int = 8,
    D_v: int = 16,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q_nope = torch.randn(B, H, S, D_nope, dtype=dtype)
    q_pe = torch.randn(B, H, S, D_rope, dtype=dtype)
    ctx_kv = torch.randn(B, S, R, dtype=dtype)
    ctx_pe = torch.randn(B, S, D_rope, dtype=dtype)
    wkv_b_k = torch.randn(H, D_nope, R, dtype=dtype) * 0.1
    wkv_b_v = torch.randn(H, D_v, R, dtype=dtype) * 0.1
    scale = (D_nope + D_rope) ** -0.5
    return q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale


class TestMlaAttentionReference:
    def test_output_shape(self):
        q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale = _tiny_mla_tensors()
        out = mla_attention_reference(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale)
        assert out.shape == (2, 8, 4, 16)
        assert torch.isfinite(out).all()

    def test_decode_step_single_query(self):
        B, H, R, D_nope, D_rope, D_v = 1, 2, 8, 6, 4, 10
        q_nope = torch.randn(B, H, 1, D_nope)
        q_pe = torch.randn(B, H, 1, D_rope)
        ctx_kv = torch.randn(B, 5, R)
        ctx_pe = torch.randn(B, 5, D_rope)
        wkv_b_k = torch.randn(H, D_nope, R) * 0.1
        wkv_b_v = torch.randn(H, D_v, R) * 0.1
        scale = (D_nope + D_rope) ** -0.5
        out = mla_attention_reference(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale)
        assert out.shape == (B, 1, H, D_v)


class TestMlaTritonImport:
    def test_module_imports_without_triton(self):
        from models import mla_triton

        expected = "triton" in sys.modules
        assert mla_triton.HAS_TRITON == expected

    def test_triton_raises_without_triton(self):
        from models import mla_triton

        if mla_triton.HAS_TRITON:
            pytest.skip("triton installed; ImportError path not exercised")
        q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale = _tiny_mla_tensors()
        with pytest.raises(ImportError, match="triton"):
            triton_mla_attention(
                q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale,
            )


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires triton + CUDA",
)
class TestMlaTritonKernelGPU:
    def test_triton_matches_reference_bf16(self):
        q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale = _tiny_mla_tensors(
            B=1, H=4, S=16, R=32, D_nope=16, D_rope=8, D_v=24, dtype=torch.bfloat16,
        )
        device = "cuda"
        tensors = [t.to(device) for t in (q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v)]
        q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v = tensors

        ref = mla_attention_reference(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale)
        tri = triton_mla_attention(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale)
        assert torch.allclose(ref, tri, atol=1e-2, rtol=1e-2)

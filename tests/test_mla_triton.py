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

    def test_causal_q_start_single_query_sees_all_keys(self):
        """A single query at q_start must attend the whole context (no future)."""
        q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale = _tiny_mla_tensors(
            B=1, H=2, S=10, R=8, D_nope=6, D_rope=4, D_v=8,
        )
        q_nope = q_nope[:, :, :1]  # (1, H, 1, D_nope)
        q_pe = q_pe[:, :, :1]
        out_causal = mla_attention_reference(
            q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale,
            is_causal=True, q_start=9,
        )
        out_plain = mla_attention_reference(
            q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale,
            is_causal=False,
        )
        assert torch.allclose(out_causal, out_plain, atol=1e-6), \
            "last-position query must attend all keys regardless of causal mode"

    def test_causal_q_start_blocks_future_keys(self):
        """Queries at q_start must not attend keys beyond their global position."""
        q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v, scale = _tiny_mla_tensors(
            B=1, H=2, S=6, R=8, D_nope=6, D_rope=4, D_v=8,
        )
        # Emulate a chunk at global positions [6, 12) over a 12-key context.
        q_nope_glob = torch.randn(1, 2, 6, 6)
        q_pe_glob = torch.randn(1, 2, 6, 4)
        ctx_kv_glob = torch.randn(1, 12, 8)
        ctx_pe_glob = torch.randn(1, 12, 4)
        out = mla_attention_reference(
            q_nope_glob, q_pe_glob, ctx_kv_glob, ctx_pe_glob, wkv_b_k, wkv_b_v, scale,
            is_causal=True, q_start=6,
        )
        # Manual masked softmax: query i (global pos 6+i) attends keys <= 6+i.
        K_nope = torch.einsum("bsr,hdr->bhsd", ctx_kv_glob, wkv_b_k)
        V = torch.einsum("bsr,hdr->bhsd", ctx_kv_glob, wkv_b_v)
        Q = torch.cat([q_nope_glob, q_pe_glob], dim=-1)
        K = torch.cat([K_nope, ctx_pe_glob.unsqueeze(1).expand(1, 2, 12, 4)], dim=-1)
        q_idx = (torch.arange(6)[:, None] + 6)
        k_idx = torch.arange(12)[None, :]
        mask = torch.where(q_idx >= k_idx, 0.0, float("-inf"))
        attn = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale + mask, dim=-1)
        manual = torch.einsum("bhqk,bhkd->bhqd", attn, V).permute(0, 2, 1, 3)
        assert torch.allclose(out, manual, atol=1e-5), "q_start causal masking mismatch"


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

    def test_full_model_sdpa_and_triton_agree(self, small_cfg, monkeypatch):
        """Full-model forward must agree between the PyTorch default config and
        the Triton config (MLA fused kernel + MoE grouped kernel). AGENTS TODO."""
        import os
        from models.transformer import Transformer

        monkeypatch.setenv("ENABLE_TRITON_KERNELS", "1")
        torch.manual_seed(0)
        ref_cfg = dict(small_cfg, attn_impl="sdpa", moe_dispatch="stacked")
        tri_cfg = dict(small_cfg, attn_impl="triton", moe_dispatch="triton_grouped")
        ref = Transformer(ref_cfg, use_checkpoint=False).to("cuda", dtype=torch.bfloat16).eval()
        tri = Transformer(tri_cfg, use_checkpoint=False).to("cuda", dtype=torch.bfloat16).eval()
        tri.load_state_dict(ref.state_dict())
        x = torch.randint(0, small_cfg["vocab_size"] - 1, (2, 32), device="cuda")
        with torch.no_grad():
            r = ref(x, start_pos=0, use_cache=False)
            t = tri(x, start_pos=0, use_cache=False)
        assert torch.allclose(r, t, atol=2e-2, rtol=2e-2), \
            f"sdpa/triton full-model logits diverge: max diff {(r - t).abs().max().item():.4f}"

    def test_triton_chunked_prefill_matches_full(self, small_cfg, monkeypatch):
        """Cached mid-sequence prefill with the Triton kernel must stay causal
        (q_start offset) and equal a full forward."""
        from models.transformer import Transformer

        monkeypatch.setenv("ENABLE_TRITON_KERNELS", "1")
        cfg = dict(small_cfg, attn_impl="triton", moe_dispatch="stacked")
        m = Transformer(cfg, use_checkpoint=False).to("cuda", dtype=torch.bfloat16).eval()
        torch.manual_seed(0)
        x = torch.randint(0, cfg["vocab_size"] - 1, (2, 16), device="cuda")
        with torch.no_grad():
            full = m(x, start_pos=0, use_cache=False)
            m.reset_cache()
            _ = m(x[:, :8], start_pos=0, use_cache=True)
            chunk2 = m(x[:, 8:], start_pos=8, use_cache=True)
        assert torch.allclose(full[:, 8:], chunk2, atol=2e-2, rtol=2e-2), \
            f"triton chunked prefill diverges: max diff {(full[:, 8:] - chunk2).abs().max().item():.4f}"

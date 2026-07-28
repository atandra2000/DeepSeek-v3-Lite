"""Fused MLA attention Triton kernel.

Fuses K_nope/V materialisation, RoPE, and SDPA into a single
FlashAttention-2-style kernel. Replaces models/mla.py:127-143.
See `documentation/triton_kernels.md` §4 for the design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import torch

if TYPE_CHECKING:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Pure-PyTorch reference for the SDPA path. Self-contained (no KV cache,
# no RoPE pre-application). Used by CPU tests.
def mla_attention_reference(
    q_nope: torch.Tensor,        # (B, H, S_q, D_nope)
    q_pe: torch.Tensor,          # (B, H, S_q, D_rope)
    ctx_kv: torch.Tensor,        # (B, S_kv, R)
    ctx_pe: torch.Tensor,        # (B, S_kv, D_rope)
    wkv_b_k: torch.Tensor,       # (H, D_nope, R)
    wkv_b_v: torch.Tensor,       # (H, D_v, R)
    softmax_scale: float,
) -> torch.Tensor:
    """Returns (B, S_q, H, D_v)."""
    B, H, S_q, D_nope = q_nope.shape
    D_rope = q_pe.shape[-1]
    S_kv = ctx_kv.size(1)
    D_v = wkv_b_v.size(1)

    K_nope = torch.einsum("bsr,hdr->bhsd", ctx_kv, wkv_b_k)
    V = torch.einsum("bsr,hdr->bhsd", ctx_kv, wkv_b_v)
    K_rope = ctx_pe.unsqueeze(1)

    Q_full = torch.cat([q_nope, q_pe], dim=-1)
    K_full = torch.cat([K_nope, K_rope.expand(B, H, S_kv, D_rope)], dim=-1)

    scores = torch.einsum("bhqd,bhkd->bhqk", Q_full, K_full) * softmax_scale
    if S_q == S_kv:
        mask = torch.triu(
            torch.full((S_q, S_kv), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask
    attn = scores.softmax(dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", attn, V)
    return out.permute(0, 2, 1, 3).contiguous()


# -----------------------------------------------------------------------------
# Triton forward kernel — FA2-style fused MLA attention
# -----------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _mla_flash_fwd_kernel(
        q_nope_ptr, q_pe_ptr,
        ctx_kv_ptr, ctx_pe_ptr,
        wkv_b_k_ptr, wkv_b_v_ptr,
        out_ptr,
        B, H, S_q, S_kv, R, D_nope, D_rope, D_v,
        stride_qn_b, stride_qn_h, stride_qn_s, stride_qn_d,
        stride_qp_b, stride_qp_h, stride_qp_s, stride_qp_d,
        stride_kv_b, stride_kv_s, stride_kv_r,
        stride_pe_b, stride_pe_s, stride_pe_d,
        stride_wk_h, stride_wk_n, stride_wk_r,
        stride_wv_h, stride_wv_v, stride_wv_r,
        stride_o_b, stride_o_s, stride_o_h, stride_o_v,
        softmax_scale,
        is_causal: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D_NOPE: tl.constexpr,
        BLOCK_D_ROPE: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One program per (B, H, query-block). K_nope and V are materialised
        # inside the inner K-block loop from per-head wkv_b_k / wkv_b_v, so
        # the K_nope / V HBM roundtrips in the SDPA path disappear. Query
        # tiling (BLOCK_Q) keeps the score tile bounded for training seq lens.
        b_id = tl.program_id(0)
        h_id = tl.program_id(1)
        q_blk = tl.program_id(2)

        s_q_off = q_blk * BLOCK_Q + tl.arange(0, BLOCK_Q)
        s_q_mask = s_q_off < S_q
        r_idx = tl.arange(0, BLOCK_R)
        d_nope_idx = tl.arange(0, BLOCK_D_NOPE)
        d_rope_idx = tl.arange(0, BLOCK_D_ROPE)
        d_v_idx = tl.arange(0, BLOCK_D_V)
        n_idx = tl.arange(0, BLOCK_N)

        q_nope_tile = tl.load(
            q_nope_ptr + b_id * stride_qn_b + h_id * stride_qn_h
            + s_q_off[:, None] * stride_qn_s + d_nope_idx[None, :] * stride_qn_d,
            mask=s_q_mask[:, None], other=0.0,
        )
        q_pe_tile = tl.load(
            q_pe_ptr + b_id * stride_qp_b + h_id * stride_qp_h
            + s_q_off[:, None] * stride_qp_s + d_rope_idx[None, :] * stride_qp_d,
            mask=s_q_mask[:, None], other=0.0,
        )

        # FA2 online-softmax accumulators
        m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D_V), dtype=tl.float32)

        # Per-head up-projection weights stay in registers across the loop
        w_k = tl.load(
            wkv_b_k_ptr + h_id * stride_wk_h
            + d_nope_idx[:, None] * stride_wk_n + r_idx[None, :] * stride_wk_r,
            mask=(d_nope_idx[:, None] < D_nope), other=0.0,
        )
        w_v = tl.load(
            wkv_b_v_ptr + h_id * stride_wv_h
            + d_v_idx[:, None] * stride_wv_v + r_idx[None, :] * stride_wv_r,
            mask=(d_v_idx[:, None] < D_v), other=0.0,
        )

        for kv_start in range(0, S_kv, BLOCK_N):
            k_off = kv_start + n_idx
            k_mask = k_off < S_kv

            kv_tile = tl.load(
                ctx_kv_ptr + b_id * stride_kv_b
                + k_off[:, None] * stride_kv_s + r_idx[None, :] * stride_kv_r,
                mask=k_mask[:, None], other=0.0,
            )
            k_nope = tl.dot(kv_tile, tl.trans(w_k))
            v_tile = tl.dot(kv_tile, tl.trans(w_v))

            k_pe = tl.load(
                ctx_pe_ptr + b_id * stride_pe_b
                + k_off[:, None] * stride_pe_s + d_rope_idx[None, :] * stride_pe_d,
                mask=k_mask[:, None], other=0.0,
            )

            s_block = (tl.dot(q_nope_tile, tl.trans(k_nope))
                       + tl.dot(q_pe_tile, tl.trans(k_pe))) * softmax_scale

            if is_causal:
                causal_mask = (s_q_off[:, None] >= k_off[None, :])
                s_block = tl.where(causal_mask, s_block, float("-inf"))
            s_block = tl.where(k_mask[None, :], s_block, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s_block, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s_block - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m_i = m_new

        acc = acc / l_i[:, None]

        tl.store(
            out_ptr + b_id * stride_o_b
            + s_q_off[:, None] * stride_o_s + h_id * stride_o_h + d_v_idx[None, :] * stride_o_v,
            acc.to(out_ptr.dtype.element_ty),
            mask=s_q_mask[:, None],
        )


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _check_mla_dim_limits(
    S_q: int, S_kv: int, R: int, D_nope: int, D_rope: int, D_v: int,
) -> None:
    """Hard-fail if any dim exceeds 256 (register budget)."""
    for name, val in [
        ("R", R), ("D_nope", D_nope), ("D_rope", D_rope), ("D_v", D_v),
    ]:
        if val > 256:
            raise ValueError(
                f"triton_mla_attention: {name}={val} exceeds the 256 cap. "
                "Reduce or tile manually before launching the kernel."
            )


# -----------------------------------------------------------------------------
# Public autograd Function
# -----------------------------------------------------------------------------
if HAS_TRITON:

    class _TritonMlaAttentionFunction(torch.autograd.Function):

        @staticmethod
        def forward(
            ctx: Any,
            q_nope: torch.Tensor,         # (B, H, S_q, D_nope)
            q_pe: torch.Tensor,           # (B, H, S_q, D_rope)
            ctx_kv: torch.Tensor,         # (B, S_kv, R)
            ctx_pe: torch.Tensor,         # (B, S_kv, D_rope)
            wkv_b_k: torch.Tensor,        # (H, D_nope, R)
            wkv_b_v: torch.Tensor,        # (H, D_v, R)
            softmax_scale: float,
            is_causal: bool,
        ) -> torch.Tensor:
            B, H, S_q, D_nope = q_nope.shape
            S_kv = ctx_kv.size(1)
            R = ctx_kv.size(-1)
            D_rope = q_pe.shape[-1]
            D_v = wkv_b_v.size(1)

            _check_mla_dim_limits(S_q, S_kv, R, D_nope, D_rope, D_v)

            BLOCK_Q = 64
            BLOCK_N = 64
            BLOCK_R = _next_pow2(R)
            BLOCK_D_NOPE = _next_pow2(D_nope)
            BLOCK_D_ROPE = _next_pow2(D_rope)
            BLOCK_D_V = _next_pow2(D_v)
            n_q_blocks = (S_q + BLOCK_Q - 1) // BLOCK_Q

            out = torch.empty(B, S_q, H, D_v, dtype=q_nope.dtype, device=q_nope.device)

            _mla_flash_fwd_kernel[(B, H, n_q_blocks)](
                q_nope, q_pe,
                ctx_kv, ctx_pe,
                wkv_b_k, wkv_b_v,
                out,
                B, H, S_q, S_kv, R, D_nope, D_rope, D_v,
                q_nope.stride(0), q_nope.stride(1), q_nope.stride(2), q_nope.stride(3),
                q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), q_pe.stride(3),
                ctx_kv.stride(0), ctx_kv.stride(1), ctx_kv.stride(2),
                ctx_pe.stride(0), ctx_pe.stride(1), ctx_pe.stride(2),
                wkv_b_k.stride(0), wkv_b_k.stride(1), wkv_b_k.stride(2),
                wkv_b_v.stride(0), wkv_b_v.stride(1), wkv_b_v.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                softmax_scale,
                is_causal=is_causal,
                BLOCK_R=BLOCK_R,
                BLOCK_D_NOPE=BLOCK_D_NOPE,
                BLOCK_D_ROPE=BLOCK_D_ROPE,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_Q=BLOCK_Q,
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=2,
            )

            ctx.save_for_backward(q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v)
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.B, ctx.H, ctx.S_q, ctx.S_kv = B, H, S_q, S_kv
            ctx.R, ctx.D_nope, ctx.D_rope, ctx.D_v = R, D_nope, D_rope, D_v
            return out

        @staticmethod
        def backward(
            ctx: Any, *grad_outputs: torch.Tensor,
        ) -> Tuple[torch.Tensor, ...]:
            # v1 stub: re-run the reference forward and use PyTorch
            # autograd. Correct but not yet the optimal re-compute
            # backward. See `documentation/triton_kernels.md` §4.
            dout = grad_outputs[0]
            if dout is None:
                return (None,) * 7
            q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v = ctx.saved_tensors
            out_ref = mla_attention_reference(
                q_nope=q_nope, q_pe=q_pe,
                ctx_kv=ctx_kv, ctx_pe=ctx_pe,
                wkv_b_k=wkv_b_k, wkv_b_v=wkv_b_v,
                softmax_scale=ctx.softmax_scale,
            )
            grads = torch.autograd.grad(
                out_ref,
                [q_nope, q_pe, ctx_kv, ctx_pe, wkv_b_k, wkv_b_v],
                grad_outputs=dout,
                allow_unused=True,
            )
            return (
                grads[0] if grads[0] is not None else torch.zeros_like(q_nope),
                grads[1] if grads[1] is not None else torch.zeros_like(q_pe),
                grads[2] if grads[2] is not None else torch.zeros_like(ctx_kv),
                grads[3] if grads[3] is not None else torch.zeros_like(ctx_pe),
                grads[4] if grads[4] is not None else torch.zeros_like(wkv_b_k),
                grads[5] if grads[5] is not None else torch.zeros_like(wkv_b_v),
                None, None,
            )


def triton_mla_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ctx_kv: torch.Tensor,
    ctx_pe: torch.Tensor,
    wkv_b_k: torch.Tensor,
    wkv_b_v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused MLA attention. Returns (B, S_q, H, D_v).

    Raises ImportError if triton is not installed.
    """
    if not HAS_TRITON:
        raise ImportError(
            "triton_mla_attention requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, use `attn_impl='sdpa'` in your config."
        )
    return _TritonMlaAttentionFunction.apply(
        q_nope, q_pe, ctx_kv, ctx_pe,
        wkv_b_k, wkv_b_v, softmax_scale, is_causal,
    )

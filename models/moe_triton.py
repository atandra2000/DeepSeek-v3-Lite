"""Fused grouped-GEMM SwiGLU kernel for DeepSeekMoE routed-expert dispatch.

Replaces the `for e in range(E): ...` Python loop in DeepSeekMoE.forward
with one Triton launch over a sorted-token layout. See
`documentation/triton_kernels.md` §3 for the design.
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


# Pure-PyTorch reference. Same arithmetic as the `stacked` path but
# routed through the sorted-token layout; used by CPU tests.
def grouped_moe_pytorch(
    x_sorted: torch.Tensor,         # (T, D)
    w1: torch.Tensor,               # (E, I, D)
    w2: torch.Tensor,               # (E, D, I)
    w3: torch.Tensor,               # (E, I, D)
    expert_offsets: torch.Tensor,   # (E+1,) INT64 cumsum boundaries
    sorted_weights: torch.Tensor,   # (T,)
) -> torch.Tensor:
    """Returns y_sorted of shape (T, D)."""
    T, D = x_sorted.shape
    E, I, _ = w1.shape
    y_sorted = torch.zeros(T, D, dtype=x_sorted.dtype, device=x_sorted.device)
    offsets_cpu = expert_offsets.tolist()
    for e in range(E):
        start, end = offsets_cpu[e], offsets_cpu[e + 1]
        if start == end:
            continue
        chunk = x_sorted[start:end]
        gate_pre = chunk @ w1[e].t()
        up = chunk @ w3[e].t()
        h = torch.nn.functional.silu(gate_pre) * up
        out = h @ w2[e].t()
        y_sorted[start:end] = out * sorted_weights[start:end].unsqueeze(-1)
    return y_sorted


# -----------------------------------------------------------------------------
# Triton forward kernel — fused grouped-GEMM SwiGLU
# -----------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _grouped_moe_fwd_kernel(
        x_ptr, w1_ptr, w2_ptr, w3_ptr,
        gw_ptr,
        offsets_ptr,
        y_ptr,
        T, D, I, E,
        stride_x_t, stride_x_d,
        stride_w1_e, stride_w1_i, stride_w1_d,
        stride_w3_e, stride_w3_i, stride_w3_d,
        stride_w2_e, stride_w2_d, stride_w2_i,
        stride_gw_t,
        stride_y_t, stride_y_d,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        # One program per (expert, token-block). Caller enforces
        # BLOCK_I = next_pow2(I), BLOCK_D = next_pow2(D), I,D <= 256.
        e = tl.program_id(0)
        pid_t = tl.program_id(1)

        start = tl.load(offsets_ptr + e)
        end = tl.load(offsets_ptr + e + 1)
        n_tokens_e = end - start
        if n_tokens_e == 0:
            return

        t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_off < n_tokens_e
        d_idx = tl.arange(0, BLOCK_D)
        i_idx = tl.arange(0, BLOCK_I)

        # gate, up: tile over D, accumulate fp32
        gate_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d_off = d_start + d_idx
            d_mask = d_off < D
            x_tile = tl.load(
                x_ptr + (start + t_off)[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            w1_tile = tl.load(
                w1_ptr + e * stride_w1_e
                + i_idx[:, None] * stride_w1_i
                + d_off[None, :] * stride_w1_d,
                mask=d_mask[None, :], other=0.0,
            )
            w3_tile = tl.load(
                w3_ptr + e * stride_w3_e
                + i_idx[:, None] * stride_w3_i
                + d_off[None, :] * stride_w3_d,
                mask=d_mask[None, :], other=0.0,
            )
            gate_acc = tl.dot(x_tile, tl.trans(w1_tile), acc=gate_acc)
            up_acc = tl.dot(x_tile, tl.trans(w3_tile), acc=up_acc)

        h = tl.sigmoid(gate_acc) * gate_acc * up_acc

        # out: single GEMM since BLOCK_I is full I
        w2_tile = tl.load(
            w2_ptr + e * stride_w2_e
            + d_idx[:, None] * stride_w2_d
            + i_idx[None, :] * stride_w2_i,
            mask=(d_idx[:, None] < D) & (i_idx[None, :] < I),
            other=0.0,
        )
        # `h` is fp32 (from gate_acc/up_acc SwiGLU accumulators). Cast to the
        # input dtype so the final bf16·bf16 dot has matching operand dtypes.
        h_typed = h.to(x_ptr.dtype.element_ty)
        out_acc = tl.dot(h_typed, tl.trans(w2_tile))

        gw = tl.load(gw_ptr + (start + t_off) * stride_gw_t, mask=t_mask, other=0.0)
        out_acc = out_acc * gw[:, None]

        tl.store(
            y_ptr + (start + t_off)[:, None] * stride_y_t + d_idx[None, :] * stride_y_d,
            out_acc.to(tl.bfloat16),
            mask=t_mask[:, None] & (d_idx[None, :] < D),
        )

    # Backward: re-compute silu(g), u, h on the fly from saved x,w1,w2,w3.
    # dh = dy@w2; dsilu = dh*u; dgate_pre = dsilu*silu'(g);
    # dup = dsilu*silu(g); dx = dgate_pre@w1 + dup@w3.
    @triton.jit
    def _grouped_moe_bwd_dx_kernel(
        x_ptr, w1_ptr, w2_ptr, w3_ptr, gw_ptr, offsets_ptr,
        dy_ptr, dx_ptr,
        T, D, I, E,
        stride_x_t, stride_x_d,
        stride_w1_e, stride_w1_i, stride_w1_d,
        stride_w3_e, stride_w3_i, stride_w3_d,
        stride_w2_e, stride_w2_d, stride_w2_i,
        stride_dy_t, stride_dy_d,
        stride_dx_t, stride_dx_d,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        e = tl.program_id(0)
        pid_t = tl.program_id(1)

        start = tl.load(offsets_ptr + e)
        end = tl.load(offsets_ptr + e + 1)
        n_tokens_e = end - start
        if n_tokens_e == 0:
            return

        t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_off < n_tokens_e
        d_idx = tl.arange(0, BLOCK_D)
        i_idx = tl.arange(0, BLOCK_I)

        # Re-compute gate_acc, up_acc
        gate_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d_off = d_start + d_idx
            d_mask = d_off < D
            x_tile = tl.load(
                x_ptr + (start + t_off)[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                mask=t_mask[:, None] & d_mask[None, :], other=0.0,
            )
            w1_tile = tl.load(
                w1_ptr + e * stride_w1_e
                + i_idx[:, None] * stride_w1_i
                + d_off[None, :] * stride_w1_d,
                mask=d_mask[None, :], other=0.0,
            )
            w3_tile = tl.load(
                w3_ptr + e * stride_w3_e
                + i_idx[:, None] * stride_w3_i
                + d_off[None, :] * stride_w3_d,
                mask=d_mask[None, :], other=0.0,
            )
            gate_acc = tl.dot(x_tile, tl.trans(w1_tile), acc=gate_acc)
            up_acc = tl.dot(x_tile, tl.trans(w3_tile), acc=up_acc)

        sig_g = tl.sigmoid(gate_acc)
        silu_g = sig_g * gate_acc
        h = silu_g * up_acc

        dy_tile = tl.load(
            dy_ptr + (start + t_off)[:, None] * stride_dy_t + d_idx[None, :] * stride_dy_d,
            mask=t_mask[:, None] & (d_idx[None, :] < D), other=0.0,
        ).to(tl.float32)
        gw = tl.load(gw_ptr + (start + t_off), mask=t_mask, other=0.0)
        dy_w = dy_tile * gw[:, None]

        w2_tile = tl.load(
            w2_ptr + e * stride_w2_e
            + d_idx[:, None] * stride_w2_d
            + i_idx[None, :] * stride_w2_i,
            mask=(d_idx[:, None] < D) & (i_idx[None, :] < I), other=0.0,
        )
        dh = tl.dot(dy_w, w2_tile)

        # silu'(g) = sig(g) * (1 + g*(1 - sig(g)))
        dsilu = dh * up_acc
        dgate_pre = dsilu * sig_g * (1.0 + gate_acc * (1.0 - sig_g))
        dup = dsilu * silu_g

        d_b = tl.arange(0, BLOCK_D)
        w1_re = tl.load(
            w1_ptr + e * stride_w1_e
            + i_idx[:, None] * stride_w1_i
            + d_b[None, :],
            mask=(i_idx[:, None] < I) & (d_b[None, :] < D), other=0.0,
        )
        w3_re = tl.load(
            w3_ptr + e * stride_w3_e
            + i_idx[:, None] * stride_w3_i
            + d_b[None, :],
            mask=(i_idx[:, None] < I) & (d_b[None, :] < D), other=0.0,
        )
        dx = tl.dot(dgate_pre, w1_re) + tl.dot(dup, w3_re)

        tl.store(
            dx_ptr + (start + t_off)[:, None] * stride_dx_t + d_idx[None, :] * stride_dx_d,
            dx.to(tl.bfloat16),
            mask=t_mask[:, None] & (d_idx[None, :] < D),
        )

    # One program per expert. No atomics: each program owns its (I, D)
    # gradient tile and writes directly after the token loop.
    @triton.jit
    def _grouped_moe_bwd_dw_kernel(
        x_ptr, w1_ptr, w2_ptr, w3_ptr, gw_ptr, offsets_ptr,
        dy_ptr,
        dw1_ptr, dw3_ptr, dw2_ptr,           # (E, I, D), (E, I, D), (E, D, I)
        T, D, I, E,
        stride_x_t, stride_x_d,
        stride_w1_e, stride_w1_i, stride_w1_d,
        stride_w3_e, stride_w3_i, stride_w3_d,
        stride_w2_e, stride_w2_d, stride_w2_i,
        stride_dw1_e, stride_dw1_i, stride_dw1_d,
        stride_dw3_e, stride_dw3_i, stride_dw3_d,
        stride_dw2_e, stride_dw2_d, stride_dw2_i,
        stride_dy_t, stride_dy_d,
        stride_gw_t,
        BLOCK_T: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        e = tl.program_id(0)

        start = tl.load(offsets_ptr + e)
        end = tl.load(offsets_ptr + e + 1)
        n_tokens_e = end - start
        if n_tokens_e == 0:
            return

        d_idx = tl.arange(0, BLOCK_D)
        i_idx = tl.arange(0, BLOCK_I)
        t_idx = tl.arange(0, BLOCK_T)

        for i_start in range(0, I, BLOCK_I):
            for d_start in range(0, D, BLOCK_D):
                i_off = i_start + i_idx
                d_off = d_start + d_idx
                i_mask = i_off < I
                d_mask = d_off < D

                dw1_local = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
                dw3_local = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
                dw2_local = tl.zeros((BLOCK_D, BLOCK_I), dtype=tl.float32)

                for t_start in range(0, n_tokens_e, BLOCK_T):
                    t_off = t_start + t_idx
                    t_mask = t_off < n_tokens_e

                    # Re-compute gate, up, h
                    gate_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
                    up_acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
                    for d_inner in range(0, D, BLOCK_D):
                        d_inner_off = d_inner + d_idx
                        d_inner_mask = d_inner_off < D
                        x_tile = tl.load(
                            x_ptr + (start + t_off)[:, None] * stride_x_t
                            + d_inner_off[None, :] * stride_x_d,
                            mask=t_mask[:, None] & d_inner_mask[None, :], other=0.0,
                        )
                        w1_tile_inner = tl.load(
                            w1_ptr + e * stride_w1_e
                            + i_idx[:, None] * stride_w1_i
                            + d_inner_off[None, :] * stride_w1_d,
                            mask=d_inner_mask[None, :], other=0.0,
                        )
                        w3_tile_inner = tl.load(
                            w3_ptr + e * stride_w3_e
                            + i_idx[:, None] * stride_w3_i
                            + d_inner_off[None, :] * stride_w3_d,
                            mask=d_inner_mask[None, :], other=0.0,
                        )
                        gate_acc = tl.dot(x_tile, tl.trans(w1_tile_inner), acc=gate_acc)
                        up_acc = tl.dot(x_tile, tl.trans(w3_tile_inner), acc=up_acc)
                    sig_g = tl.sigmoid(gate_acc)
                    silu_g = sig_g * gate_acc
                    h = silu_g * up_acc

                    dy_tile = tl.load(
                        dy_ptr + (start + t_off)[:, None] * stride_dy_t
                        + d_off[None, :] * stride_dy_d,
                        mask=t_mask[:, None] & d_mask[None, :], other=0.0,
                    ).to(tl.float32)
                    gw = tl.load(gw_ptr + (start + t_off) * stride_gw_t,
                                 mask=t_mask, other=0.0)
                    dy_w = dy_tile * gw[:, None]

                    w2_tile = tl.load(
                        w2_ptr + e * stride_w2_e
                        + d_off[:, None] * stride_w2_d
                        + i_off[None, :] * stride_w2_i,
                        mask=d_mask[:, None] & i_mask[None, :], other=0.0,
                    )
                    dh = tl.dot(dy_w, w2_tile)

                    dsilu = dh * up_acc
                    dgate_pre = dsilu * sig_g * (1.0 + gate_acc * (1.0 - sig_g))
                    dup = dsilu * silu_g

                    x_tile = tl.load(
                        x_ptr + (start + t_off)[:, None] * stride_x_t
                        + d_off[None, :] * stride_x_d,
                        mask=t_mask[:, None] & d_mask[None, :], other=0.0,
                    )

                    # Mask broadcast: pair each 1-D mask with the axis it owns.
                    # i_mask/d_mask are 1-D; the tensors they gate are 2-D with
                    # shapes (BLOCK_T, BLOCK_I), (BLOCK_I, BLOCK_T), (BLOCK_D, BLOCK_T) —
                    # so use [None, :] (axis 0) rather than [:, None] which would
                    # require axis-0 size to equal BLOCK_I. dw2 dot uses un-transposed h.
                    dgate_masked = tl.where(i_mask[None, :], dgate_pre, 0.0)
                    dup_masked = tl.where(i_mask[None, :], dup, 0.0)
                    t_i_mask = t_mask[:, None] & i_mask[None, :]
                    h_masked = tl.where(t_i_mask, h, 0.0)
                    d_t_mask = d_mask[:, None] & t_mask[None, :]
                    dy_w_t = tl.where(d_t_mask, tl.trans(dy_w), 0.0)

                    dw1_local += tl.dot(tl.trans(dgate_masked), x_tile)
                    dw3_local += tl.dot(tl.trans(dup_masked), x_tile)
                    dw2_local += tl.dot(dy_w_t, h_masked)

                mask_1d = (i_idx[:, None] < I) & (d_off[None, :] < D)
                tl.store(
                    dw1_ptr + e * stride_dw1_e
                    + i_off[:, None] * stride_dw1_i
                    + d_off[None, :] * stride_dw1_d,
                    dw1_local, mask=mask_1d,
                )
                tl.store(
                    dw3_ptr + e * stride_dw3_e
                    + i_off[:, None] * stride_dw3_i
                    + d_off[None, :] * stride_dw3_d,
                    dw3_local, mask=mask_1d,
                )
                mask_2d = (d_off[:, None] < D) & (i_off[None, :] < I)
                tl.store(
                    dw2_ptr + e * stride_dw2_e
                    + d_off[:, None] * stride_dw2_d
                    + i_off[None, :] * stride_dw2_i,
                    dw2_local, mask=mask_2d,
                )


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _check_dim_limits(I: int, D: int) -> None:
    """Hard-fail if I or D exceeds the 256 register-pressure cap."""
    if I > 256 or D > 256:
        raise ValueError(
            f"triton_grouped_moe_dispatch: BLOCK_I=ceil_pow2({I}) and "
            f"BLOCK_D=ceil_pow2({D}) must each be ≤ 256. Got I={I}, D={D}. "
            "For larger dims, fall back to `moe_dispatch='stacked'`."
        )


# -----------------------------------------------------------------------------
# Public autograd Function
# -----------------------------------------------------------------------------
if HAS_TRITON:

    class _TritonGroupedMoeFunction(torch.autograd.Function):

        @staticmethod
        def forward(
            ctx: Any,
            x_sorted: torch.Tensor,         # (T, D) BF16
            w1: torch.Tensor,               # (E, I, D) BF16
            w2: torch.Tensor,               # (E, D, I) BF16
            w3: torch.Tensor,               # (E, I, D) BF16
            sorted_weights: torch.Tensor,   # (T,) BF16
            expert_offsets: torch.Tensor,   # (E+1,) INT64
        ) -> torch.Tensor:
            T, D = x_sorted.shape
            E, I, _ = w1.shape
            _check_dim_limits(I, D)

            BLOCK_T = 32
            BLOCK_D = _next_pow2(D)
            BLOCK_I = _next_pow2(I)
            n_blocks_per_expert = (T + BLOCK_T - 1) // BLOCK_T
            grid = (E, n_blocks_per_expert)

            y_sorted = torch.empty(T, D, dtype=x_sorted.dtype, device=x_sorted.device)

            _grouped_moe_fwd_kernel[grid](
                x_sorted, w1, w2, w3,
                sorted_weights,
                expert_offsets,
                y_sorted,
                T, D, I, E,
                x_sorted.stride(0), x_sorted.stride(1),
                w1.stride(0), w1.stride(1), w1.stride(2),
                w3.stride(0), w3.stride(1), w3.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                sorted_weights.stride(0),
                y_sorted.stride(0), y_sorted.stride(1),
                BLOCK_T=BLOCK_T,
                BLOCK_D=BLOCK_D,
                BLOCK_I=BLOCK_I,
                num_warps=4,
                num_stages=2,
            )

            ctx.save_for_backward(
                x_sorted, w1, w2, w3, sorted_weights, expert_offsets
            )
            ctx.T, ctx.D, ctx.I, ctx.E = T, D, I, E
            ctx.BLOCK_T = BLOCK_T
            ctx.BLOCK_D = BLOCK_D
            ctx.BLOCK_I = BLOCK_I
            return y_sorted

        @staticmethod
        def backward(
            ctx: Any, *grad_outputs: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                   torch.Tensor, torch.Tensor, None]:
            dy_sorted = grad_outputs[0]
            if dy_sorted is None:
                return None, None, None, None, None, None
            x_sorted, w1, w2, w3, sorted_weights, expert_offsets = ctx.saved_tensors
            T, D, I, E = ctx.T, ctx.D, ctx.I, ctx.E
            BLOCK_T = ctx.BLOCK_T
            BLOCK_D = ctx.BLOCK_D
            BLOCK_I = ctx.BLOCK_I

            n_blocks_per_expert = (T + BLOCK_T - 1) // BLOCK_T
            grid_dx = (E, n_blocks_per_expert)
            dx = torch.empty(T, D, dtype=x_sorted.dtype, device=x_sorted.device)
            _grouped_moe_bwd_dx_kernel[grid_dx](
                x_sorted, w1, w2, w3, sorted_weights, expert_offsets,
                dy_sorted, dx,
                T, D, I, E,
                x_sorted.stride(0), x_sorted.stride(1),
                w1.stride(0), w1.stride(1), w1.stride(2),
                w3.stride(0), w3.stride(1), w3.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                dy_sorted.stride(0), dy_sorted.stride(1),
                dx.stride(0), dx.stride(1),
                BLOCK_T=BLOCK_T,
                BLOCK_D=BLOCK_D,
                BLOCK_I=BLOCK_I,
                num_warps=4,
                num_stages=2,
            )

            dw1 = torch.zeros(E, I, D, dtype=torch.float32, device=w1.device)
            dw2 = torch.zeros(E, D, I, dtype=torch.float32, device=w1.device)
            dw3 = torch.zeros(E, I, D, dtype=torch.float32, device=w1.device)
            _grouped_moe_bwd_dw_kernel[(E,)](
                x_sorted, w1, w2, w3, sorted_weights, expert_offsets,
                dy_sorted,
                dw1, dw3, dw2,
                T, D, I, E,
                x_sorted.stride(0), x_sorted.stride(1),
                w1.stride(0), w1.stride(1), w1.stride(2),
                w3.stride(0), w3.stride(1), w3.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                dw1.stride(0), dw1.stride(1), dw1.stride(2),
                dw3.stride(0), dw3.stride(1), dw3.stride(2),
                dw2.stride(0), dw2.stride(1), dw2.stride(2),
                dy_sorted.stride(0), dy_sorted.stride(1),
                sorted_weights.stride(0),
                BLOCK_T=BLOCK_T,
                BLOCK_D=BLOCK_D,
                BLOCK_I=BLOCK_I,
                num_warps=4,
                num_stages=2,
            )
            return (
                dx,
                dw1.to(w1.dtype),
                dw2.to(w2.dtype),
                dw3.to(w3.dtype),
                sorted_weights,
                None,
            )


def triton_grouped_moe_dispatch(
    x_sorted: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    sorted_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Fused grouped-GEMM SwiGLU forward. Returns y_sorted (T, D).

    Raises:
        ImportError: triton not installed.
        ValueError:  I or D exceeds 256.
    """
    if not HAS_TRITON:
        raise ImportError(
            "triton_grouped_moe_dispatch requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, use `moe_dispatch='stacked'` in your config."
        )
    return _TritonGroupedMoeFunction.apply(
        x_sorted, w1, w2, w3, sorted_weights, expert_offsets
    )

# MTP — Multi-Token Prediction + Speculative Decoding

Sources: `models/mtp.py`, `inference/speculative.py`.

## MTPBlock

A single MTP block — independent RMSNorm on the previous hidden state and
the target embedding, fusion via a `Linear(2·dim → dim)`, causal
self-attention (`nn.MultiheadAttention`, SDPA under the hood — **not**
the MLA module; it has its own `_causal_mask` buffer and no KV cache),
then a SwiGLU FFN. Residual connections around both attn and FFN.

## MTPModule

One prediction head for depth `d`. Holds a `MTPBlock` and a final
`RMSNorm`. The output head is **shared** with the main model via
`set_output_head(main_model.head)` — `test_shared_head_mtp` enforces this.

## MultiTokenPrediction

Wraps a `Transformer` with `depth` MTP heads (canonical `depth=1`):

- Registers `embed = main_model.embed` (parameter sharing —
  `test_registered_embed`).
- Shares `main_model.head` across all MTP modules.
- `forward(tokens)` returns `(main_logits, mtp_pairs)` where each pair is
  `(logits, targets)` already length-aligned. For depth `d`, the usable
  window is `seq_len - d - 2`; shorter sequences skip the depth
  (`test_forward_short_sequence`).
- `compute_loss` returns
  `(main + mtp_weight * mean_depth_mtp_loss, main_loss, mtp_loss)`.
  Canonical `mtp_loss_weight=0.3`. Only `loss` is divided by
  `gradient_accumulation_steps` in `train_step` (see `CONTEXT.md` §9).
- Empty `mtp_pairs` (sequence too short) returns `main_loss` for all
  three slots.

## Speculative decoder (`inference/speculative.py`)

`SpeculativeDecoder(main_model, mtp_module, acceptance_threshold=0.8)`:

1. **Prefill** the main model's KV cache with the prompt.
2. Each `generate_step`:
   - Main model decodes `t1` from `last_token` (greedy argmax).
   - `forward_with_hidden(..., use_cache=True)` produces the hidden state
     for `t1`; the MTP head takes `(hidden_last, embed(t1))` and drafts
     `t2` (greedy argmax).
   - Acceptance ratio = `min(1, p_main(t2) / p_draft(t2))` (greedy
     argmax comparison, not weighted rejection sampling). If
     `ratio >= threshold`, accept the draft and emit both `t1` and `t2`;
     otherwise emit only `t1`.
3. The decoder **reuses the main model's KV cache** — there is no
   separate MTP cache. `generate()` calls `main_model.reset_cache()`
   before starting (`test_generate_cache_reset`).

Measured acceptance ≈ 0.8 on smoke tests; expected ~1.5–2× throughput at
`draft_depth=2`. Acceptance is prompt-dependent — measure per-batch on a
held-out set, not a single prompt (see `SKILLS.md`).
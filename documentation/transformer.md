# Transformer — top-level wiring

Source: `models/transformer.py`.

## Layout (422M canonical, 18 layers)

| Layers | Block | FFN |
|--------|-------|-----|
| 0–1 (2 dense) | `TransformerBlock` with `SwiGLUFFN` | `inter_dim=1536` |
| 2–17 (16 MoE) | `TransformerBlock` with `DeepSeekMoE` | `moe_inter_dim=384`, 20 routed (top-4) + 1 shared |

The split is driven by `n_dense_layers`: `TransformerBlock` uses
`SwiGLUFFN` when `layer_id < n_dense_layers`, else `DeepSeekMoE`. With
`n_dense_layers=2`, layers 0–1 are dense and 2–17 are MoE.

## Components

- **`ParallelEmbedding`** — vocab embedding for single-GPU. Initialized
  with `std=0.006`.
- **`SwiGLUFFN`** — `w2(silu(w1(x)) * w3(x))`, used by dense layers and
  by each MoE expert.
- **`TransformerBlock`** — pre-norm MLA + pre-norm FFN. Residuals around
  both.
- **`Transformer`** — the full stack.

## Config shape

`Transformer.__init__` accepts a flat config dict **or** a nested
`{"model": {...}}` dict (unwrapped via `config.get("model", config)`).
This dual shape is load-bearing: `Pretrainer` may pass either; tests
exercise both via `test_construction_nested_config`.

## Weight tying

`weight_tying: true` sets `self.head.weight = self.embed.weight` (storage
shared). `count_parameters` deduplicates by tensor `id()` so the tied
weight is counted once. Removing tying breaks generation quality
(`SKILLS.md`).

## Causal mask cache

`_build_causal_mask(seqlen, device)` caches an additive `(1,1,S,S)` mask
keyed by `(seqlen, device)` in `_mask_cache`. The mask is **skipped**
when `seqlen == 1` (single-token decode) — load-bearing for decode speed.

## Forward contracts

- `forward(tokens, start_pos=0, use_cache=True) → (B, S, V)`. With
  `use_cache=True`, caches for all positions in
  `[start_pos : start_pos+seqlen]`.
- `forward_with_hidden(tokens, start_pos=0, use_cache=False) → (logits, h_norm)`.
  Used by `MultiTokenPrediction` to feed MTP heads. In
  `SpeculativeDecoder.generate_step` it is called with
  `use_cache=True` — the cache grows during drafting.
- `reset_cache()` — clears KV caches in all MLA layers. Called by
  `generate()` before each generation.
- `moe_layers()` — generator yielding only the `DeepSeekMoE` FFN modules
  (used by `Pretrainer` for bias updates and balance metrics).

## Generation

`generate(input_ids, max_new_tokens, temperature, top_p, top_k, eos_token_id)`:

- `@torch.inference_mode()`. Saves and restores `self.training`.
- Prefill the full prompt, then decode one token at a time using the KV
  cache (`start_pos = prompt_len + step`).
- Stops on EOS (if any sampled) or when `output.size(1) >= max_seq_len`.
- `_sample` — temperature, then top-k, then top-p (nucleus). Temperature
  0 → argmax (greedy, deterministic).

## `count_parameters`

`(total, trainable)` deduplicated by `id(parameter)` so shared/tied
weights count once. `test_count_with_weight_tying` and
`test_count_with_mtp` enforce the dedup behaviour.
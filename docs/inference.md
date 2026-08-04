# DeepSeek-v3-Lite — Inference & Serving

> **Read this if** you're debugging generation, KV cache, or speculative decode. **Skip if** you're training only → [Training](training.md).

**Depends on:** [Foundations & Architecture](concepts/foundations.md), [MLA & Mixed Precision](concepts/attention-and-precision.md), [DeepSeekMoE & MTP](concepts/moe-mtp.md)

---

## Table of Contents

1. [Abstract](#abstract)
2. [Complexity Analysis](#complexity-analysis)
3. [Training vs Inference](#training-vs-inference)
4. [KV Cache — Prefill and Decode](#kv-cache--prefill-and-decode)
5. [Flash Decoding — Split-K for Long Contexts](#flash-decoding--split-k-for-long-contexts)
6. [Standard Generation — `generate()` Walkthrough](#standard-generation--generate-walkthrough)
7. [Sampling Theory — Temperature, Top-K, Top-P](#sampling-theory--temperature-top-k-top-p)
8. [Greedy vs Sampled Generation — Tradeoffs](#greedy-vs-sampled-generation--tradeoffs)
9. [Interactive REPL and CLI](#interactive-repl-and-cli)
10. [Checkpoint Loading](#checkpoint-loading)
11. [Speculative Decoding](#speculative-decoding)
12. [Programmatic API](#programmatic-api)
13. [Inference Memory Budget](#inference-memory-budget)
14. [Known Limitations](#known-limitations)
15. [Debugging Checklist](#debugging-checklist)
16. [Appendix A — Decode timeline](#appendix-a--decode-timeline)
17. [Appendix B — FAQ](#appendix-b--faq)
18. [Appendix C — Glossary](#appendix-c--glossary)
19. [Worked Example — Prefill + 3 Decode Steps](#worked-example--prefill--3-decode-steps)
20. [Comparison — Standard vs Speculative](#comparison--standard-vs-speculative)
21. [Integration with Chat Template](#integration-with-chat-template)
22. [Appendix D — Extended Walkthroughs](#appendix-d--extended-walkthroughs)
23. [Check Your Understanding](#check-your-understanding)
24. [References](#references)

---

## Abstract

Inference in DeepSeek-V3-Lite is **single-GPU autoregressive generation** with MLA KV-cache acceleration. Two paths exist:

1. **Standard decode** — `models/transformer.py:Transformer.generate` with temperature / top-k / top-p sampling
2. **Speculative decode** — `inference/speculative.py:SpeculativeDecoder` using the MTP draft head for an estimated ~1.7–1.9× throughput

Entry point:

```bash
python -m inference.generate --config configs/pretrain_a100_422m.yaml --checkpoint checkpoints/pretrain_a100
```

### 60-Second Summary

Generation is the *inverse* of training. Training feeds the model every prefix of a sequence and back-propagates a next-token loss; inference feeds the model a single prefix and asks it to *produce* the continuation, one token at a time, feeding each new token back in. Two techniques make this tractable: the **KV cache**, which never recomputes past keys and values, and **sampling**, which converts a logit vector into an actual token without falling into the degenerate loops of pure argmax. This chapter covers the sampling math from scratch (temperature, top-k, top-p), the full decode loop with its positional invariants, the cache lifecycle, and the speculative-decoding path that drafts a second token per step from the MTP head. Every code claim below is anchored to a real symbol in the source.

### Inference as conditional generation

Training learns $p_\theta(x_t \mid x_{<t})$ for all positions in parallel. Inference **samples** one token at a time:

$$
x_{t+1} \sim p_\theta(\cdot \mid x_1, \ldots, x_t)
$$

**Without KV cache:** $O(T^2)$ per sequence. **With MLA cache:** amortised $O(T)$ per generated token after prefill.

**Prerequisites:** [foundations](concepts/foundations.md) §11, [MLA](concepts/attention-and-precision.md) §KV Cache Management.

---

## Complexity Analysis

### Training forward (no cache)

For sequence length $S$, hidden dim $d$, layers $L$:

| Component | Time | Space |
|---|---|---|
| Attention per layer | $O(S^2 \cdot d)$ | $O(S^2 \cdot H)$ maps (SDPA avoids materialising) |
| Dense FFN per layer | $O(S \cdot d \cdot I)$ | $O(S \cdot d)$ |
| MoE FFN per layer | $O(S \cdot k \cdot d \cdot I_{\text{moe}})$ | $O(S \cdot d)$ + routing metadata |

Total per forward: $O(L \cdot S^2 \cdot d)$ when $S$ is large.

### Inference with KV cache

After prefill, each decode step processes $S_{\text{query}} = 1$:

| Component | Per decode step |
|---|---|
| Attention | $O(S_{\text{total}} \cdot d)$ — one query attends all cached keys |
| FFN / MoE | $O(d \cdot I_{\text{moe}})$ — constant in sequence length |

**Total generation cost:**

$$
O(S_{\text{prompt}}^2) + T_{\text{gen}} \cdot O(S_{\text{prompt}} + T_{\text{gen}})
$$

Linear in output length after prefill.

### Why the quadratic term survives

The $O(S_{\text{prompt}}^2)$ is the prefill: one forward over the whole prompt computes attention for all positions at once — parallel, but quadratic in the prompt length. The decode phase is where the cache pays off: each step attends $S_q = 1$ query against $S_{\text{prompt}} + \text{step}$ cached keys, so the per-step cost grows *linearly* with context length and the total is $O(T_{\text{gen}} \cdot (S_{\text{prompt}} + T_{\text{gen}}))$. Everything after the first sampled token is memory-bandwidth-bound (reading the cache), not compute-bound — this is exactly the regime where MLA's compressed cache and flash-style kernels matter (see [Flash Decoding](#flash-decoding--split-k-for-long-contexts)).

### MLA cache compression

Per layer, per token cached bytes (BF16):

$$
\text{bytes} = 2 \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}) = 2 \times (192 + 24) = 432 \text{ bytes}
$$

At $S = 2048$, $L = 18$: $432 \times 2048 \times 18 \approx 15.9$ MB — vs $\sim 113$ MB for standard MHA at the same dims (K+V at $2 \times 12 \times 64$ floats/token/layer × 2 bytes), a ~7× reduction.

---

## Training vs Inference

| Aspect | Training | Inference |
|---|---|---|
| `use_cache` | `False` | `True` |
| `torch.no_grad` | No (grad enabled) | Yes (`@inference_mode`) |
| Batch | $8 \times 2048$ | $1 \times (\text{prompt} + 1)$ |
| MTP | Loss auxiliary | Draft head only |
| MoE bias | Updated each step | Frozen (buffer persists) |
| Compile | `torch.compile` on | Optional (usually eager) |

The MoE row deserves one clarification: at inference the gate **biases** (a `register_buffer`ed vector on `models/moe.py:AuxLossFreeGate`, not parameters) are loaded from the checkpoint and never updated — the aux-loss-free load-balancing controller (`models/moe.py:AuxLossFreeGate.update_bias`) from [Training](training.md) only runs inside `train_step`. Frozen biases are exactly right for serving: routing should be deterministic per prompt.

---

## KV Cache — Prefill and Decode

MLA caches two tensors per layer (`models/mla.py:MultiHeadLatentAttention.forward`):

```
kv_cache: (B, max_seq_len, kv_lora_rank=192)
pe_cache: (B, max_seq_len, qk_rope_head_dim=24)
```

### Prefill (prompt processing)

```
forward(prompt_tokens, start_pos=0, use_cache=True)
  → writes cache[0:prompt_len]
  → logits for all prompt positions (only last used for sampling)
```

### Decode (one token at a time)

```
forward([new_token], start_pos=current_pos, use_cache=True)
  → reads cache[0:current_pos]
  → writes cache[current_pos]
  → logits shape (B, 1, V)
```

**Mask optimisation:** When `seqlen == 1`, causal mask is `None` — the single query cannot attend to future positions that do not exist in the cache slice. The branch lives in `models/transformer.py:Transformer.forward`:

```python
if seqlen > 1:
    kv_len = end_pos if use_cache else seqlen
    mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)
else:
    mask = None
```

Note the two `use_cache`-conditioned details, both load-bearing: `kv_len` is `end_pos` (the full cached context) when a cache spans the past, and `start_pos` offsets the query positions so a *cached mid-sequence prefill* (chunked prefill, `start_pos > 0, seqlen > 1`) cannot attend to its own future. The mask is cached by `(seqlen, kv_len, start_pos, device)` inside `models/transformer.py:Transformer._build_causal_mask`, so the 2048-length prefill mask is built once per run.

**Why prefill is fast:** One forward over `prompt_len` positions amortises attention. Each decode step computes one new query against all cached keys.

### Cache lifecycle — allocation, growth, reset

The cache is **lazy and length-fixed**: it is allocated on the first `use_cache=True` forward and never grows along the sequence dimension (it is pre-allocated to `max_seq_len`). `models/mla.py:MultiHeadLatentAttention._ensure_cache` is the allocator:

```python
def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
    need_alloc = self.kv_cache is None or bsz > self._cache_batch or self.kv_cache.device != device or self.kv_cache.dtype != dtype
    if not need_alloc:
        return
    new_bsz = max(bsz, self._cache_batch * 2, 16)
    self.kv_cache = torch.zeros(new_bsz, self.max_seq_len, self.kv_lora_rank, device=device, dtype=dtype)
    self.pe_cache = torch.zeros(new_bsz, self.max_seq_len, self.qk_rope_head_dim, device=device, dtype=dtype)
    self._cache_batch = new_bsz
```

Three properties to internalise:

1. **Batch growth doubles, length is fixed.** `new_bsz = max(bsz, _cache_batch * 2, 16)` — the first allocation pads to at least 16 rows, and any later request for a *larger* batch reallocates at double the previous capacity (amortised $O(1)$ reallocation cost per batch-size increase). The sequence dimension is always `max_seq_len`; the zeros are why the decode loop can write at any `start_pos` and read a `[0:end_pos]` prefix without reallocating.
2. **Reallocation is positional state loss.** The trigger is `bsz > self._cache_batch` **or** a device/dtype change. If you prefill with batch 1 and then call with batch 8, `_ensure_cache` reallocates — the old contents are gone. Call `reset_cache()` (or restart generation) when the batch shape changes.
3. **`reset_cache` is a full teardown.** `models/mla.py:MultiHeadLatentAttention.reset_cache` sets both tensors to `None` and `_cache_batch` back to 0, so the *next* forward reallocates from scratch. `models/transformer.py:Transformer.reset_cache` fans this out across all 18 layers. There is no partial clear: generation either starts clean or inherits whatever prefix the previous call left behind.

The cache is a `torch.zeros` allocation, not a growing list, so an over-long sequence must be caught *before* the write — and it is, in the same `forward` that writes:

```python
end_pos = start_pos + seqlen
if end_pos > self.max_seq_len:
    raise RuntimeError(f"Layer {self.layer_idx}: end_pos {end_pos} exceeds max_seq_len {self.max_seq_len}")
```

This guard, plus the `output.size(1) >= self.max_seq_len` break inside `generate` (see below), is why an over-long prompt fails loudly instead of silently corrupting memory. Relatedly, the RoPE frequency table is extended on demand in doubling steps up to `max_seq_len` by `models/mla.py:MultiHeadLatentAttention._extend_rope` — decode beyond the prefill length reuses the same table (cached `freqs_cis` rows, sliced by absolute position), so there is no per-step table rebuild.

### Write and read positions

Inside `models/mla.py:MultiHeadLatentAttention.forward`, the cache write is a positional slice:

```python
if use_cache:
    self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
    ctx_kv = self.kv_cache[:bsz, :end_pos]
    ctx_pe = self.pe_cache[:bsz, :end_pos]
else:
    ctx_kv = kv_normed
    ctx_pe = k_pe
```

- **Write** `[start_pos:end_pos]`, **read** `[:end_pos]` — the attended context is exactly "everything up to and including this step", which is what makes `mask=None` safe at `seqlen == 1`.
- `.detach()` matters: the cached latents are *values*, not graph nodes. At inference mode nothing needs gradients anyway, but the detach keeps the cache usable even if a forward happens under grad-enabled context.
- `ctx_kv` is the compressed latent $(B, \text{end\_pos}, 192)$; `ctx_pe` is the RoPE key $(B, \text{end\_pos}, 24)$. Both feed the attention core (absorbed or SDPA) — see [MLA](concepts/attention-and-precision.md) for what happens next.

**Pitfall — stale context:** because the cache is positional and never cleared by a forward, any call after a previous cached call inherits the old prefix. `Transformer.generate` always starts with `self.reset_cache()`, but a bare `model(tokens, use_cache=True)` call in a notebook does not. Between independent prompts, call `model.reset_cache()`.

---

## Flash Decoding — Split-K for Long Contexts

**60-second summary.** In the decode phase each step has exactly one query token, and that query must attend to the *entire* cached context. Flash decoding is a kernel technique that splits the attention reduction over the KV dimension across many parallel programs (split-K), then combines their partial results — turning a single long serial reduction into many short parallel ones. It matters exactly when `S_kv` grows large.

**Why it exists.** Prefill is compute-bound and parallel over queries; decode is bandwidth-bound and has one query per step. A single attention kernel program per query must stream the whole cache and accumulate the softmax-weighted sum *serially* over `S_kv / BLOCK_N` tiles. On a modern GPU with ~100+ streaming multiprocessors (SMs) that is terrible utilisation: at `S_q = 1` there is no query-level parallelism to fill the machine. The GPU sits idle while one program per layer walks the cache.

**Intuition.** A dot product is a sum; sums split. Attention output for one query is a weighted average of values:

$$
o = \sum_{t=1}^{S_{kv}} a_t\, v_t, \qquad a_t = \frac{\exp(q \cdot k_t / \sqrt{d})}{\sum_{t'} \exp(q \cdot k_{t'} / \sqrt{d})}
$$

Split-K partitions the key/value rows $\{1 \dots S_{kv}\}$ into $K$ contiguous chunks and lets $K$ programs each compute a *partial* softmax: their own max $m^{(c)}$, their own normalization $l^{(c)} = \sum \exp(\cdot - m^{(c)})$, and their own weighted output sum $o^{(c)}$. A cheap second pass merges them with the online-softmax rescaling:

$$
m = \max_c m^{(c)}, \qquad l = \sum_c l^{(c)}\, e^{m^{(c)} - m}, \qquad o = \frac{1}{l} \sum_c o^{(c)}\, e^{m^{(c)} - m}
$$

Because each chunk's stats are self-contained (that is the flash-attention trick — see [Triton Kernels](concepts/kernels-and-ops.md) for the same math in the MLA kernel), the merge is exact: no approximation is introduced by splitting. The speedup is bounded by $\min(\text{chunks}, \text{parallel units available})$, so it is a latency win for long contexts, not a FLOP reduction.

**How this repo's MLA kernel maps onto it.** The fused kernel behind `models/mla_triton.py:triton_mla_attention` already tiles both query and KV dimensions (`BLOCK_Q=64`, `BLOCK_N=64`) and keeps the register-bounded dims ($R \le 256$) in registers; its per-query reduction over KV is a serial loop over `BLOCK_N` tiles — the "flash" form with a single split. At the canonical `max_seq_len = 2048` and `BLOCK_N = 64`, that is 32 tiles per query: entirely reasonable for one program, and the kernel is already an order of magnitude faster than materialising full K/V (see Appendix D). Split-K — many programs per query, each owning a slice of the 32 tiles, plus a combine kernel — is the technique you reach for when context grows to tens of thousands of tokens, where the serial tile loop starts to dominate the step time. `[INFERENCE]` No long-context GPU benchmark has been run in this repo (no GPU run exists at all); the split-K discussion is the standard argument from the flash-decoding literature (Dao et al., 2023), and the repo's kernel structure (tile loop + online softmax) is exactly the shape that would accept a split-K transform.

**Pitfall.** Do not confuse split-K (parallelism *within one query's* reduction) with tiling (the `BLOCK_N` loop itself). Both appear in flash-attention-family kernels; only split-K trades extra programs for a shorter serial chain, and it only pays off when `S_kv / BLOCK_N` is large enough that the reduction is the bottleneck — at `S_kv = 2048` the overhead of the combine pass can exceed the savings. The repo's `max_seq_len=2048` config does not need it.

---

## Standard Generation — `generate()` Walkthrough

**60-second summary.** `models/transformer.py:Transformer.generate` is the entire inference path in one function: reset the cache, prefill the prompt, then loop — sample one token, append it, check stopping conditions, and forward only the new token into the cache. It is annotated `@torch.inference_mode()`, restores the caller's train/eval state on exit, and guarantees the cache write position never exceeds `max_seq_len`.

**Why it exists.** Autoregressive generation is a loop with a *lot* of invariants to get right: cache write positions, EOS bookkeeping per batch row, the `max_seq_len` bound, and the training-mode contract. Centralising it in one method — rather than in each caller — is what keeps `generate_interactive` and the tests honest.

**Intuition.** Think of the cache as a tape and `start_pos` as the write head. Prefill lays down the whole prompt at positions `0 … prompt_len-1`. Each loop iteration: the head is at `prompt_len + step`, we sample the next token from the *last* position's logits, then advance the head by forwarding that single token. The tape is pre-allocated to `max_seq_len`, so "stop" is simply "don't let the head run past the end of the tape."

### The code

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             top_p: float = 0.9, top_k: int = 0, eos_token_id: Optional[int] = None) -> torch.Tensor:
    """Autoregressive generation with KV-cache, top-p and top-k sampling."""
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    was_training = self.training
    self.reset_cache()
    self.eval()
    bsz, prompt_len = input_ids.shape
    output = input_ids.clone()
    prefill_logits = self.forward(output, start_pos=0, use_cache=True)
    next_logits = prefill_logits[:, -1, :]
    finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
    for step in range(max_new_tokens):
        next_token = self._sample(next_logits, temperature, top_p, top_k)
        output = torch.cat([output, next_token], dim=1)
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        if output.size(1) >= self.max_seq_len:
            break
        decode_logits = self.forward(next_token, start_pos=prompt_len + step, use_cache=True)
        next_logits = decode_logits[:, -1, :]
    if was_training:
        self.train()
    return output
```

### Line-by-line

| Step | Code | Notes |
|---|---|---|
| 1 | `if temperature < 0.0: raise ValueError` | Negative temperature is a caller bug; fail before touching the cache. Guarded by `tests/test_models.py:TestGeneration.test_generate_negative_temperature_raises`. |
| 2 | `was_training = self.training; self.reset_cache(); self.eval()` | Capture state, clear all 18 layer caches, disable dropout (there is none in this repo, but `eval()` keeps the contract honest). |
| 3 | `output = input_ids.clone()` | The returned tensor is the prompt **plus** the generated continuation — callers slice `output[:, prompt_len:]` to separate them. |
| 4 | `prefill_logits = self.forward(output, start_pos=0, use_cache=True)` | One parallel forward over the whole prompt; writes cache positions `0 … prompt_len-1`. |
| 5 | `next_logits = prefill_logits[:, -1, :]` | Only the last position is sampled — earlier logits are discarded (they were only needed to *fill* the cache). |
| 6 | Loop `max_new_tokens` times | See below. |
| 7 | `if was_training: self.train()` | Restores the caller's mode on normal exit. |

### The loop invariants

These four properties are what make the loop correct; each is worth stating precisely because each has a classic failure mode.

1. **Cache write position invariant.** At loop iteration `step` (0-indexed), the forward call is `self.forward(next_token, start_pos=prompt_len + step, use_cache=True)`, writing cache position `prompt_len + step` and reading `[0 : prompt_len + step + 1]`. The prefill wrote positions `[0, prompt_len)`, so the tape fills contiguously. Off-by-one here (e.g. `start_pos=step` or `start_pos=prompt_len + step + 1`) corrupts the cache: a wrong write slot plus the `[:end_pos]` read means the next step attends to garbage or, worse, a *future* token's key/value. This is the single most common generation bug, and it is silent — no shape error, just gibberish after the first token.

2. **Output length invariant.** At the top of iteration `step`, `output.size(1) == prompt_len + step`; sampling appends the token at index `prompt_len + step`, so `output.size(1) == prompt_len + step + 1` afterwards. The `max_seq_len` check compares against this running total, not against `max_new_tokens`.

3. **The `max_seq_len` bound.** `if output.size(1) >= self.max_seq_len: break` runs *before* the next forward. Since the forward's `end_pos = start_pos + 1 = prompt_len + step + 1 = output.size(1)` after the append, this guarantees `end_pos <= max_seq_len` — the MLA `RuntimeError` guard is never triggered by a well-formed call. Without this break, generation would raise `Layer N: end_pos … exceeds max_seq_len` at the first over-long step; the break converts that crash into clean early termination. Guarded by `tests/test_models.py:TestGeneration.test_generate_respects_max_seq_len`.

4. **EOS semantics per batch row.** `finished` is a per-row boolean mask; a row is marked done the step it samples `eos_token_id`, and the loop stops only when `finished.all()`. Two consequences worth knowing: (a) with `B > 1`, already-finished rows **keep generating** (their logits are still sampled and appended) until every row finishes — there is no per-row masking of `next_logits`; (b) with `eos_token_id=None` (the default), generation never stops early and runs the full `max_new_tokens` loop. Guarded by `tests/test_models.py:TestGeneration.test_generate_eos_termination`.

5. **Training-mode restore.** `was_training` is captured before `self.eval()`. On normal exit, `self.train()` restores the original mode, so calling `generate` inside a training pipeline (e.g. for a scheduled eval) does not silently leave the model in eval mode — guarded by `tests/test_models.py:TestGeneration.test_generate_restores_train_mode`. Note the restore is *not* in a `try/finally`: if `_sample` or a forward raises, the model stays in eval mode until the caller handles it.

6. **Cache isolation between calls.** Because `reset_cache()` runs first, two back-to-back `generate` calls share no state — the second call re-prefills from scratch. Guarded by `tests/test_models.py:TestGeneration.test_generate_kv_cache_isolation`.

### `forward_with_hidden` (MTP / speculative only)

Returns `(logits, h)` where `h` is the **pre-final-norm** trunk hidden state — `models/transformer.py:Transformer.forward_with_hidden`. It is identical to `forward` except it also hands the raw residual-stream output to the caller, which is exactly what the MTP block conditions on (MTP applies its own norms, so it needs the *pre*-norm state; see [MTP](concepts/moe-mtp.md) §8). Used by `inference/speculative.py:SpeculativeDecoder.generate_step` to feed the draft head after the main model commits token $t_1$; `use_cache=False` in the training wrapper (`MultiTokenPrediction`) so the draft path never pollutes a cache during training.

---

## Sampling Theory — Temperature, Top-K, Top-P

**60-second summary.** The model's final layer produces a logit $z_i$ per vocabulary item; a softmax turns that into a distribution $p_i$, and sampling draws a token from it. Three knobs modify that distribution before the draw: **temperature** sharpens or flattens it, **top-k** hard-truncates to the $k$ most-likely tokens, and **top-p** (nucleus) keeps the *smallest* set whose cumulative probability reaches $p$. The implementation lives in `models/transformer.py:Transformer._sample`, a static method with an execution order of temperature → top-k → softmax → top-p → multinomial.

### Why sample at all?

Argmax ("greedy") decoding is a perfectly valid sampler — it is the mode of the distribution — but pure greedy generation degenerates: a single misstep cannot be recovered from, and long continuations collapse into repetitive loops ("I am a helpful assistant" × 50). Sampling draws from the whole distribution, so the model's *uncertainty* becomes output diversity, and the temperature/top-k/top-p knobs let you trade diversity against quality per use case ([Greedy vs Sampled](#greedy-vs-sampled-generation--tradeoffs)).

### Temperature — the semantics of $\tau$

Start from the logits $z \in \mathbb{R}^V$. The softmax defines the model's native distribution $p$:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Temperature $\tau$ rescales the logits *before* the softmax:

$$
p_i^{(\tau)} = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
$$

Why does this work? Write the ratio of two probabilities:

$$
\frac{p_i^{(\tau)}}{p_j^{(\tau)}} = e^{(z_i - z_j)/\tau}
$$

For $\tau > 1$ the exponent shrinks, so all ratios pull toward 1 — the distribution flattens, entropy rises, and lower-probability tokens get drawn more often. For $0 < \tau < 1$ the exponent grows, so the gaps widen — the distribution sharpens toward the mode. The limiting cases:

| $\tau$ | Behaviour |
|---|---|
| $\to 0^+$ | Greedy argmax (deterministic) |
| $1.0$ | Native model distribution |
| $> 1$ | Flatter (higher entropy) |
| $\to \infty$ | Uniform over the vocabulary |

**Argmax as temperature 0, precisely.** For $\tau \to 0^+$, let $z_{\max} = \max_j z_j$. Factor out the max:

$$
p_i^{(\tau)} = \frac{e^{(z_i - z_{\max})/\tau}}{\sum_j e^{(z_j - z_{\max})/\tau}}
$$

Every non-maximal term $e^{(z_i - z_{\max})/\tau} \to 0$ as $\tau \to 0^+$ (negative exponent over a vanishing denominator), while the maximal terms stay at $e^0 = 1$. So the mass concentrates entirely on the argmax token(s) — softmax at $\tau \to 0$ *is* argmax. The code implements this limit directly, skipping the numerical nonsense of dividing by an infinitesimal:

```python
if temperature == 0.0:
    return logits.argmax(dim=-1, keepdim=True)
```

Note the check is exact float equality: `temperature=1e-12` is *not* zero, so it takes the softmax path (mathematically fine — `torch.softmax` subtracts the max internally, so no overflow), but it is effectively greedy in practice. Ties in `argmax` resolve to the first index deterministically. Guarded by `tests/test_models.py:TestTransformerAdditional.test_sample_argmax` (temperature 0 → argmax) and `tests/test_models.py:TestGeneration.test_generate_greedy_deterministic`.

### Top-k — the hard cutoff

Keep the $k$ largest logits, mask everything else to $-\infty$ so the softmax gives them probability exactly 0:

```python
if top_k > 0:
    kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
    logits = logits.masked_fill(logits < kth_vals, float("-inf"))
```

`topk(...)[0][:, -1:]` is the $k$-th largest value per row; `masked_fill` zeroes every entry strictly below it (ties at the boundary survive). `top_k=0` disables the branch entirely — 0 is the sentinel, not "keep 0 tokens". Because `-inf` logits produce $e^{-\infty} = 0$ under softmax, masked tokens can never be sampled, and the surviving $k$ tokens are renormalised by the softmax automatically. Note the mask is applied **after** temperature scaling in this repo — the cutoff is on the tempered logits.

### Top-p (nucleus) — the cumulative-probability algorithm

(Holtzman et al., 2020). Instead of a fixed count, keep the *smallest* set of tokens whose cumulative probability reaches $p$:

```python
sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
cumulative = sorted_probs.cumsum(dim=-1)
remove = (cumulative - sorted_probs) > top_p
sorted_probs = sorted_probs.masked_fill(remove, 0.0)
sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
```

Walk through the algorithm:

1. **Sort descending.** `sorted_probs` is the probability mass in decreasing order; `sorted_idx` remembers where each entry came from in the original vocabulary.
2. **Cumulative sum.** `cumulative[k] = Σ_{j≤k} sorted_probs[j]` — the mass of the top $k+1$ tokens.
3. **The `remove` mask — the subtle line.** `remove[k] = (cumulative[k] - sorted_probs[k]) > top_p`. The subtraction backs out the current token's own mass, so `remove[k]` is true iff the mass *before* this token already cleared the threshold. The first token (k=0) has `cumulative - sorted = 0`, so it is never removed; more generally, **the token that pushes the cumulative sum across `top_p` is kept**, and everything after it is removed. This is the "smallest set whose sum ≥ p" semantics, implemented without a loop.
4. **Zero and renormalise.** Removed tokens get probability 0; the survivors are re-scaled to sum to 1 (`clamp(min=1e-10)` guards the pathological denominator; with the rule above at least one token always survives, so the clamp is defensive). The *shape* of the survivors' relative probabilities is unchanged — this is truncation, not reshaping.
5. **Sample in sorted space, map back.** `torch.multinomial(sorted_probs, 1)` draws an index into the sorted array; `sorted_idx.gather(-1, …)` translates it back to the original vocabulary id. The gather is required — without it you would emit "the 3rd-most-likely token" instead of "the token whose original id is X".

**Why `top_p < 1.0` gates the branch.** `top_p=1.0` means "keep everything" — the branch is skipped and `torch.multinomial(probs, 1)` samples the full distribution. The default in `generate` is `top_p=0.9`, so **the default is not greedy**: every call samples from the 90%-mass nucleus. To disable top-p entirely, pass `top_p=1.0`.

**Combined top-k + top-p rationale:** Top-k alone uses a fixed cutoff that may include irrelevant tokens if the distribution is flat, or cut the mass too aggressively if it is peaked. Top-p alone can produce a very large nucleus when many tokens share similar probability (a flat tail). Combined — top-k first as a hard ceiling, then top-p to adapt *within* that ceiling — gives the best of both. `tests/test_models.py:TestGeneration.test_sample_top_k` and `test_sample_top_p` guard the two branches.

### The whole `_sample`, annotated

```python
@staticmethod
def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """Temperature + top-k + top-p sampling. Temperature==0 -> argmax."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = (cumulative - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
    else:
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

It is a `@staticmethod` — it never touches model state, only the logits tensor — which is why `inference/speculative.py:SpeculativeDecoder.generate_step` can call it for the main-token draw (`Transformer._sample(..., temperature, top_p=1.0, top_k=0)`) without a model instance. `_sample` returns shape `(B, 1)` (keepdim on the argmax; `num_samples=1` on the multinomial), which is exactly what `generate` concatenates.

**Execution order:** temperature scaling → top-k mask → softmax → top-p filter → multinomial. Two orderings are load-bearing: temperature acts on raw logits (scaling before softmax is not equivalent to scaling after), and top-k's `-inf` mask must precede the softmax (a post-softmax mask would need renormalisation).

**Pitfalls.**

- **`temperature == 0.0` is a literal equality.** `1e-9` and `-0.0` behave differently than `0.0` (`-0.0 == 0.0` is true in IEEE, so `-0.0` hits the argmax branch; a *negative* temperature does not — `generate` rejects it before `_sample` runs).
- **`top_p` and `top_k` defaults interact with temperature.** The CLI default `--temperature 0.7 --top_p 0.9` is a *sampled* decode; only `--temperature 0` is greedy.
- **`float("-inf")` logits under softmax are exactly 0 probability** — never sampled, and `torch.multinomial` with all-zero rows would error; the top-k branch always leaves ≥ `min(top_k, V) ≥ 1` entries, so this cannot happen.
- **Nucleus emptiness.** The `cumulative - sorted_probs` construction guarantees the crossing token survives, so the renormalisation denominator is always positive; `clamp(min=1e-10)` is belt-and-braces, not the load-bearing logic.

---

## Greedy vs Sampled Generation — Tradeoffs

**60-second summary.** Greedy decoding (argmax, `temperature=0`) is deterministic, cheap, and stable — the right choice for benchmarks, code, and anything where a wrong token is expensive. Sampled decoding (`temperature>0`, optionally top-k/top-p) trades a little expected quality for diversity and naturalness — the right choice for dialogue and open-ended text.

**Why the tradeoff exists.** The argmax sequence is the *mode* of the joint distribution, but the mode of a long sequence is not the sequence of per-position modes: greedy commits at every step to the single most-likely token, and one committed error can steer the whole continuation into a low-probability region. Sampling keeps probability mass on alternatives, so the output explores the distribution — at the cost of occasionally drawing a token that greedy would have rejected.

| Property | Greedy (`τ = 0`) | Sampled (`τ > 0`) |
|---|---|---|
| Output | Deterministic (same seed-independent result) | Stochastic (varies per run) |
| Per-step cost | Same (argmax is cheaper than multinomial, but negligible) | Same + sampling overhead |
| Diversity | None — repeats and loops on long generations | Depends on τ and the nucleus |
| Factual/code tasks | Preferred (low tolerance for wrong tokens) | Risky unless τ ≈ 0 |
| Dialogue/creative | Degenerate loops | Preferred |
| Reproducibility | Trivial | Needs a manual seed to reproduce |

**Practical knob map for this repo.** `generate(temperature=..., top_p=..., top_k=...)`:

- **`temperature=0`** — greedy; the only fully deterministic mode (`test_generate_greedy_deterministic`).
- **`0.3–0.7`** — mild sharpening: keeps the mode dominant but allows occasional exploration. A common dialogue range (the CLI defaults to 0.7).
- **`1.0` + `top_p=0.9`** — the model's native distribution, truncated to the 90% nucleus; the library default.
- **`τ > 1`** — deliberate flattening; useful for data augmentation, rarely for user-facing text.
- **Top-k as a cap** — with `top_k=50, top_p=0.95` you get the classic "nucleus inside a ceiling" recipe; with `top_k=1` you get argmax again (the k-th largest value is the max, so only it survives the mask).

**What this repo deliberately does not implement.** Beam search and contrastive decoding are absent; there is no `num_return_sequences` and no repetition penalty. For a single-GPU research codebase the sampling surface is intentionally minimal — the three knobs compose and are fully covered by `tests/test_models.py:TestGeneration` and `tests/test_models.py:TestTransformerAdditional`.

---

## Interactive REPL and CLI

**60-second summary.** `inference/generate.py:generate_interactive` is a chat loop: it formats the accumulated `messages` with the tokenizer's chat template, generates a completion, decodes the *new* tokens, and appends the response to the history. `inference/generate.py:main` is the bootstrap: parse args, load config, build the model, load weights, optionally wire the MTP draft head, and hand off to the loop.

### The REPL loop

```
Commands:
  /exit   — quit
  /clear  — reset conversation history (does NOT reset KV cache)

Flow:
  user input → chat template → tokenise → generate → decode → append to history
```

The loop body, from `inference/generate.py:generate_interactive`:

```python
messages.append({"role": "user", "content": user_input})
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(args.device)
if decoder is not None:
    output_ids = decoder.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, eos_token_id=eos_id)
else:
    output_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, eos_token_id=eos_id)
new_tokens = output_ids[0, input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"\nAssistant: {response}")
messages.append({"role": "assistant", "content": response})
```

Three details worth internalising:

1. **`top_p` *is* forwarded** in the standard path (`top_p=args.top_p`). (Older versions of this doc claimed otherwise; the current source passes it.) The speculative path cannot take it — `SpeculativeDecoder.generate` has no `top_p` parameter — so `--top_p` is silently ignored there.
2. **The whole history is re-prefilled every turn.** `generate` starts with `reset_cache()`, so the KV cache does *not* accumulate across turns; the multi-turn context lives in `messages`, which is re-tokenised and re-prefilled in full each turn. Consequences: `/clear` just empties `messages` (the cache is rebuilt from scratch on the next turn regardless), and REPL latency grows with the history length because every turn is a full prefill of everything said so far.
3. **The slice `output_ids[0, input_ids.shape[1]:]`** strips the prompt: the returned tensor from `generate`/`decoder.generate` includes the input, and decoding the whole thing would re-print the user's message.

`generate_interactive` is decorated `@torch.inference_mode()` and wraps the prompt loop in `try/except (EOFError, KeyboardInterrupt)` so Ctrl-C or piped stdin exits cleanly.

### CLI arguments

```bash
python -m inference.generate \
  --config configs/pretrain_a100_422m.yaml \
  --checkpoint checkpoints/pretrain_a100 \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --top_p 0.9 \
  --use_speculative \
  --acceptance_threshold 0.8 \
  --device cuda
```

| Argument | Default | Notes |
|---|---|---|
| `--config` | required | YAML with `model:` section |
| `--checkpoint` | required | Directory or `.safetensors` file |
| `--max_new_tokens` | 512 | Max tokens to generate |
| `--temperature` | 0.7 | Passed to `generate` (and to the speculative main-token draw) |
| `--top_p` | 0.9 | Forwarded in the standard path; ignored by `SpeculativeDecoder` |
| `--use_speculative` | off | Requires MTP weights in checkpoint |
| `--acceptance_threshold` | 0.8 | Draft acceptance bar |
| `--device` | auto | `cuda` if available else `cpu` |

### `main()` bootstrap sequence

`inference/generate.py:main`:

1. **Guard.** If `transformers` is not installed, raise `RuntimeError` with install instructions. The import is deliberately deferred (`HAS_TRANSFORMERS` flag) so the pure-decode helpers in the module stay importable in CPU-only test environments without `transformers`.
2. `load_config(args.config)` — `inference/generate.py:load_config` checks the file exists and that a `model:` section is present; anything else raises a descriptive error.
3. `Transformer(model_cfg).to(args.device)` then `model.eval()`.
4. **Checkpoint resolution.** If `--checkpoint` is a directory: `ckpt_mgr = CheckpointManager(ckpt_dir)` and `step = ckpt_mgr.latest_step()` (the largest *complete* step — see [Checkpoint Loading](#checkpoint-loading)); a directory with no complete checkpoint raises `RuntimeError("No checkpoints found …")`. If `--checkpoint` is a file: the step is parsed from the filename stem (`int(stem.split("_")[-1])`, e.g. `model_step_4000` → 4000), falling back to `latest_step()` if the stem is not numeric.
5. `ckpt_mgr.load(model, step, device=args.device, strict=False)` — loads weights; `strict=False` is required because of weight tying (see below).
6. **Speculative wiring (only with `--use_speculative`):** build `MTPModule(model_cfg, depth=1)`, `mtp_module.eval()`, **attach the shared head first** (`models/mtp.py:MTPModule.set_output_head(model.head)` — without this, the first draft step raises `RuntimeError("output_head not set")`), then load the `mtp.`-prefixed keys from the checkpoint (with a `[warn]` if none exist — uninitialised draft head).
7. Tokenizer: `AutoTokenizer.from_pretrained(cfg.get("data", {}).get("tokenizer_path", "deepseek-ai/deepseek-coder-v2-lite"))`.
8. `generate_interactive(model, tokenizer, args, mtp_module)`.

**`transformers` is optional** for imports — only required for CLI tokenizer. Tests import decode helpers without it.

---

## Checkpoint Loading

**60-second summary.** The checkpoint is a `.safetensors` file written by `utils/checkpoint.py:CheckpointManager.save`; `CheckpointManager.load` restores weights into an *already-constructed* model, tolerating missing keys because of weight tying. MTP weights travel in the same file under the `mtp.` prefix, and the speculative path re-shares the LM head after loading.

### The minimal pattern

```python
ckpt_mgr = CheckpointManager(ckpt_dir)
step = ckpt_mgr.latest_step()
ckpt_mgr.load(model, step, device=device, strict=False)
```

`latest_step()` (`utils/checkpoint.py:CheckpointManager.latest_step`) returns the largest step for which **all three** files exist — `model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json` (`_checkpoint_complete`) — so a torn save (crash mid-write) never loads. `load` (`utils/checkpoint.py:CheckpointManager.load`) then reads `model_step_{step}.safetensors` with `device=` placement.

**Weight tying:** with `weight_tying: true` the LM head *shares storage* with the embedding (`self.head.weight = self.embed.weight` in `models/transformer.py:Transformer.__init__`), and the saver writes the shared tensor once — the duplicate `head.weight` key is dropped (`_atomic_save_safetensors`). `load_state_dict(strict=False)` restores the surviving `embed.weight` and the tied `head.weight` comes back automatically through the shared storage. `strict=False` is therefore not optional sloppiness: with `strict=True` the missing `head.weight` key would raise. The regression surface is `tests/test_models.py:TestEmbedding.test_weight_tying_shared`.

### MTP weights (speculative mode)

```python
from safetensors.torch import load_file
state = load_file("model_step_N.safetensors", device=device)
mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
mtp_module.load_state_dict(mtp_state, strict=False)
mtp_module.set_output_head(model.head)  # re-share after load
```

**Order is load-bearing: `set_output_head` before `load_state_dict`.** The checkpoint stores the head's weights under the *main model's* `head.*` keys (the head is shared, so it is never saved under `mtp.*`) — a standalone `MTPModule` constructed with `output_head=None` would therefore fail `load_state_dict(strict=True)` (missing `output_head.weight`) and, worse, *forward* with a `None` head. `main` wires the head first, then loads the `mtp.`-prefixed keys; `strict=False` tolerates the absent `output_head` keys. The exact failure mode — and the fix — is pinned by `tests/test_inference.py:TestSpeculativeDecoder.test_draft_without_head_raises_and_attach_after_load_works` ("Regression: interactive --use_speculative crashed because the draft module's output_head was never attached"). See [MTP](concepts/moe-mtp.md) §10.5 for the same pitfall from the training side.

**Pitfall — "No MTP weights" is a warning, not an error.** If the checkpoint has no `mtp.` keys, `main` prints `[warn] No MTP weights in checkpoint; draft head is uninitialised.` and continues. The draft head then proposes near-random tokens and acceptance collapses to ~0 — the decode still runs (every step falls back to `token_main`), it just gains nothing. Check the warning text if speculative mode seems slower than standard decode.

---

## Speculative Decoding

**60-second summary.** `inference/speculative.py:SpeculativeDecoder` uses the MTP head trained with $\lambda = 0.3$ as a **draft model** — no separate checkpoint, no extra weights beyond the ~7.1M the MTP block adds. Each step: run the main model on the last token, sample `token_main`; run the trunk *again* on `token_main` to obtain the hidden state the draft conditions on; the MTP block proposes `token_draft`; a threshold test accepts or rejects it. Accepted, two tokens are emitted per main-model forward pair; rejected, only `token_main`. The theory, acceptance math, and cache-consistency analysis live in [MTP](concepts/moe-mtp.md) §11–§13; this section walks the code and the repo's deliberate simplifications.

### `generate_step` algorithm

```python
@torch.inference_mode()
def generate_step(self, last_token: torch.Tensor, start_pos: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    main_logits = self.main_model(last_token, start_pos=start_pos, use_cache=True)
    main_probs = torch.softmax(main_logits[:, -1, :], dim=-1)
    # Sample the main token with temperature (argmax when temperature == 0);
    # the draft stays greedy (its most-likely continuation is what we verify).
    token_main = Transformer._sample(main_logits[:, -1, :], temperature, top_p=1.0, top_k=0).squeeze(0)
    t1_pos = start_pos + 1
    _, hidden = self.main_model.forward_with_hidden(token_main.unsqueeze(0), start_pos=t1_pos, use_cache=True)
    hidden_last = hidden[:, -1:, :]
    token_main_emb = self.main_model.embed(token_main.unsqueeze(-1))
    draft_logits, _ = self.mtp(hidden_last, token_main_emb)
    draft_probs = torch.softmax(draft_logits[:, -1, :], dim=-1)
    token_draft = draft_probs.argmax(dim=-1)
    # Acceptance compares raw (unscaled) probabilities of the draft token.
    p_main_of_draft = main_probs[0, token_draft[0]].item()
    p_draft_of_draft = draft_probs[0, token_draft[0]].item()
    return token_main, token_draft, p_main_of_draft >= self.threshold * max(p_draft_of_draft, 1e-12)
```

Position by position:

1. `main_logits = self.main_model(last_token, start_pos=start_pos, use_cache=True)` — forward #1: the trunk reads the cache prefix and **writes** position `start_pos`.
2. `token_main = Transformer._sample(...)` — the main token is *sampled* with the caller's temperature (`top_p=1.0, top_k=0` — nucleus/ceiling are forced off in the speculative path). `temperature=0` degenerates to argmax, as usual.
3. `_, hidden = self.main_model.forward_with_hidden(token_main.unsqueeze(0), start_pos=t1_pos, use_cache=True)` — forward #2, at `t1_pos = start_pos + 1`: **this is why the main model runs twice per step.** The MTP contract is $x_{t+2} \leftarrow (h_{t+1}, e_{t+1})$ — the draft needs the *hidden state* of the token it conditions on, and the only way to get it is to run the trunk on `token_main`. The cache write at `t1_pos` is a write of the same values the next step would produce anyway — see [MTP](concepts/moe-mtp.md) §13 for the full double-write analysis.
4. `draft_logits, _ = self.mtp(hidden_last, token_main_emb)` — one MTP-block pass; `token_main_emb` comes from the *shared* embedding (`self.main_model.embed`), so the draft conditions on the same token representation the main model committed.
5. `token_draft = draft_probs.argmax(dim=-1)` — the draft is **always greedy**, regardless of temperature. Deliberate: the verifier checks the draft's single most-likely continuation; a sampled draft would make the acceptance test noisy.
6. Acceptance: `p_main_of_draft >= self.threshold * max(p_draft_of_draft, 1e-12)` — a deterministic threshold check, **not** the Metropolis–Hastings acceptance of Leviathan et al. The probabilities are the raw softmaxes of the *unscaled* logits — temperature affects what `token_main` is, but the threshold test always compares the temperature-1 distributions. The `1e-12` floor guards `0 >= τ·0` when the draft's argmax probability underflows to zero (spurious acceptance).

### Acceptance rule

This repo uses a **simplified** threshold check (not full Metropolis-Hastings from Leviathan et al., 2023):

```python
accept = p_main[token_draft] >= threshold * p_draft[token_draft]
```

Default `threshold = 0.8`.

**Expected speedup:**

$$
\mathbb{E}[\text{tokens per step}] = 1 + \mathbb{P}(\text{accept}) \approx 1 + 0.8 = 1.8
$$

**Caveats:** (1) the `≈0.8` acceptance figure came from smoke tests — this repo has **no trained checkpoint yet**, so real acceptance on a trained MTP head is unknown and prompt-dependent; treat every throughput figure as an estimate; (2) the threshold rule is a throughput heuristic, **not lossless** speculative sampling — it biases the output distribution toward draft-favoured tokens and never resamples on rejection (see [MTP](concepts/moe-mtp.md) §11.4–11.5 for the exact deviations: no rejection resampling, deterministic accept, and "greedy with draft-flavoured detours" even at temperature 0).

### `generate()` outer loop

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             eos_token_id: Optional[int] = None) -> torch.Tensor:
    output = input_ids.clone()
    n_generated = 0
    if hasattr(self.main_model, "reset_cache"):
        self.main_model.reset_cache()
    _ = self.main_model(output, start_pos=0, use_cache=True)
    while n_generated < max_new_tokens:
        start_pos = output.size(1) - 1
        last_token = output[:, -1:]
        token_main, token_draft, was_accepted = self.generate_step(last_token, start_pos=start_pos, temperature=temperature)
        output = torch.cat([output, token_main.unsqueeze(0)], dim=1)
        n_generated += 1
        if eos_token_id is not None and token_main.item() == eos_token_id:
            break
        if was_accepted and n_generated < max_new_tokens:
            output = torch.cat([output, token_draft.unsqueeze(0)], dim=1)
            n_generated += 1
            if eos_token_id is not None and token_draft.item() == eos_token_id:
                break
    return output
```

1. Prefill: `main_model(output, start_pos=0, use_cache=True)` (after `reset_cache` — `generate` owns cache hygiene; `generate_step` alone does not, see [MTP](concepts/moe-mtp.md) §13.4).
2. While `n_generated < max_new_tokens`:
   - `start_pos = output.size(1) - 1` — the position of the *last emitted* token; `generate_step` writes the new token at `start_pos + 1` (i.e. `t1_pos`), so the same invariant as standard decode holds: the cache is always one ahead of the last emitted token.
   - Append `token_main`, count it.
   - If accepted **and** the budget remains, append `token_draft` and count it — two tokens per loop iteration.
   - EOS is checked separately for each token (`token_main.item()`, then `token_draft.item()`), batch size 1 only.
3. `temperature` is used for the main-token draw (unlike older revisions of this code); the draft stays greedy and there is no top-p/top-k.

**Cache coherence:** `forward_with_hidden` at `start_pos=t1_pos` must match the position after `token_main` is committed to the cache — it does, because forward #2 happens *after* forward #1 wrote `start_pos` and the write at `t1_pos` is deterministic given the same cache prefix. The full position-by-position trace is [MTP](concepts/moe-mtp.md) §13.

**Honest compute accounting:** each accepted step costs two trunk forwards (not one), so the wall-clock win is real only when two tokens are emitted per iteration *and* the acceptance rate clears ~50% (the second forward roughly doubles the per-iteration cost). [MTP](concepts/moe-mtp.md) §11.5 works the break-even math; the repo's smoke-derived estimate of 1.7–1.9× assumes acceptance around 0.8 — unverified on real weights.

---

## Programmatic API

```python
import torch, yaml
from models.transformer import Transformer
from models.mtp import MTPModule
from inference.speculative import SpeculativeDecoder
from utils.checkpoint import CheckpointManager

cfg = yaml.safe_load(open("configs/pretrain_a100_422m.yaml"))
model = Transformer(cfg).cuda().eval()
CheckpointManager("checkpoints/pretrain_a100").load(model, step=4000, strict=False)

# Standard sampling with top-p
output = model.generate(input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9)

# Speculative
mtp = MTPModule(cfg["model"], depth=1).cuda().eval()
mtp.set_output_head(model.head)  # required — see Checkpoint Loading
# ... load mtp weights ...
decoder = SpeculativeDecoder(model, mtp, acceptance_threshold=0.8)
output = decoder.generate(input_ids, max_new_tokens=100)
```

Notes: `Transformer(cfg)` accepts both nested (`{"model": …}`) and flat configs (`models/transformer.py:Transformer.__init__`); `SpeculativeDecoder` assumes batch size 1 (`generate_step` indexes `main_probs[0, …]`); and `decoder.generate` returns the full `input_ids ++ continuation` tensor, so slice `[input_ids.shape[1]:]` for the response.

---

## Inference Memory Budget

For interactive generation ($B=1$, prompt 512, generating 256 tokens) — **all figures are estimates; no GPU run has been executed**:

| Component | Approximate |
|---|---|
| Model weights (BF16, 411.6M) | ~0.82 GB (0.84 GB with the ~7.1M MTP block) |
| KV cache at max_seq_len=2048 | ~16 MB |
| Activations (single-token decode) | negligible |
| CUDA context | 2–14 GB |

**Total:** ~1–2 GB for weights + overhead — fits on consumer GPUs for inference-only (no optimizer state; the 4.9 GB FP32 AdamW state from [Training](training.md) is absent at inference).

The cache line comes from the 432 bytes/token/layer figure in [Complexity Analysis](#complexity-analysis): $432 \times 2048 \times 18 \approx 15.9$ MB, and it is **capped** — the cache is pre-allocated to `max_seq_len` and never grows, so the worst case is known in advance. Use `utils/memory.py:estimate_model_memory_gb(model, seq_len, batch_size, inference=True)` for the analytical budget (the `inference=True` flag drops optimizer and activation bytes).

---

## Known Limitations

1. **`--top_p` applies to the standard path only** — `generate_interactive` forwards `top_p=args.top_p` to `model.generate`, but the speculative path has no `top_p` parameter (the main-token draw is `top_p=1.0, top_k=0`). Use the standard path for nucleus sampling.
2. **No batch inference** — `SpeculativeDecoder` assumes batch size 1 (indexes row 0 directly).
3. **No continuous batching** — serving-at-scale patterns not implemented; the cache is batch-static per session.
4. **Triton MoE kernel not validated for decode** — the canonical config falls back to the stacked MoE dispatch (the Triton kernel is register-capped at $I, D \le 256$, and the canonical `moe_inter_dim=384` exceeds it); use `moe_dispatch="stacked"` for inference. See [Triton Kernels](concepts/kernels-and-ops.md).
5. **`transformers` required for CLI** — optional import; tests import helpers without it.
6. **Speculative path is sampling-limited** — the main token is sampled with `temperature` (greedy at 0), but top-k/top-p are hardcoded off and the **draft is always greedy**; acceptance uses raw temperature-1 probabilities. No rejection resampling, so the output distribution is not guaranteed to match the main model's.
7. **Simplified speculative sampling** — threshold accept rule, not optimal Metropolis–Hastings (Leviathan et al.). The rule biases the output distribution (it does not resample on rejection), so it is not lossless. Future work: full speculative sampling for distribution-correctness at equal acceptance.
8. **Batch-EOS semantics** — in `Transformer.generate` with `B > 1`, finished rows keep generating until *every* row emits EOS; there is no per-row masking of future samples.
9. **No training-mode exception safety** — the `self.train()` restore in `generate` is not wrapped in `try/finally`; an exception mid-generation leaves the model in eval mode.

---

## Debugging Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Gibberish after first token | `start_pos` off-by-one | Trace `prompt_len + step` in `generate()`; the first decode forward must use `start_pos=prompt_len` |
| Same output every run | `temperature=0` or broken sampling | Check `_sample` args; `temperature == 0.0` is an exact equality — `1e-9` is not greedy-deterministic in *code path*, though it is effectively greedy in *output* |
| `RuntimeError: output_head not set` | `MTPModule` built without a head | Call `set_output_head(model.head)` before the first draft forward — see `main()` and [MTP](concepts/moe-mtp.md) §10.5 |
| Speculative never accepts | Random MTP weights | Train with `mtp_depth=1`, load `mtp.*` keys; check for the `[warn] No MTP weights` line |
| OOM on long context | Cache exceeds `max_seq_len` | Truncate prompt or raise `max_seq_len` in config; the MLA `end_pos` guard raises before corruption |
| Missing `head.weight` on load | Weight tying | Expected — `strict=False` restores it through the shared `embed.weight` storage |
| Stale context between prompts | KV cache not reset | `Transformer.generate` resets internally; bare `forward(use_cache=True)` calls do not — call `model.reset_cache()` |
| REPL re-prints the prompt | Slicing bug | Decode `output_ids[0, input_ids.shape[1]:]`, not the full tensor |
| `Layer N: end_pos exceeds max_seq_len` | `generate`'s length break bypassed (e.g. manual loop) | Respect the bound: break before the forward whose `start_pos + 1 > max_seq_len` |

---

## Appendix A — Decode timeline

Standard decode, prompt=4 tokens, generating 3 tokens:

```
Step 0 (prefill): forward(tokens[0:4], start_pos=0)     cache: [0,1,2,3]
Step 1:           forward([t5], start_pos=4)             cache: [0,1,2,3,4]
Step 2:           forward([t6], start_pos=5)             cache: [0,1,2,3,4,5]
Step 3:           forward([t7], start_pos=6)             cache: [0,1,2,3,4,5,6]
```

Speculative decode at step 1 (if draft accepted):

```
Step 1: main→t5, draft→t6, accept → emit [t5, t6] in one logical step
        cache grows by 2 positions
```

---

## Appendix B — FAQ

**Q: Must I call `reset_cache` between REPL turns?** No — every `generate()` call resets the cache internally and re-prefills the full message history. `/clear` drops the `messages` list, so the next turn's prompt contains only the new message; the cache is rebuilt from scratch either way. Between *bare* `forward(use_cache=True)` calls (notebook use), yes — call `model.reset_cache()`.

**Q: Can I use speculative without MTP checkpoint?** The draft head will be random — acceptance rate near zero, and every step falls back to `token_main`. The `[warn] No MTP weights in checkpoint; draft head is uninitialised.` line tells you this happened.

**Q: Why greedy draft in speculative path?** Simplifies the acceptance check; MTP was trained with teacher forcing on ground-truth prefixes, so its argmax continuation is the most informative candidate to verify. Temperature is applied to the *main* token's sample, not the draft.

**Q: `temperature=0.7` with speculative — what exactly is greedy?** The draft (`token_draft`) is always the draft distribution's argmax. The main token is sampled at temperature 0.7. Acceptance always compares the raw (unscaled, temperature-1) probabilities.

**Q: Does the KV cache grow during a long chat?** Not across turns — each turn re-prefills from scratch. Within one `generate` call the cache is a fixed `max_seq_len` allocation; generation stops when it would exceed the bound.

---

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| Prefill | Process full prompt in one forward |
| Decode | Single-token forwards with cache |
| `start_pos` | Cache write offset; equals `prompt_len + step` at loop iteration `step` |
| `end_pos` | `start_pos + seqlen`; the exclusive write bound and the cache-read prefix length |
| Nucleus | The smallest token set whose cumulative probability reaches `top_p` |
| `acceptance_threshold` | Min $p_{\text{main}} / p_{\text{draft}}$ ratio (0.8) |
| `generate_step` | One speculative decode iteration |
| Split-K | Splitting one query's KV reduction across parallel kernel programs |

---

## Worked Example — Prefill + 3 Decode Steps

**Setup:** $B=1$, prompt `[101, 2345, 678]` (length 3), greedy decode, `max_new_tokens=2`.

| Step | `input_ids` | `start_pos` | Cache positions written | Sampled token |
|---|---|---|---|---|
| Prefill | `[101,2345,678]` | 0 | 0,1,2 | (logits at pos 2 used) |
| Decode 0 | `[9012]` | 3 | 3 | argmax of last logits |
| Decode 1 | `[5555]` | 4 | 4 | argmax |

**Tensor shapes:**
- Prefill logits: $(1, 3, 100\,018)$ — the vocabulary has 100,018 ids (see [Data Pipeline](concepts/data-pipeline.md))
- Decode logits: $(1, 1, 100\,018)$ each step

**Common bug:** Using `start_pos=2` on first decode step (off by one from prompt length 3).

---

## Comparison — Standard vs Speculative

| Property | `Transformer.generate` | `SpeculativeDecoder.generate` |
|---|---|---|
| Sampling | temp / top-k / top-p | temp (main token only); top-k/top-p off; draft greedy |
| Tokens per forward | 1 | 1–2 |
| MTP required | No | Yes (trained weights) |
| KV cache | Main model only | Main model only |
| Batch size | $B \ge 1$ | $B = 1$ |
| Acceptance rule | N/A | $p_{\text{main}} \ge 0.8 \cdot p_{\text{draft}}$ |
| Cache reset | Internal (`reset_cache` first) | Internal (`reset_cache` first) |
| Trunk forwards per token | 1 | ~1–2 (2 when drafting, 1.5 expected) |

---

## Integration with Chat Template

`generate_interactive` uses:

```python
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)
```

The template inserts role tokens (`user`, `assistant`) per DeepSeek-Coder format. **Tokenizer path** comes from `data.tokenizer_path` in YAML (`deepseek-ai/deepseek-coder-v2-lite`).

Multi-turn: each turn appends to `messages`, and the *entire* history is re-tokenised and re-prefilled (the cache is reset and rebuilt per `generate` call). The prompt therefore grows with the conversation — REPL latency grows with history length, and very long chats hit the `max_seq_len` bound (generation then stops early, per the length break in `generate`).

---

## Appendix D — Extended Walkthroughs

### MLA KV Cache During Decode

During autoregressive decode (seqlen=1), MLA reads and writes compressed latent vectors — not full K/V heads:

**What's cached per layer:**

```python
# Write (one token):
self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()  # (B, 1, 192) — compressed latent
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()        # (B, 1, 24)  — RoPE key

# Read (full context):
ctx_kv = self.kv_cache[:bsz, :end_pos]  # (B, end_pos, 192) — all latents
ctx_pe = self.pe_cache[:bsz, :end_pos]  # (B, end_pos, 24)  — all RoPE keys
```

**Per-step memory read:**

At each decode step, MLA reads:
- `end_pos × 192` floats (latents)
- `end_pos × 24` floats (RoPE keys)
- Total: `end_pos × 216` floats per layer

For 128K context (a hypothetical long-context config — the canonical `max_seq_len` is 2048): $131072 \times 216 \times 2 = 56.6$ MB per layer per decode step.

Compare to standard MHA: $131072 \times 1536 \times 2 = 402.7$ MB per layer per decode step. **MLA reads 7.1× less data per decode step** — at the canonical 2048-token window the numbers scale down proportionally (15.9 MB vs 113 MB total, see [Complexity Analysis](#complexity-analysis)). Because decode is memory-bandwidth-bound, this ratio is roughly the decode speedup MLA buys; the per-step latency grows linearly with context either way, which is what split-K ([Flash Decoding](#flash-decoding--split-k-for-long-contexts)) addresses at scale.

**SDPA path materialization trade-off:**

During decode, the SDPA path:
1. Reads all cached latents (`ctx_kv`)
2. Materializes $K_{\text{nope}}$ and $V$ via BMM (expand latent → full K/V per head)
3. Concatenates with RoPE key
4. Runs `F.scaled_dot_product_attention` with Q (1 token) vs all cached K/V

This materialization happens **every decode step** on the SDPA path — it's the trade-off: the SDPA path uses a fused GPU kernel (faster) but doesn't use the true absorption trick. The manual path keeps everything in latent space (no materialization) but is slower due to Python loops. At the 411.6M scale, the SDPA path is faster overall. (The Triton fused path — `models/mla_triton.py:triton_mla_attention` — fuses materialization + RoPE + attention into one kernel and is the third option; see [Triton Kernels](concepts/kernels-and-ops.md).)

### Standard Generation — full loop with shapes

`Transformer.generate()` in `models/transformer.py:Transformer.generate` implements the full prefill + decode loop:

```python
@torch.inference_mode()
def generate(self, input_ids, max_new_tokens=512, temperature=1.0,
             top_p=0.9, top_k=0, eos_token_id=None):
    self.reset_cache()
    self.eval()

    # 1. Prefill: process entire prompt in one forward pass
    prefill_logits = self.forward(output, start_pos=0, use_cache=True)
    next_logits = prefill_logits[:, -1, :]

    # 2. Decode: one token at a time using KV cache
    for step in range(max_new_tokens):
        next_token = self._sample(next_logits, temperature, top_p, top_k)
        output = torch.cat([output, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).any():
            break
        if output.size(1) >= self.max_seq_len:
            break

        # Process only the new token (using cached K/V from previous steps)
        decode_logits = self.forward(next_token, start_pos=prompt_len + step, use_cache=True)
        next_logits = decode_logits[:, -1, :]

    return output
```

**`reset_cache()`** clears MLA caches (kv_cache and pe_cache) across all 18 layers — essential for independent generations.

**Prefill phase:** The full prompt (T tokens) is processed in one forward pass with parallel attention across all positions 0..T-1. The KV cache is populated with all prompt tokens. Only the logits at the last position are used for sampling.

**Decode phase:** Each step processes exactly 1 new token. The `start_pos` parameter tells MLA where in the cache to write the new K/V — after a prefill of length `prompt_len`, decode step `step` writes at position `prompt_len + step`. Off-by-one here causes cache corruption and gibberish output.

**Termination:** Generation stops on EOS token, when output reaches `max_seq_len` (prevents cache overflow — MLA caches are pre-allocated to `max_seq_len`), or after `max_new_tokens` iterations. (The section [Standard Generation](#standard-generation--generate-walkthrough) above annotates the *current* source, including the per-row `finished` mask and the train-mode restore; the sketch here is the same loop in condensed form.)

### Top-p/Top-k Sampling

`Transformer._sample(logits, temperature, top_p, top_k)` implements three sampling modes with a fixed execution order:

**1. Temperature scaling:**

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

- $\tau \to 0$: greedy argmax (deterministic)
- $\tau = 1.0$: native model distribution
- $\tau > 1$: flatter, higher entropy

**2. Top-k masking:**

Keep only the $k$ largest logits; mask all others to $-\infty$ before softmax. This hard-cutoff prevents sampling from low-probability tails. `top_k=0` disables.

```python
if top_k > 0:
    kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
    logits = logits.masked_fill(logits < kth_vals, float("-inf"))
```

**3. Top-p (nucleus) filtering:**

(Holtzman et al., 2020) Sort probabilities descending, find the smallest cumulative set whose probability exceeds `top_p`, then zero out everything else and renormalize:

```python
if top_p < 1.0:
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
```

**Execution order:** temperature → top-k → softmax → top-p → multinomial. Top-k is applied first (hard cutoff), then top-p adapts within the remaining set.

**Combined top-k + top-p rationale:** Top-k alone uses a fixed cutoff that may include irrelevant tokens if the distribution is flat. Top-p alone can produce a very large nucleus when many tokens have similar probability. Combined: top-k limits maximum candidates, top-p adapts within that set — best of both worlds. (The full derivation of each step lives in [Sampling Theory](#sampling-theory--temperature-top-k-top-p).)

---

## Check Your Understanding

1. **Temperature semantics.** With logits $z = (5.0, 3.0, 1.0)$, what are the three probabilities at $\tau = 1$? At $\tau \to 0^+$, which token wins, and why does the code special-case `temperature == 0.0` to argmax instead of just dividing by a tiny number? *(Answer: softmax(5,3,1) ≈ (0.844, 0.114, 0.042); at $\tau \to 0^+$ the mass concentrates on the argmax because every non-max term $e^{(z_i - z_{max})/\tau} \to 0$; dividing by a tiny-but-nonzero $\tau$ is mathematically equivalent but wastes a kernel launch and loses the exact deterministic tie-break.)*

2. **Top-p boundary.** `probs = (0.4, 0.3, 0.2, 0.1)` and `top_p = 0.6`. Which tokens survive the nucleus, and why does the code use `(cumulative - sorted_probs) > top_p` rather than `cumulative > top_p`? *(Answer: sorted = (0.4, 0.3, 0.2, 0.1); cumulative = (0.4, 0.7, 0.9, 1.0); remove = (0, 0.4, 0.7, 0.9) > 0.6 → tokens 3 and 4 removed, so the 0.4 and 0.3 survive — the token that *crosses* the threshold is kept. The `- sorted_probs` back-out keeps the crossing token; the naive `cumulative > top_p` would drop it.)*

3. **Position invariant.** A prompt of length 5 generates with `max_new_tokens=10` and `max_seq_len=12`. At which `start_pos` does the final forward run, and what stops the loop first? *(Answer: decode step 3 uses `start_pos = 5 + 3 = 8`, the forward's `end_pos = 9`; the loop breaks when `output.size(1) = 12 ≥ max_seq_len` — after step 6 — so only 6 tokens are generated, not 10. The `max_seq_len` bound fires before `max_new_tokens`.)*

4. **Cache growth.** A bare MLA forward with `use_cache=True` at batch 1 allocates a cache of batch 16 (`max(1, 0·2, 16)`). What happens to the cached prefix if a later forward uses batch 32, and why does `Transformer.generate` never hit this? *(Answer: `_ensure_cache` reallocates to batch 32 (doubling), discarding the old contents — the positional state is lost. `generate` never hits it because it resets the cache at the start and uses a fixed `bsz` throughout, so `bsz ≤ _cache_batch` always holds after the first allocation.)*

---

## References

- `inference/generate.py`, `inference/speculative.py`
- `models/transformer.py` — `generate()`, `_sample()`, `forward_with_hidden()`
- [MTP](concepts/moe-mtp.md) — MTP theory, acceptance math, cache consistency
- [transformer](concepts/foundations.md) — wiring
- [MLA](concepts/attention-and-precision.md) — KV cache internals
- [utils](concepts/kernels-and-ops.md) — checkpoint format
- [Triton Kernels](concepts/kernels-and-ops.md) — fused MLA/MoE kernels, register caps
- [Holtzman et al., 2020 — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — nucleus sampling
- [Leviathan et al., 2023 — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Dao et al., 2023 — FlashDecoding](https://arxiv.org/abs/2205.14135) — split-K rationale


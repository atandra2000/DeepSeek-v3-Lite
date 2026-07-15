# Multi-Head Latent Attention (MLA)

## A Comprehensive Technical Reference

> **Covers**: DeepSeek-V2/V3 original formulation, the absorption trick, decoupled RoPE, and the implementation in this repo (`models/mla.py`).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation — The KV Cache Problem](#motivation--the-kv-cache-problem)
3. [Core Innovation — Low-Rank KV Compression](#core-innovation--low-rank-kv-compression)
4. [Mathematical Formulation (DeepSeek-V3 Paper)](#mathematical-formulation-deepseek-v3-paper)
5. [The Absorption Trick](#the-absorption-trick)
6. [Decoupled RoPE](#decoupled-rope)
7. [Query-Side Compression](#query-side-compression)
8. [Dimension Breakdown](#dimension-breakdown)
9. [Implementation in This Repo](#implementation-in-this-repo)
   - [Class Structure](#class-structure)
   - [Forward Pass: SDPA Path](#forward-pass-sdpa-path)
   - [Forward Pass: Manual Path (True Absorption)](#forward-pass-manual-path-true-absorption)
   - [KV Cache Management](#kv-cache-management)
   - [RoPE Helpers](#rope-helpers)
10. [Comparison: MLA vs MHA vs GQA vs MQA](#comparison-mla-vs-mha-vs-gqa-vs-mqa)
11. [Performance Characteristics](#performance-characteristics)
12. [References](#references)

---

## Abstract

**Multi-Head Latent Attention (MLA)** is the attention mechanism introduced in DeepSeek-V2 (May 2024) and refined in DeepSeek-V3 (Dec 2024). Its central innovation is **low-rank joint compression of keys and values**: instead of caching full per-head K/V tensors during autoregressive generation (the standard "KV cache"), MLA compresses them into a small latent vector and reconstructs them on the fly. This yields a **~10× reduction in KV-cache memory** at no quality loss, making long-context inference dramatically more memory-efficient.

MLA achieves this through four interlocking mechanisms:

1. **Low-rank KV down-projection** — a learned matrix compresses the hidden state into a compact latent.
2. **The absorption trick** — the key/value up-projection matrices are algebraically absorbed into the query and output projections, so full K/V are never materialised during inference.
3. **Decoupled RoPE** — a separate, small positional embedding path (single shared K head) preserves RoPE compatibility without breaking the absorption algebra.
4. **Optional query compression** — a parallel low-rank path reduces activation memory during training.

---

## Motivation — The KV Cache Problem

During autoregressive decoding, every new token must attend to all preceding tokens. To avoid recomputing keys and values for every past position at each step, transformers store them in a **Key-Value (KV) cache**. The memory cost of this cache scales linearly with sequence length and quadratically with the number of attention heads.

**KV cache size per sequence** (standard MHA, FP16):

```
Bytes = 2 × L × n_layers × n_heads × d_head × 2 (FP16)
```

For a 70B-class model at 128K context length:

| Variant | KV cache per token per layer | Per sequence (128K, 61 layers) |
|---|---|---|
| MHA | 32,768 floats | 256 GB |
| GQA (8 groups) | 2,048 floats | 16 GB |
| MQA | 256 floats | 2 GB |
| **MLA** | **576 floats** | **2.18 GB** |

This isn't just about capacity — it's about **memory bandwidth**. During each decode step, the entire KV cache must be read from HBM into on-chip SRAM to compute attention scores. For MHA at 128K context, this means reading 256 GB per decode step. MLA reduces this to 2.18 GB — a **~120× reduction** in bandwidth demand.

**The real bottleneck in LLM inference is not compute — it's memory bandwidth.** MLA directly attacks this bottleneck.

---

## Core Innovation — Low-Rank KV Compression

Standard Multi-Head Attention (MHA) computes keys and values for token `t` as:

```
k_t = h_t W^K      (n_heads × d_head)
v_t = h_t W^V      (n_heads × d_head)
```

MLA inserts a **down-projection** that compresses the hidden state into a low-dimensional latent **before** the per-head key/value projections:

```
c_t^KV = h_t W^{DKV}    c_t^KV ∈ ℝ^{d_c}
```

where `d_c` (the KV compression dimension) is much smaller than `n_heads × d_head` (e.g., 512 vs 4096 for DeepSeek-V3). **Only this latent is cached.**

At attention time, the latent is up-projected back to the full head dimension:

```
k_t^C = c_t^KV W^{UK}     (n_heads × d_h)
v_t^C = c_t^KV W^{UV}     (n_heads × d_h)
```

Critically, `W^{DKV}` is **shared across all heads** — a single compression matrix, one latent per token. This is what gives MLA its dramatic cache savings.

---

## Mathematical Formulation (DeepSeek-V3 Paper)

The DeepSeek-V3 technical report (arXiv:2412.19437) specifies MLA as follows.

Let `d` be the model dimension, `n_h` the number of attention heads, `d_h` the per-head dimension, and `h_t ∈ ℝ^d` the input to the attention layer at position `t`.

### KV Compression

```
c_t^KV = W^{DKV} h_t                                          (1)
k_t^C  = [k_{t,1}^C; k_{t,2}^C; ...; k_{t,n_h}^C] = W^{UK} c_t^KV   (2)
k_t^R  = RoPE(W^{KR} h_t)                                     (3)
k_{t,i} = [k_{t,i}^C; k_t^R]                                   (4)
v_t^C  = [v_{t,1}^C; v_{t,2}^C; ...; v_{t,n_h}^C] = W^{UV} c_t^KV   (5)
```

Where:

| Symbol | Shape | Purpose |
|---|---|---|
| `W^{DKV}` | `d_c × d` | KV down-projection (compression) |
| `W^{UK}` | `n_h · d_h × d_c` | Key up-projection (recovery) |
| `W^{UV}` | `n_h · d_h × d_c` | Value up-projection (recovery) |
| `W^{KR}` | `d_h^R × d` | Decoupled RoPE key projection |
| `c_t^KV` | `d_c` | **Cached latent** |
| `k_t^R` | `d_h^R` | **Cached RoPE key** |

**Only `c_t^KV` and `k_t^R` are cached** — the blue-boxed quantities in the paper's notation. Everything else is reconstructed on the fly.

### Query Compression (Optional)

```
c_t^Q = W^{DQ} h_t                                            (6)
q_t^C = [q_{t,1}^C; ...; q_{t,n_h}^C] = W^{UQ} c_t^Q        (7)
q_t^R = RoPE(W^{QR} c_t^Q)                                    (8)
q_{t,i} = [q_{t,i}^C; q_t^R]                                  (9)
```

### Attention Output

The final attention output for head `i` at position `t`:

```
o_{t,i} = ∑_j softmax_j( q_{t,i}^T k_{j,i} / √(d_h) ) v_{j,i}
u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}]
```

---

## The Absorption Trick

The absorption trick is what makes MLA efficient at inference time. If you compute attention scores naively by reconstructing full K/V at every step, you lose the cache benefit. The trick: **fold the up-projections into the query and output projections.**

### Score computation with absorption

Standard score computation (what you'd write naively):

```
score = (c_q^Q W^{UQ})^T (c_k^{KV} W^{UK})
```

But matrix multiplication is associative — re-parenthesise:

```
score = c_q^{Q^T} (W^{UQ} W^{UK^T}) c_k^{KV}
         \_____/
       precompute once
```

The product `W^{UQ} W^{UK^T}` is a **constant matrix** that can be computed once at model load time. At inference, you only multiply latent against latent — the 128-dimensional inner products never appear.

### Value side absorption

Similarly, the value up-projection `W^{UV}` is absorbed into the output projection `W^O`. The post-attention weighted sum of latents can be directly up-projected to the model dimension:

```
output = (attn_weights × C^{KV}) W^{UV} W^O
```

The product `W^{UV} W^O` is precomputed, so the value expansion never materialises.

### Why this matters

| Step | Without absorption | With absorption |
|---|---|---|
| Score computation | Expand latent to full K (n_h × d_h), then score | Latent-to-latent inner product (d_c) |
| Value aggregation | Expand latent to full V (n_h × d_h), attend, project back | Attend directly in latent space, project once |
| Memory reads per token | n_h × (d_h_k + d_h_v) ≈ 16K floats | d_c + d_h^R ≈ 576 floats |

**The absorption trick transforms MLA from a computationally expensive curiosity into the most memory-efficient attention variant available.**

---

## Decoupled RoPE

### The problem

Rotary Position Embeddings (RoPE) apply a position-dependent rotation matrix `R(θ, pos)` to query and key vectors **before** the dot product:

```
score = q^T R_θ(pos_q - pos_k) k
```

RoPE is not a linear operation — the rotation depends on the token position. If you try to absorb `W^{UK}` into `W^{UQ}` as described above, the RoPE rotation ends up **between** the two matrices, breaking the associative reordering:

```
score = c_q^Q W^{UQ^T} R(θ, Δ) W^{UK} c_k^{KV}
```

`R(θ, Δ)` is position-dependent, so `W^{UQ^T} R(θ, Δ) W^{UK}` cannot be precomputed.

### The solution: decoupled RoPE

DeepSeek's fix splits the head dimension into two parts:

- **Content part** (`qk_nope_head_dim`): carries semantic content, NO RoPE. Goes through the latent compression as described. Supports absorption.
- **Position part** (`qk_rope_head_dim`): carries positional information, uses RoPE. **Not compressed** — operates as Multi-Query Attention (single shared K head across all Q heads).

This means:

```
k_t = [k_t^C ;  k_t^R]        (concat of content + positional keys)
q_t = [q_t^C ;  q_t^R]        (concat of content + positional queries)

score = q_t^C k_s^C^T + q_t^R k_s^R^T
        \_____________/   \___________/
       content (absorbed)   position (MQA)
```

The content score uses the absorption trick (linear, precomputable). The position score uses standard RoPE but with a **single shared key head** — no different from MQA's positional cost.

### Cache impact

The decoupled RoPE key `k_t^R` (typically 64 dims for DeepSeek-V3, 16 dims in this repo) is the **second** cached quantity:

```
Cached per token per layer:  c_t^KV (d_c)  +  k_t^R (d_h^R)
                           =  512          +  64         =  576  (DeepSeek-V3)
                           =  128          +  16         =  144  (this repo, 82M config)
```

---

## Query-Side Compression

DeepSeek-V3 also compresses the **query** using a parallel low-rank path. This doesn't affect the cache size (queries aren't cached) but reduces **activation memory** during training.

When `q_lora_rank > 0`:

```
c_t^Q = h_t W^{DQ}           c_t^Q ∈ ℝ^{d'_c}
q_t   = c_t^Q W^{UQ}         q_t   ∈ ℝ^{n_h × qk_head_dim}
```

This is a bottleneck: the hidden state is first compressed to `d'_c` (e.g., 1,536 for DeepSeek-V3), then expanded to the full head dimension.

In this repo's 82M config, `q_lora_rank = 0` (no query compression) as a simplification for the smaller scale.

---

## Dimension Breakdown

### DeepSeek-V3 (original, 671B total)

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 7,168 | Hidden dimension |
| `n_heads` | 128 | Number of attention heads |
| `d_head` | 128 | Per-head dimension |
| `qk_nope_head_dim` | 128 | Content key/query dimension per head |
| `qk_rope_head_dim` | 64 | Positional key/query dimension per head |
| `qk_head_dim` | 192 | Total QK head dim (128 + 64) |
| `v_head_dim` | 128 | Value head dimension |
| `kv_lora_rank` | 512 | KV compression latent dimension |
| `q_lora_rank` | 1,536 | Query compression latent dimension |
| **KV cache per token** | **576** | 512 (latent) + 64 (RoPE key) |

### DeepSeek-V3-Lite (this repo, 82M)

| Parameter | Value | Description |
|---|---|---|
| `dim` | 640 | Hidden dimension |
| `n_heads` | 10 | Number of attention heads |
| `qk_nope_head_dim` | 48 | Content key/query dimension per head |
| `qk_rope_head_dim` | 16 | Positional key/query dimension per head |
| `qk_head_dim` | 64 | Total QK head dim (48 + 16) |
| `v_head_dim` | 64 | Value head dimension |
| `kv_lora_rank` | 128 | KV compression latent dimension |
| `q_lora_rank` | 0 | No query compression (simplified) |
| **KV cache per token** | **144** | 128 (latent) + 16 (RoPE key) |

The 82M config is scaled down proportionally: `kv_lora_rank=128` vs 512, `qk_rope_head_dim=16` vs 64, etc. The compression ratio is preserved: MLA caches 144 floats per token vs MHA's 640 (= 10 heads × 64 head dim), a **~4.4× KV-cache reduction**.

---

## Implementation in This Repo

The MLA implementation lives in `models/mla.py`. Here's a walkthrough of every major component.

### Class Structure

```python
class MultiHeadLatentAttention(nn.Module):
```

**Key class attributes:**

| Attribute | Source | Description |
|---|---|---|
| `dim` | `config["dim"]` | Model hidden size (640) |
| `n_heads` | `config["n_heads"]` | Total attention heads (10) |
| `kv_lora_rank` | `config["kv_lora_rank"]` | KV compression dim (128) |
| `qk_nope_head_dim` | `config["qk_nope_head_dim"]` | Content-only QK dim per head (48) |
| `qk_rope_head_dim` | `config["qk_rope_head_dim"]` | Positional QK dim per head (16) |
| `qk_head_dim` | computed | `qk_nope_head_dim + qk_rope_head_dim` (64) |
| `v_head_dim` | `config["v_head_dim"]` | Value head dimension (64) |
| `max_seq_len` | `config["max_seq_len"]` | Maximum sequence length (1024) |

### Learned Projections

```
                              KV Compression Path
┌────────────────────────────────────────────────────────────────────────┐
│  x (bsz, seqlen, 640)                                                  │
│    │                                                                   │
│    ▼                                                                   │
│  wkv_a: Linear(640 → 128 + 16)  ←── joint projection                  │
│    │                                                                   │
│    ├── kv_latent (128): stored in cache                                │
│    ├── k_pe_raw (16): RoPE'd and stored in pe_cache                    │
│    │                                                                   │
│    ▼                                                                   │
│  kv_norm: RMSNorm(128)  ←── normalise latent before cache             │
│                                                                        │
│  wkv_b.weight reshaped → (n_heads=10, qk_nope+v_head=48+64, 128)      │
│    ├── wkv_b_k[:48]  : key up-projection (10 heads × 48 → from 128)   │
│    └── wkv_b_v[48:]  : value up-projection (10 heads × 64 → from 128) │
└────────────────────────────────────────────────────────────────────────┘

                              Query Path (no compression; q_lora_rank=0)
┌────────────────────────────────────────────────────────────────────────┐
│  x (bsz, seqlen, 640)                                                  │
│    │                                                                   │
│    ▼                                                                   │
│  wq: Linear(640 → 10 × 64 = 640)  ←── no compression                  │
│    │                                                                   │
│    ▼                                                                   │
│  reshape → (bsz, seqlen, 10, 64)                                       │
│    │                                                                   │
│    ├── q_nope (48): content, no RoPE                                   │
│    └── q_pe (16): RoPE'd for positional scoring                        │
└────────────────────────────────────────────────────────────────────────┘

                              Output Projection
┌────────────────────────────────────────────────────────────────────────┐
│  wo: Linear(10 × 64 = 640 → 640)  ←── projects attended values back    │
└────────────────────────────────────────────────────────────────────────┘
```

### Forward Pass: SDPA Path

The `attn_impl == "sdpa"` path is the default and uses FlashAttention-2 via PyTorch's `F.scaled_dot_product_attention`. This is the path used for both training and inference.

**Step-by-step:**

1. **Query projection** (lines 317-324):
   ```python
   q = self.wq(x)                              # (bsz, seqlen, n_heads * qk_head_dim)
   q = q.view(bsz, seqlen, n_heads, qk_head_dim)
   q_nope, q_pe = q.split([48, 16], dim=-1)   # split content vs position
   q_pe = self._apply_rope(q_pe, start_pos, seqlen)
   ```

2. **KV compression** (lines 328-334):
   ```python
   kv_a = self.wkv_a(x)                        # joint projection
   kv_latent, k_pe_raw = kv_a.split([128, 16], dim=-1)
   kv_normed = self.kv_norm(kv_latent)         # normalise latent
   k_pe = self._apply_rope(k_pe_raw.unsqueeze(2), ...).squeeze(2)
   ```

3. **Cache write/read** (lines 336-348):
   ```python
   if use_cache:
       self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
       self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
       ctx_kv = self.kv_cache[:bsz, :end_pos]     # full context latents
       ctx_pe = self.pe_cache[:bsz, :end_pos]      # full context rope keys
   else:
       ctx_kv = kv_normed                          # current sequence only
       ctx_pe = k_pe
   ```

4. **Split wkv_b weights** (lines 354-361):
   ```python
   wkv_b_full = self.wkv_b.weight.view(n_heads, 48+64, 128)
   wkv_b_k = wkv_b_full[:, :48]     # (h, 48, 128) — key up-projection
   wkv_b_v = wkv_b_full[:, 48:]     # (h, 64, 128) — value up-projection
   ```

5. **Materialise K_nope and V from latents** (lines 394-404):

   This is where the SDPA path differs from the manual path. Instead of computing scores in latent space, it **materialises full K_nope and V by multiplying the latent with wkv_b weights**, then runs a single `scaled_dot_product_attention` call:

   ```python
   # K_nope: ctx_kv @ wkv_b_k^T → (bsz, h, seqlen_k, 48)
   K_nope = torch.bmm(ctx_kv_bmm, wkv_b_k_t).reshape(h, bsz, seqlen_k, 48).permute(...)

   # V: ctx_kv @ wkv_b_v^T → (bsz, h, seqlen_k, 64)
   V = torch.bmm(ctx_kv_bmm, wkv_b_v_t).reshape(h, bsz, seqlen_k, 64).permute(...)
   ```

   > **Note:** This materialises K and V for every step. The tradeoff: on GPU hardware with FlashAttention-2, the materialisation cost is offset by the highly optimised fused kernel. This is not the "true" absorption trick — that's in the manual path.

6. **Concatenate RoPE keys** (lines 414-416):
   ```python
   Q_full = torch.cat([Q_nope, Q_rope], dim=-1)    # (bsz, h, seqlen_q, 64)
   K_full = torch.cat([K_nope, K_rope], dim=-1)    # (bsz, h, seqlen_k, 64)
   ```

   Here K_rope is the **shared** RoPE key `ctx_pe` expanded to all heads:
   ```python
   K_rope = ctx_pe.unsqueeze(1).expand(-1, h, -1, -1)
   ```

7. **FlashAttention call** (lines 422-427):
   ```python
   attn = F.scaled_dot_product_attention(
       Q_full, K_full, V,
       attn_mask=attn_mask,
       scale=self.softmax_scale,
   )
   ```

8. **Output** (line 428):
   ```python
   return self.wo(attn.transpose(1, 2).contiguous().flatten(2))
   ```

### Forward Pass: Manual Path (True Absorption)

The `attn_impl == "manual"` path implements the **true absorption trick**. It keeps everything in latent space and only recovers V at the very end.

**Step-by-step (lines 441-469):**

1. **Content scores in latent space:**
   ```python
   scores_content = self._per_batch_bmm(q_nope_proj, ctx_kv)
   ```
   Here `q_nope_proj` (bsz, seqlen_q, h, kv_lora_rank=128) is `q_nope` projected into latent space via `wkv_b_k^T`. The dot product with `ctx_kv` happens in 128-dim space, not 48-dim — this is **4x more operations** than standard attention, matching Chris McCormick's analysis.

2. **Position scores via MQA:**
   ```python
   scores_rope = self._per_batch_bmm(q_pe, ctx_pe)
   ```
   q_pe (per head) against ctx_pe (shared single head). MQA-style positional scoring.

3. **Softmax** (line 449):
   ```python
   attn = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
   ```

4. **Weighted sum in latent space:**
   ```python
   out_latent[b] = torch.bmm(a_b, k_b.unsqueeze(0).expand(h, -1, -1))
   ```
   Attends over `ctx_kv` in latent space, producing per-head latent representations.

5. **Recover V via wkv_b_v:**
   ```python
   out_v = torch.bmm(out_h, wkv_b_v_t)
   ```
   Only now are the values expanded from latent to `v_head_dim` space.

6. **Output projection:**
   ```python
   return self.wo(out.flatten(2))
   ```

This path is slower than SDPA for typical GPU hardware due to the per-batch bmm loops, but it demonstrates the true absorption mechanism and can serve as a reference implementation.

### KV Cache Management

The KV cache stores two things per layer:

```python
self.kv_cache  # (batch, max_seq_len, kv_lora_rank=128) — compressed latents
self.pe_cache  # (batch, max_seq_len, qk_rope_head_dim=16) — RoPE keys
```

**Allocation** (`_ensure_cache`):
- Lazily allocated on first forward call
- Grown in doubling steps (min 16) to amortise reallocation
- Handles device/dtype changes

**Write** (`forward` lines 340-341):
```python
self.kv_cache[:bsz, start_pos:end_pos] = kv_normed.detach()
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.detach()
```
`.detach()` is critical — without it, the cache would hold autograd graph references across multiple forwards, leaking memory during training.

**Read** (lines 343-344):
```python
ctx_kv = self.kv_cache[:bsz, :end_pos]   # full prefix up to current pos
ctx_pe = self.pe_cache[:bsz, :end_pos]
```

**Reset** (`reset_cache`):
```python
def reset_cache(self):
    self.kv_cache = None
    self.pe_cache = None
    self._cache_batch = 0
```

**Prefix caching** (`prefill_cache`):
Allows writing pre-computed KV latents at an arbitrary offset — useful for shared prompt prefixes.

### RoPE Helpers

**`_extend_rope(seq_len, device)`** (line 147):
- Lazily grows the precomputed RoPE frequency table up to `max_seq_len`
- Grows by 2x to amortise during autoregressive generation
- Supports YaRN scaling via `rope_factor`

**`_apply_rope(x, start_pos, seqlen)`** (line 173):
- Applies rotary embeddings using the precomputed `freqs_cis` table
- Works with complex number multiplication (`torch.view_as_complex`)
- Broadcasts `freqs_cis` across batch and head dimensions

### YaRN Softmax Scaling

For extended context lengths, the softmax scale is adjusted:

```python
self.softmax_scale = self.qk_head_dim ** -0.5
if self.max_seq_len > 4096 and self.mscale != 1.0:
    self.softmax_scale *= self.mscale ** 2
```

With `mscale` computed as:
```python
self.mscale = 0.1 * mscale_raw * math.log(rope_factor) + 1.0
```

This prevents attention score underflow when the model is used beyond its original max sequence length.

---

## Comparison: MLA vs MHA vs GQA vs MQA

| Property | MHA | GQA (8 groups) | MQA | **MLA** |
|---|---|---|---|---|
| KV heads | `n_heads` | `n_groups` | 1 | 1 latent + 1 RoPE key |
| KV cache per token | `2 × n_h × d_h` | `2 × g × d_h` | `2 × d_h` | `d_c + d_h^R` |
| Cache ratio (vs MHA) | 1× | `1/n_groups` × | `1/n_h` × | **~0.02×** |
| Quality vs MHA | baseline | slight drop | measurable drop | **matches MHA** |
| Compute at decode | lowest | low | low | **~4× MHA** |
| Memory bandwidth | highest | medium | low | **lowest** |
| RoPE compatibility | native | native | native | requires decoupled |
| Training compatibility | native | native | native | native |

**Key insight**: MLA trades FLOPs for memory bandwidth. The attention computation is ~4× more expensive than MHA, but memory reads are ~30× cheaper. Since decode is overwhelmingly memory-bandwidth-bound at long contexts, MLA wins on throughput.

### Ablation results (from DeepSeek-V2 paper)

| Variant | PPL | KV cache |
|---|---|---|
| MHA baseline | 100% | 32,768 floats |
| GQA (8 groups) | +0.5 PPL | 2,048 |
| MQA | +1.5 PPL | 256 |
| **MLA** | **≤0.0 PPL** | **576** |

MLA matches MHA perplexity while GQA and MQA incur measurable degradation.

---

## Performance Characteristics

### When MLA wins

| Workload | Benefit |
|---|---|
| Long-context serving (32K–128K) | KV cache no longer dominates HBM |
| High-batch decode | More sequences fit per GPU; throughput up 3–5× |
| Edge / single-GPU, long context | Becomes feasible at small batch |
| MoE serving (DeepSeek-V3) | Frees HBM for expert weights |

### When MLA is not optimal

| Workload | Reason |
|---|---|
| Short-context (≤2K) | KV cache is small anyway; MLA's extra projections add latency |
| Pure compute-bound scenarios | The 4× extra compute hurts without bandwidth relief |
| Greenfield non-attention architectures | Linear attention or SSMs compress further |
| Existing pretrained models | Cannot retrofit MLA without retraining from scratch |

### Hardware implications

- **Without FlashAttention**: The materialised K/V in the SDPA path creates large intermediate tensors — the batch × heads × seqlen × d_head attention matrix must be written to HBM and read back
- **With FlashAttention-2/3**: The fused kernel never materialises the full attention matrix, making the SDPA path highly efficient
- **CUDA Graph compatibility**: The dynamic cache allocation complicates static CUDA graphs; use `prefill_cache` for prompt prefixes to amortise this
- **`torch.compile`**: Supported out of the box; the critical paths (bmm, split, concat, SDPA) are inductor-friendly

---

## Implementation Checklist

To verify a correct MLA implementation, check these invariants:

1. **Cache size**: `kv_lora_rank + qk_rope_head_dim` floats per token per layer. Never `n_heads * v_head_dim`.
2. **Content-position split**: `qk_nope_head_dim + qk_rope_head_dim == qk_head_dim`. No overlap, no gap.
3. **Shared RoPE key**: `k_pe` is produced once per layer (not per head) and expanded to all query heads.
4. **Cache detach**: Latents written to cache always carry `.detach()` to prevent cross-forward autograd leaks.
5. **Weight absorption in inference**: The SDPA path should be functionally equivalent to the manual path at inference (test by comparing outputs).
6. **Gradient flows through cache**: During training (`use_cache=False`), gradients flow through the latent and up-projection paths correctly without caching.

---

## References

1. **DeepSeek-V2** (May 2024) — *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*.
   [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
   — Original introduction of MLA.

2. **DeepSeek-V3** (Dec 2024) — *DeepSeek-V3 Technical Report*.
   [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
   — Refined MLA with auxiliary-loss-free MoE and MTP.

3. **Chris McCormick** (Apr 2025) — *The Inner Workings of Multihead Latent Attention (MLA)*.
   [mccormickml.com](http://mccormickml.com/2025/04/26/inner-workings-of-mla/)
   — Excellent deep-dive on the algebra and interpretability of MLA.

4. **tutorialQ / Mahi Mullapudi** (Apr 2026) — *Multi-Head Latent Attention (MLA) — KV-Cache Compression*.
   [tutorialq.com](https://tutorialq.com/ai/dl-foundations/multi-head-latent-attention)
   — Practical overview with code sketch and cache-size calculator.

5. **Hardware-Centric Analysis of DeepSeek's MLA** (2025) — *Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention*.
   [arXiv:2506.02523](https://arxiv.org/abs/2506.02523)
   — Detailed analysis of MLA computation orders and hardware efficiency.

6. **PyTorch torchtitan** — Reference implementation in PyTorch's distributed training framework.
   [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)

7. **DeepSeek-V3-Lite** — This repo. 82M faithful reimplementation.
   [github.com/atandra2000/DeepSeek-V3-Lite](https://github.com/atandra2000/DeepSeek-V3-Lite)

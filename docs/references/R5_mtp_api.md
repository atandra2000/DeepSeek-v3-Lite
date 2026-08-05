# DeepSeek-v3-Lite — R5 Multi-Token Prediction API Reference

> **60-second summary:** `models/mtp.py` implements DeepSeek-V3's Multi-Token Prediction (MTP) — $D$ auxiliary heads that each fuse the main trunk's hidden state with the embedding of the *next* token, then predict the token *after that*. Training adds an auxiliary loss (weighted `0.3`) averaged over depths; inference reuses depth-1 as a draft model for speculative decoding. The MTP heads share the main model's embedding and output head by reference, adding only ~7.1M parameters (411.6M → 418.7M total with MTP).
>
> This is the API reference; for the derivation, length-alignment algebra, and speculative-decoding theory see [Multi-Token Prediction](../concepts/moe-mtp.md) (T5).

---

## 5.1 Public surface

| Symbol | Anchor | One-line purpose |
|---|---|---|
| `MTPBlock` | `models/mtp.py:MTPBlock.__init__` | Per-depth fusion block: dual RMSNorm → concat-projection → causal self-attention → SwiGLU FFN |
| `MTPBlock._get_causal_mask` | `models/mtp.py:MTPBlock._get_causal_mask` | Lazily built, cached upper-triangular `-inf` causal mask `(S, S)` |
| `MTPBlock.forward` | `models/mtp.py:MTPBlock.forward` | Fuse `(prev_hidden, target_emb)` → `(B, S, dim)` residual output |
| `MTPModule` | `models/mtp.py:MTPModule.__init__` | One prediction head for depth `d`: block + final RMSNorm + shared output head |
| `MTPModule.set_output_head` | `models/mtp.py:MTPModule.set_output_head` | Attach the shared `nn.Linear` head (required before forward) |
| `MTPModule.forward` | `models/mtp.py:MTPModule.forward` | `(prev_hidden, target_emb)` → `(logits, h_norm)`; raises if head unset |
| `MultiTokenPrediction` | `models/mtp.py:MultiTokenPrediction.__init__` | Wrapper: main `Transformer` + `D` `MTPModule`s sharing embed & head |
| `MultiTokenPrediction.forward` | `models/mtp.py:MultiTokenPrediction.forward` | `tokens` → `(main_logits, mtp_pairs)` with per-depth length alignment |
| `MultiTokenPrediction.compute_loss` | `models/mtp.py:MultiTokenPrediction.compute_loss` | `main_loss + 0.3·mtp_loss`; MTP loss is the mean across depths |

Module facts: ~115 lines, imports only `torch` / `torch.nn` / `torch.nn.functional`. No Triton paths, no dtype casts — token tensors must already be `torch.long` (the caller casts; see §5.7).

---

## 5.2 Config contract

Read by `MultiTokenPrediction.__init__` and by the `Pretrainer` gate (`training/pretrain.py:Pretrainer.__init__`). Canonical values from `configs/pretrain_a100_422m.yaml` (`model:` block):

| Key | Default | Meaning | Read where |
|---|---|---|---|
| `mtp_depth` | `1` | Number of MTP heads / auxiliary prediction depths | `MultiTokenPrediction.__init__`, `Pretrainer.__init__` (gate: `mtp_depth > 0`) |
| `mtp_loss_weight` | `0.3` | Scalar multiplying the mean-over-depths MTP loss | `MultiTokenPrediction.__init__`, `Pretrainer.__init__` (gate: `mtp_weight > 0.0`) |
| `mtp` (dict) | — | Alternative nested form `{depth: 1, weight: 0.3}` used when top-level `mtp_depth` is absent | `MultiTokenPrediction.__init__` only |

Resolution order in `MultiTokenPrediction.__init__`: if the literal key `"mtp_depth"` is in the config dict, use `config["mtp_depth"]` and `config.get("mtp_loss_weight", 0.3)`; otherwise fall back to `config.get("mtp", {})` with `mtp_cfg.get("depth", 1)` / `mtp_cfg.get("weight", 0.3)`. The two YAML configs shipped use the top-level form (`mtp_depth: 1`, `mtp_loss_weight: 0.3`). The model config itself is resolved as `config.get("model", config)` — both flat configs and wrapped `{model: …}` configs are accepted.

MTP is **enabled** in training only when both gates pass in `training/pretrain.py:Pretrainer.__init__`: `mtp_depth > 0 and config.mtp_weight > 0.0`. Otherwise `Pretrainer.mtp_wrapper` stays `None` and the plain `Transformer` path runs. When enabled, the parameter count reported for μP LR scaling is the MTP wrapper total (418.7M), so the scaled LR becomes **8.07e-4** instead of 8.14e-4 (see [Training](../training.md) and
[G2](../guides/G2_mup_and_lr_tuning.md)).

---

## 5.3 `MTPBlock`

### `MTPBlock.__init__`

```python
def __init__(self, config: dict):
```

Config keys read: `dim`, `n_heads`, `inter_dim`. Submodules (all bias-free):

| Attribute | Type | Shape |
|---|---|---|
| `norm_h`, `norm_e` | `nn.RMSNorm(dim, eps=1e-6)` | — |
| `proj` | `nn.Linear(dim * 2, dim, bias=False)` | `(dim, 2·dim)` |
| `norm_attn` | `nn.RMSNorm(dim, eps=1e-6)` | — |
| `attn` | `nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False)` | torch built-in **MHA** (not MLA — see §5.8) |
| `norm_ffn` | `nn.RMSNorm(dim, eps=1e-6)` | — |
| `w1`, `w3` | `nn.Linear(dim, inter_dim, bias=False)` | `(inter_dim, dim)` |
| `w2` | `nn.Linear(inter_dim, dim, bias=False)` | `(dim, inter_dim)` |
| `_causal_mask` | buffer `torch.empty(0, 0)`, `persistent=False` | grows lazily |
| `_causal_mask_size` | `int = 0` | cache key, not a parameter |

Canonical dims: `dim=768`, `n_heads=12`, `inter_dim=1536`. Per-depth parameter count: `2·768·768 (proj) + 3·768·1536 (w1/w2/w3) + 12·64·768·2 (MHA) ≈ 4.1M` (includes the final norm's 768; excludes the shared head) — this is the bulk of the ~7.1M MTP delta (`count_parameters` dedups shared tensors by `id`, see `models/transformer.py:count_parameters`).

### `MTPBlock._get_causal_mask`

```python
def _get_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
```

Builds `torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)` — upper triangle `-inf` (position `i` attends to `j ≤ i`). Cached per `(seqlen, device)`: rebuilt only when `seqlen > self._causal_mask_size` or the device changed; returns the `[:seqlen, :seqlen]` slice of the buffer. Return shape `(S, S)`. Called only from `MTPBlock.forward`.

### `MTPBlock.forward`

```python
def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
```

| In/Out | Shape | Notes |
|---|---|---|
| `prev_hidden` | `(B, S, dim)` | pre-norm trunk hidden state (or previous depth's output) |
| `target_emb` | `(B, S, dim)` | embedding of the token one position ahead |
| return | `(B, S, dim)` | residual-stream output `h′` |

Computation, verbatim structure:

```python
fused = self.proj(torch.cat([self.norm_h(prev_hidden), self.norm_e(target_emb)], dim=-1))
seqlen = fused.size(1)
attn_in = self.norm_attn(fused)
attn_out, _ = self.attn(attn_in, attn_in, attn_in,
                        attn_mask=self._get_causal_mask(seqlen, fused.device), is_causal=False)
fused = fused + attn_out
ffn_in = self.norm_ffn(fused)
return fused + self.w2(F.silu(self.w1(ffn_in)) * self.w3(ffn_in))
```

Shape walk: `cat(…, dim=-1)` → `(B, S, 2·dim)` → `proj` → `(B, S, dim)`. `attn` is **self-attention** (query = key = value = `attn_in`), `batch_first=True`, explicit `(S, S)` mask (hence `is_causal=False` — causality comes from the mask). Both `fused + attn_out` and `fused + w2(…)` are residual adds in the fused stream (pre-norm at every stage: `norm_attn`, `norm_ffn`). `F.silu` is SiLU/Swish; the FFN is SwiGLU (`w1`/`w3` gates, `w2` projects).

---

## 5.4 `MTPModule`

### `MTPModule.__init__`

```python
def __init__(self, config: dict, depth: int = 1):
```

Config keys read: `dim`, `vocab_size`. Attributes: `depth` (1-indexed depth — the head built for depth $d$ reads embedding offset $d$ and predicts $d+1$ ahead), `block: MTPBlock`, `norm: nn.RMSNorm(dim, eps=1e-6)`, `output_head: Optional[nn.Linear] = None`. The head is **not** created here — it is injected by `set_output_head` (shared-head mechanics, §5.5).

### `MTPModule.set_output_head`

```python
def set_output_head(self, head: nn.Linear) -> None:
```

Stores the reference; the head is a plain attribute assignment, so it is registered as a submodule (appears in `state_dict()` as `output_head.weight`) but is **not copied** — it is the main model's `head` tensor, shared by reference. Must be called before `forward`; until then every forward raises.

### `MTPModule.forward`

```python
def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```

| In/Out | Shape | Notes |
|---|---|---|
| `prev_hidden` | `(B, S, dim)` | must equal `target_emb.shape` exactly |
| `target_emb` | `(B, S, dim)` | embedding of shifted tokens |
| return `[0]` logits | `(B, S, vocab_size)` | `output_head(h_norm)` |
| return `[1]` `h_norm` | `(B, S, dim)` | `norm(block(…))` — feeds the next depth |

Guards, in order:
1. `if self.output_head is None: raise RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")` — this is the shared-head contract; see §5.8.
2. `if prev_hidden.shape != target_emb.shape: raise ValueError(f"Shape mismatch: {prev_hidden.shape} vs {target_emb.shape}")`.

Then `h = self.block(prev_hidden, target_emb)`, `h_norm = self.norm(h)`, return `(self.output_head(h_norm), h_norm)`. Note the norm sits **before** the head and the *normed* hidden is what chains into the next depth (the trunk's `forward_with_hidden` convention — logits on normed `h`, raw `h` in the stream).

---

## 5.5 `MultiTokenPrediction`

### `MultiTokenPrediction.__init__`

```python
def __init__(self, config: dict, main_model: nn.Module):
```

- Resolves `depth`/`mtp_weight` per §5.2.
- `model_cfg = config.get("model", config)`; builds
  `self.mtp_modules = nn.ModuleList([MTPModule(model_cfg, d + 1) for d in range(self.depth)])` — depths are 1-indexed.
- **Shared embedding:** `self.add_module("embed", main_model.embed)` — the
  same `nn.Embedding` instance, so MTP consumes the main model's embedding with zero extra parameters.
- **Shared head:** `shared_head = main_model.head`; every module gets
  `mtp.set_output_head(shared_head)`. The same `nn.Linear` (which, under `weight_tying`, already shares its weight tensor with `embed.weight`, see `models/transformer.py:Transformer.__init__`). Parameter accounting: `models/transformer.py:count_parameters` dedups by tensor `id`, so the shared embed/head are counted once and MTP adds only its block/norm weights (~7.1M).

`main_model` is held as `self.main_model` and is expected to expose `embed`, `head`, and `forward_with_hidden` (the `Transformer` class does; anchor `models/transformer.py:Transformer.forward_with_hidden`).

### `MultiTokenPrediction.forward`

```python
def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
```

Guard: `tokens.dim() < 2` → `ValueError(f"Expected (bsz, seq) tokens, got {tokens.shape}")`. Token dtype must be `torch.long` (no cast here).

Pipeline:

```python
seq_len = tokens.size(1)
main_logits, prev_h = self.main_model.forward_with_hidden(tokens)
mtp_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
for d, mtp in enumerate(self.mtp_modules):
    usable = seq_len - d - 2
    if usable <= 0:
        break
    h_in = prev_h[:, :usable]
    emb_in = self.embed(tokens[:, d + 1: d + 1 + usable])
    tgt = tokens[:, d + 2: d + 2 + usable]
    logits, hidden = mtp(h_in, emb_in)
    mtp_pairs.append((logits, tgt))
    prev_h = hidden
return main_logits, mtp_pairs
```

**Length-alignment rule:** at 0-indexed depth `d`, `usable = seq_len − d − 2`. Depth `d` predicts token `i + d + 2` from (hidden of token `i`, embedding of token `i + d + 1`) for `i ∈ [0, usable)`. Equivalently: the last `d + 2` positions of the sequence contribute no MTP target at this depth (the `+1` embedding shift and the `+2` target shift each consume one position). The chain `prev_h ← h_norm` shrinks by one position per depth, so each successive head predicts strictly later offsets with strictly shorter spans. When `usable ≤ 0` (sequence too short for the depth) the loop `break`s and fewer pairs are returned — never a crash.

| Out | Shape | Notes |
|---|---|---|
| `main_logits` | `(B, seq_len, vocab_size)` | unmodified main-model logits |
| `mtp_pairs[d]` | `(logits (B, usable_d, V), tgt (B, usable_d))` | `usable_d = seq_len − d − 2`, d 0-indexed |

### `MultiTokenPrediction.compute_loss`

```python
def compute_loss(self, main_logits: torch.Tensor, targets: torch.Tensor,
                 mtp_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Verbatim core:

```python
main_loss = F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)),
                            targets.reshape(-1), ignore_index=-100)
if not mtp_pairs:
    return main_loss, main_loss, main_loss.new_zeros(())
depth_losses: List[torch.Tensor] = []
for logits, tgt in mtp_pairs:
    if tgt.numel() == 0:
        continue
    depth_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        tgt.reshape(-1), ignore_index=-100))
mtp_loss = torch.stack(depth_losses).mean() if depth_losses else main_loss.new_zeros(())
return main_loss + self.mtp_weight * mtp_loss, main_loss, mtp_loss
```

**Loss math.** Let $V$ = vocab size, $B$ = batch, $S$ = sequence length, $\mathcal{L}_{\mathrm{CE}}$ = `F.cross_entropy` with `ignore_index=-100` (token positions labelled −100 contribute zero gradient; the loss is averaged over *non-ignored* tokens):

- $\mathcal{L}_{\mathrm{main}} = \mathcal{L}_{\mathrm{CE}}(\mathrm{logits}_{(B·S) \times V}, \mathrm{targets}_{(B·S)})$
- Per depth: $\mathcal{L}_d = \mathcal{L}_{\mathrm{CE}}(\mathrm{logits}_d, \mathrm{tgt}_d)$ over the aligned span — the `reshape(-1, V)` / `reshape(-1)` flattens exactly as the main loss, so each $\mathcal{L}_d$ is a token-mean over its `(B · usable_d)` positions.
- $\mathcal{L}_{\mathrm{mtp}} = \mathrm{mean}_d\, \mathcal{L}_d$ — **mean across depths**, not a concatenated token-mean. With canonical `mtp_depth=1` the two coincide; they diverge for `D > 1` (deeper heads have fewer tokens and are down-weighted implicitly by the depth-mean).
- $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{main}} + w \cdot \mathcal{L}_{\mathrm{mtp}}$, $w$ = `mtp_weight` (**0.3** canonical).

Empty-pair edge cases: `mtp_pairs` falsy → `mtp_loss = main_loss.new_zeros(())` (scalar 0 on the same device/dtype) and total = main. Inside the loop, a pair with `tgt.numel() == 0` is skipped; if *all* pairs are empty, `torch.stack` is bypassed and `mtp_loss` is again a zero scalar. Return tuple is always `(total, main, mtp)` in that order.

---

## 5.6 Who calls it

### Training — `training/pretrain.py`

- `training/pretrain.py:Pretrainer.__init__` — constructs
  `MultiTokenPrediction(config.model_config, raw_model)` when `mtp_depth > 0 and config.mtp_weight > 0.0`, moves it to device, counts parameters (logs `MTP enabled (depth=…, weight=…): … total / … trainable`), uses the wrapper total for μP LR scaling, and stores it as `self.mtp_wrapper` / `self.model`.
- `training/pretrain.py:Pretrainer.train_step` — MTP path: `main_logits,
  mtp_pairs = self.model(tokens)` then `total_loss, main_loss, mtp_loss = self.mtp_wrapper.compute_loss(main_logits, targets, mtp_pairs)`; the loss is divided by `gradient_accumulation_steps` before backward; the returned metrics dict carries `"loss"` (main CE) and `"mtp_loss"` (deferred-detached, `None` when no pairs).
- `training/pretrain.py:Pretrainer.save_checkpoint` — saves the wrapper's
  `mtp_modules.*` keys under the `mtp.` prefix (`mtp.` + key) next to the base model state; `head.weight` is dropped when weight-tying is on, so the `mtp_modules.<d>.output_head.weight` copy is the only head tensor on disk.
- `training/pretrain.py:Pretrainer.load_checkpoint` — when the checkpoint meta
  says `has_mtp`, extracts keys with the `mtp.` prefix, strips it (`removeprefix("mtp.")`), and `load_state_dict(…, strict=False)` into the wrapper (compiled-wrapper safe via `_orig_mod`).

### Inference — `inference/speculative.py`, `inference/generate.py`

- `inference/speculative.py:SpeculativeDecoder.__init__` — holds
  `main_model`, `mtp` (`MTPModule`), `threshold = acceptance_threshold` (default `0.8`).
- `inference/speculative.py:SpeculativeDecoder.generate_step` — main model
  scores the last token, samples `token_main` (via `models/transformer.py:Transformer._sample`), runs `main_model.forward_with_hidden(token_main, start_pos=t1_pos, use_cache=True)` to get `hidden_last`, embeds `token_main` with `main_model.embed`, then drafts `draft_logits, _ = self.mtp(hidden_last, token_main_emb)` (single-position call: `(1, 1, dim)` in, `(1, 1, V)` out). Acceptance criterion (greedy verification variant — see [DeepSeekMoE & MTP](../concepts/moe-mtp.md) for why): $p_{\mathrm{main}}(\mathrm{draft}) \ge \text{threshold} \cdot \max(p_{\mathrm{draft}}(\mathrm{draft}), 10^{-12})$.
- `inference/speculative.py:SpeculativeDecoder.generate` — prefill pass with
  `use_cache=True`, then the accept/reject loop appending `token_main` and, on acceptance, `token_draft`.
- `inference/generate.py:main` — when `--use-speculative`, builds a **standalone**
  `MTPModule(model_cfg, depth=1)`, calls `set_output_head(model.head)` (without this, the first draft step raises — see §5.8), loads `mtp.`-prefixed checkpoint weights, and passes it to `generate_interactive`.
- `inference/generate.py:generate_interactive` — optional `mtp_module`
  parameter; constructs a `SpeculativeDecoder` when provided.

### Tests

`tests/test_models.py` (`TestMTPBlock`, `TestMTPModule`, `TestMultiTokenPrediction` — forward shapes, `usable = seq - d - 2` alignment, no-head RuntimeError, shape-mismatch ValueError, loss math), `tests/test_inference.py` (speculative accept/reject incl. unset-head error), `tests/test_training.py` (MTP train step, `mtp.` checkpoint round-trip), `tests/test_utils.py` (MTP weight round-trip).

---

## 5.7 Parameter & checkpoint inventory

| Scope | Keys (prefix) | Notes |
|---|---|---|
| Wrapper state | `mtp_modules.<d>.block.{norm_h,norm_e,proj,norm_attn,attn,norm_ffn,w1,w2,w3}.*` | `attn` = `in_proj_weight`, `out_proj.weight` |
| Wrapper state | `mtp_modules.<d>.norm.weight` | final pre-head norm |
| Wrapper state | `mtp_modules.<d>.output_head.weight` | the **shared** head tensor (identical to `head.weight`); not counted twice by `count_parameters` |
| Wrapper state | `embed.weight` | the shared embedding (alias of `main_model.embed`) |
| Checkpoint (training) | `mtp.<key>` | `Pretrainer.save_checkpoint` wraps only `mtp_modules.*` keys |
| Checkpoint (inference load) | strip `mtp.` → `load_state_dict(strict=False)` | standalone `MTPModule` ignores foreign keys |

MTP adds ~7.1M parameters total (418,713,984 vs 411,632,256); active params per token remain ~185M because depth-1 reuses the trunk hidden state.

---

## 5.8 Pitfalls

- **Unset shared head → `RuntimeError`.** `MTPModule.forward` raises
  `MTPModule(depth=…): output_head not set.` until `set_output_head` is called. `MultiTokenPrediction.__init__` wires it automatically; every standalone use (`generate.py`, `SpeculativeDecoder`, tests) must call it explicitly.
- **MTP attention is torch MHA, not MLA.** `MTPBlock.attn` is
  `nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False)` with full `(B, S, dim)` Q/K/V — a deliberate simplification vs the trunk's MLA (see [MLA](../concepts/attention-and-precision.md) and `R3_mla_api.md`). No KV-lora compression, no RoPE, no cache integration in the block.
- **Explicit mask, not `is_causal`.** Causality comes from the `(S, S)` `-inf`
  upper-triangular mask (`is_causal=False`); the mask cache key is `(seqlen, device)` and rebuilds only on growth/device change.
- **Short sequences silently drop depths.** `usable = seq_len − d − 2 ≤ 0`
  breaks the loop — a `(B, 2)` input yields zero pairs and a pure-main loss, not an error. `tgt.numel() == 0` pairs are skipped inside `compute_loss`.
- **dtype is caller-owned.** No int64 cast in `models/mtp.py`; the uint32→long
  cast happens in `training/pretrain.py:Pretrainer.train_step`. Feeding float tokens fails inside `nn.Embedding`.
- **MTP off unless both gates pass.** `mtp_depth = 0` **or**
  `mtp_weight = 0.0` in `TrainingConfig` disables the whole wrapper — the `Pretrainer` then trains the bare `Transformer`.
- **`compute_loss` needs the pairs.** Passing only `(main_logits, targets)`
  yields `mtp_loss = 0` and trains main-only even though MTP weights exist; the train loop always passes `mtp_pairs` (empty list included) so this path only triggers on misuse.
- **Loss is a depth-mean, not a token-mean.** For `D > 1`, shallower (longer)
  depths do not dominate the auxiliary loss; each depth contributes equally after its own token-mean.

---

## 5.9 Check your understanding

1. At canonical `max_seq_len = 2048`, `mtp_depth = 1`: how many tokens does the
   depth-1 head predict per sequence, and at what offsets? *(Answer: `usable = 2048 − 0 − 2 = 2046`; predicts positions 2…2047 from (hidden of i, embedding of i+1), targets `tokens[:, 2:]`.)*
2. Why does `count_parameters` report ~7.1M for MTP rather than
   ~84M (`blocks + norm + head`)? *(Answer: the head and embedding are the main model's tensors shared by reference; `count_parameters` dedups by tensor id, `models/transformer.py:count_parameters`.)*
3. What happens on the first draft step of `--use-speculative` if
   `set_output_head` were removed from `inference/generate.py:main`? *(Answer: `MTPModule.forward` raises the `output_head not set` RuntimeError.)*
4. With `mtp_weight = 0.3` and `D = 2`, is the auxiliary term
   $0.3 \cdot (\mathcal{L}_1 + \mathcal{L}_2)/2$? *(Answer: yes — `torch.stack(depth_losses).mean()`, a mean across depths, scaled by 0.3.)*

---

**Related:** [DeepSeekMoE & MTP](../concepts/moe-mtp.md) ·
[Foundations & Architecture](../concepts/foundations.md) ·
[Training Pipeline](../training.md) ·
[Inference & Serving](../inference.md) ·
`R2_transformer_api.md` (`forward_with_hidden`, `_sample`, `count_parameters`) · `R1_config_schema.md` (`mtp_depth`, `mtp_loss_weight`) · `R7_training_api.md` (`Pretrainer.train_step`) · `R9_inference_api.md` (`SpeculativeDecoder`, `generate_interactive`)

## References

- [DeepSeekMoE & MTP](../concepts/moe-mtp.md) — MTP chapter (derivation, alignment algebra, speculation theory)
- [Foundations & Architecture](../concepts/foundations.md) — trunk architecture
- [Training Pipeline](../training.md) — MTP loss consumption
- [Inference & Serving](../inference.md) — speculative decoding at serving time
- [R2 — Transformer API](../references/R2_transformer_api.md) (`forward_with_hidden`, `_sample`, `count_parameters`)
- [R1 — Config Schema](../references/R1_config_schema.md) (`mtp_depth`, `mtp_loss_weight`)
- [R7 — Training API](../references/R7_training_api.md) (`Pretrainer.train_step`)
- [R9 — Inference API](../references/R9_inference_api.md) (`SpeculativeDecoder`, `generate_interactive`)

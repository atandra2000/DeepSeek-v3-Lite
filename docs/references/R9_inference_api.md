# DeepSeek-v3-Lite — R9 Inference API Reference

> Module `inference/` · cross-refs: [Inference & Serving](../inference.md), [MTP](../concepts/moe-mtp.md), [Triton Kernels](../concepts/kernels-and-ops.md), [Model Architecture](../concepts/foundations.md).

**60-second summary.** This file is the API contract for everything that runs a trained model: the two entry points in `inference/generate.py` (a pure config loader, an interactive REPL, and the CLI `main`) and the MTP draft decoder in `inference/speculative.py`. It also pins the on-model `Transformer.generate` usage pattern and the shared sampling parameter space (temperature / top-p / top-k) that all three paths funnel through `models/transformer.py:Transformer._sample`. All numbers here are API facts from source; any latency or acceptance figures are estimates — no GPU run has been executed in this repo.

**Module map.**

| Symbol | Anchor | Purpose |
|---|---|---|
| Config loader | `inference/generate.py:load_config` | YAML load + `model:` section validation |
| Interactive REPL | `inference/generate.py:generate_interactive` | Chat loop over standard/KV-cache or speculative decode |
| CLI entry | `inference/generate.py:main` | argparse → model + checkpoint + MTP head + tokenizer |
| Speculative decoder | `inference/speculative.py:SpeculativeDecoder` | Main model → draft → threshold accept/reject |
| On-model decode | `models/transformer.py:Transformer.generate` | KV-cached autoregressive sampling |
| Sampler | `models/transformer.py:Transformer._sample` | temperature + top-k + top-p core |

---

## 1. Sampling parameter space (shared contract)

Every sampling path in the repo ends at `models/transformer.py:Transformer._sample` — a `@staticmethod` so both the model and `SpeculativeDecoder` call it without an instance. It maps `(bsz, vocab_size)` logits to `(bsz, 1)` sampled token ids.

```python
@staticmethod
def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor
```

**Semantics, in application order:**

1. **`temperature == 0.0`** → early return `logits.argmax(dim=-1, keepdim=True)` — greedy, deterministic. The comparison is an *exact* equality: `1e-9` takes the stochastic path (effectively greedy in output, but not in code path).
2. **`temperature > 0`** → `logits / temperature` before softmax; higher temperature flattens the distribution, lower sharpens it.
3. **`top_k > 0`** → keep only the top-`k` logits (`k` clamped to the vocab size via `min(top_k, logits.size(-1))`), mask the rest to `-inf` **before** softmax. `top_k == 0` disables the ceiling. `top_k = 1` degenerates to greedy.
4. **softmax** → probabilities.
5. **`top_p < 1.0`** (nucleus) → sort probabilities descending; a token is *removed* when the cumulative mass strictly **before** it already exceeds `top_p` (`remove = (cumulative - sorted_probs) > top_p`) — the token whose own mass crosses the threshold is **kept**. Zero the removed, renormalize over the remainder (`.clamp(min=1e-10)` guards a fully-emptied nucleus), then `torch.multinomial`.
6. **`top_p == 1.0`** → multinomial over the full distribution.

Negative temperature is **not** checked inside `_sample`; the only guard in the repo is in `models/transformer.py:Transformer.generate` (`temperature < 0.0` → `ValueError`).

**Defaults per entry point:**

| Entry point | temperature | top_p | top_k | Notes |
|---|---|---|---|---|
| `Transformer.generate` | `1.0` | `0.9` | `0` | full sampling space exposed |
| CLI `--temperature` / `--top_p` | `0.7` | `0.9` | — (no flag) | interactive defaults |
| `SpeculativeDecoder.generate` / `.generate_step` | `1.0` | forced `1.0` | forced `0` | temperature-only; nucleus/ceiling disabled in the speculative path |

---

## 2. `inference/generate.py`

### 2.1 `load_config`

```python
def load_config(path: str) -> dict
```

One-line purpose: load a YAML training/inference config and validate its shape.

- **Contract:** `FileNotFoundError` if `path` does not exist; `ValueError("Config must be a dict with a 'model' section")` if the parsed YAML is not a dict or lacks a `model` key. Returns the raw dict unchanged.
- **Callers:** `inference/generate.py:main` (passes `cfg["model"]` straight to `Transformer(model_cfg)`), and tests.

### 2.2 `generate_interactive`

```python
@torch.inference_mode()
def generate_interactive(model: torch.nn.Module, tokenizer, args, mtp_module: Optional[MTPModule] = None) -> None
```

One-line purpose: blocking chat REPL — one user turn per loop iteration, full conversation history kept in `messages`.

- **`args` contract (duck-typed, produced by `main`'s `ArgumentParser`):** reads `args.use_speculative`, `args.acceptance_threshold`, `args.device`, `args.max_new_tokens`, `args.temperature`, `args.top_p`. No type annotation — any namespace with these attributes works.
- **Decode path selection:** if `mtp_module is not None` **and** `args.use_speculative`, build `SpeculativeDecoder(model, mtp_module, acceptance_threshold=args.acceptance_threshold)` once (prints `"Speculative decoding enabled."`); otherwise fall back to `model.generate(...)`. The decoder is created once, not per turn — its cache is pre-filled per call by its own `generate`.
- **Loop:** print banner; `eos_id = tokenizer.eos_token_id`; each turn:
  - `input()` → `"/exit"` breaks, `"/clear"` empties `messages` (KV-cache note: `model.generate` and `decoder.generate` both call `reset_cache` internally, so clearing chat history alone is cache-safe), blank input skips.
  - `EOFError` / `KeyboardInterrupt` → `"Exiting."`, break.
  - `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(args.device)`.
  - Speculative: `decoder.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, eos_token_id=eos_id)` — **`args.top_p` is ignored here** (see §3.3).
  - Standard: `model.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, eos_token_id=eos_id)` — `top_k` stays at its default `0`.
  - New tokens = `output_ids[0, input_ids.shape[1]:]`; `tokenizer.decode(new_tokens, skip_special_tokens=True)`; the assistant turn is appended to `messages` so the next turn has full context.
- **Device/dtype:** `input_ids` are moved to `args.device`; the model must already live there. `@torch.inference_mode()` disables autograd for the whole session.
- **Callers:** `inference/generate.py:main`.

### 2.3 `main`

```python
def main()
```

One-line purpose: CLI entry that assembles model + checkpoint + optional MTP head + tokenizer, then drops into the REPL. Run as `python -m inference.generate --config configs/pretrain_a100_422m.yaml --checkpoint <dir-or-file>`.

- **Hard dependency guard:** raises `RuntimeError` if `transformers` is not installed — the pure-decode helpers (`load_config`, and the module itself) import fine without it, but `AutoTokenizer` is required for the interactive entry.
- **Arguments (argparse):**

| Flag | Default | Meaning |
|---|---|---|
| `--config` | required | YAML path (see `inference/generate.py:load_config`) |
| `--checkpoint` | required | checkpoint **directory** or a `model_step_N.safetensors` file path |
| `--max_new_tokens` | `512` | generation budget |
| `--temperature` | `0.7` | sampling temperature |
| `--top_p` | `0.9` | nucleus threshold (standard path only) |
| `--use_speculative` | off (`store_true`) | enable MTP draft decode |
| `--acceptance_threshold` | `0.8` | draft accept threshold |
| `--device` | auto (`cuda` if available else `cpu`) | run device |

- **Model flow:** `Transformer(model_cfg).to(args.device)`; `.eval()`. Config filename note: the canonical `configs/pretrain_a100_422m.yaml` is the *filename* — the model it builds is 411.6M params (418.7M with MTP).
- **Checkpoint flow:**
  - `ckpt_dir` = `args.checkpoint` if it is a directory, else its parent.
  - Step resolution: directory → `utils/checkpoint.py:CheckpointManager.latest_step()` (first complete checkpoint, newest first); `RuntimeError("No checkpoints found in ...")` if none. File path → parse `int(stem.split("_")[-1])` from the `model_step_N` stem, falling back to `latest_step()` on `ValueError`.
  - `utils/checkpoint.py:CheckpointManager.load(model, step, device=args.device, strict=False)`. `strict=False` is load-bearing: with `weight_tying: true` the checkpoint has **no `head.weight`** (shared storage with `embed.weight`), so a strict load would fail on the missing key.
- **MTP attach flow (only when `--use_speculative`):**
  1. `MTPModule(model_cfg, depth=1).to(args.device)`; `.eval()`.
  2. `mtp_module.set_output_head(model.head)` — **required before any draft forward**; without it `models/mtp.py:MTPModule.forward` raises `RuntimeError("... output_head not set")`.
  3. Load `mtp.`-prefixed keys from `model_step_{step}.safetensors` (safetensors `load_file(..., device=args.device)`), stripping the `"mtp."` prefix, `strict=False`. If the checkpoint has no `mtp.*` keys: `print("[warn] No MTP weights in checkpoint; draft head is uninitialised.")` and continue — the decode still runs but acceptance collapses toward zero.
- **Tokenizer:** `cfg["data"]["tokenizer_path"]`, default `deepseek-ai/deepseek-coder-v2-lite`, via `AutoTokenizer.from_pretrained`.
- **Callers:** nothing in-repo — it is the process entry point (`if __name__ == "__main__": main()`).

---

## 3. `inference/speculative.py` — `inference/speculative.py:SpeculativeDecoder`

```python
class SpeculativeDecoder:
    """MTP-based speculative decoder: main model predicts T1, draft predicts T2; verify and accept or fall back."""
```

One-line purpose: depth-1 MTP speculative decoding — one draft token proposed per main-model step, accepted by a deterministic threshold rule (a deliberate simplification of Metropolis–Hastings; see [MTP](../concepts/moe-mtp.md) §11).

### 3.1 `inference/speculative.py:SpeculativeDecoder.__init__`

```python
def __init__(self, main_model: nn.Module, mtp_module: MTPModule, acceptance_threshold: float = 0.8)
```

- **Attributes:** `self.main_model`, `self.mtp`, `self.threshold` (public, read by tests).
- **Contract:** `main_model` is typed `nn.Module` but is used as a `Transformer` — the decoder calls `main_model.forward_with_hidden(...)`, `main_model.embed(...)`, and `main_model.reset_cache()`. Passing any other module type fails at call time, not construction. `mtp_module` must have had `set_output_head` called (see §5.1) before the first `generate_step`.
- **Callers:** `inference/generate.py:generate_interactive`; tests (`tests/test_inference.py`).

### 3.2 `inference/speculative.py:SpeculativeDecoder.generate_step`

```python
@torch.inference_mode()
def generate_step(self, last_token: torch.Tensor, start_pos: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, bool]
```

One-line purpose: one speculative step — sample the main token, propose a greedy draft, return accept/reject.

**Returns** `(token_main, token_draft, was_accepted)` — both tensors shape `(bsz,)` (callers pass batch 1), `was_accepted` a Python `bool`.

**Algorithm (position-by-position):**

1. `main_logits = self.main_model(last_token, start_pos=start_pos, use_cache=True)` — trunk forward #1; writes position `start_pos` to the KV cache.
2. `token_main = Transformer._sample(main_logits[:, -1, :], temperature, top_p=1.0, top_k=0).squeeze(0)` — main token sampled with the caller's temperature; **nucleus and top-k are forced off**. `temperature=0` degenerates to argmax (greedy main).
3. `_, hidden = self.main_model.forward_with_hidden(token_main.unsqueeze(0), start_pos=t1_pos, use_cache=True)` with `t1_pos = start_pos + 1` — trunk forward #2 (via `models/transformer.py:Transformer.forward_with_hidden`), needed because the draft conditions on the hidden state of `token_main` (the MTP contract $x_{t+2} \leftarrow (h_{t+1}, e_{t+1})$). This is why the main model runs **twice per step**; the cache write at `t1_pos` duplicates the next step's prefill write (see [MTP](../concepts/moe-mtp.md) §13 for the double-write analysis).
4. `token_main_emb = self.main_model.embed(token_main.unsqueeze(-1))` — shared embedding, so the draft conditions on the same token representation the main model committed.
5. `draft_logits, _ = self.mtp(hidden_last, token_main_emb)` — one `models/mtp.py:MTPModule.forward` pass; `token_draft = draft_probs.argmax(dim=-1)` — **the draft is always greedy**, regardless of `temperature`.
6. **Acceptance rule** — compares raw (temperature-1, unscaled) probabilities of the draft token:
   ```python
   p_main_of_draft = main_probs[0, token_draft[0]].item()
   p_draft_of_draft = draft_probs[0, token_draft[0]].item()
   accept = p_main_of_draft >= self.threshold * max(p_draft_of_draft, 1e-12)
   ```
   The `1e-12` floor guards spurious `0 >= τ·0` acceptance when the draft's argmax probability underflows to zero. Temperature shapes *which* token the main model proposes but never the acceptance test itself.

**Temperature semantics summary:** temperature-only sampling for the main token; greedy draft; acceptance on temperature-1 distributions. No top-k/top-p anywhere on this path.

**Callers:** `SpeculativeDecoder.generate` only.

### 3.3 `inference/speculative.py:SpeculativeDecoder.generate`

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             eos_token_id: Optional[int] = None) -> torch.Tensor
```

One-line purpose: end-to-end speculative decode over a prompt; returns `(bsz, prompt_len + n_generated)` on the input device.

**Shape contract:** output length is in `[prompt_len + 1, prompt_len + max_new_tokens]` — at least one token always emitted (the first main token is appended unconditionally). `input_ids` must be `torch.long` on the model's device (no uint32 cast here; `Transformer.forward` performs that cast only inside its own boundary).

**Loop invariants:**

1. Cache hygiene: `self.main_model.reset_cache()` if the model exposes it (it does — `models/transformer.py:Transformer.reset_cache`), then prefill `self.main_model(output, start_pos=0, use_cache=True)`. `generate` owns cache reset; `generate_step` alone does not.
2. While `n_generated < max_new_tokens`:
   - `start_pos = output.size(1) - 1` — position of the last *emitted* token; `generate_step` writes the new token at `start_pos + 1`, so the cache stays one ahead of the emitted sequence.
   - `last_token = output[:, -1:]`; call `generate_step`.
   - Append `token_main`; `n_generated += 1`; break if it equals `eos_token_id`.
   - If `was_accepted` **and** `n_generated < max_new_tokens`: append `token_draft`; `n_generated += 1`; break if the draft is EOS.
3. The main token is emitted unconditionally (accept/reject only gates the *extra* draft token) — this is the "fall back" half of the class docstring.

**Notable absences:** no `top_p`/`top_k` parameters, and no `temperature < 0` guard — a negative temperature is passed through to `_sample` unvalidated (only `Transformer.generate` validates).

**Callers:** `inference/generate.py:generate_interactive` (speculative branch); tests (`tests/test_inference.py` — construction, cache reset, EOS, length bounds, threshold sweep).

---

## 4. On-model generation — `Transformer.generate`

```python
@torch.inference_mode()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
             top_p: float = 0.9, top_k: int = 0, eos_token_id: Optional[int] = None) -> torch.Tensor
```

One-line purpose: the standard (non-speculative) KV-cached decode loop; full API contract in [R2](../references/R2_transformer_api.md). Usage pattern as exercised here:

- **Guards:** `temperature < 0.0` → `ValueError` (the only negative-temperature guard in the repo).
- **Cache + mode lifecycle:** `self.reset_cache()`; save `was_training`; `self.eval()`; restore `self.train()` on exit — safe to call on a training-mode model. KV-cache isolation between consecutive `generate` calls is guaranteed by the unconditional `reset_cache` (see `tests/test_models.py` `test_generate_kv_cache_isolation`).
- **Prefill:** `models/transformer.py:Transformer.forward` over the full prompt (`start_pos=0, use_cache=True`); `next_logits = prefill_logits[:, -1, :]`.
- **Decode loop:** per step — `next_token = self._sample(next_logits, temperature, top_p, top_k)` (shape `(bsz, 1)`); `output = torch.cat([output, next_token], dim=1)`; EOS via a `finished` boolean mask with an early break when `finished.all()` (batch-aware — unlike the speculative path, which is single-sequence); break when `output.size(1) >= self.max_seq_len` (hard cap from `model_cfg["max_seq_len"]`, canonical 2048); then `self.forward(next_token, start_pos=prompt_len + step, use_cache=True)` — decode position is always `prompt_len + step`.
- **Device/dtype:** `input_ids` on the model device (`torch.long`; `models/transformer.py:Transformer.forward` also accepts `uint32` mmap'd shard ids and casts at the boundary); output lives on `input_ids.device`. `@torch.inference_mode()` wraps the whole loop.
- **Batch semantics:** `bsz > 1` works (finished-mask + batched `_sample`); the loop still exits on `finished.all()`, so slow sequences can be cut short by fast ones.
- **Callers:** `inference/generate.py:generate_interactive` (standard branch); `tests/test_models.py` and `tests/test_inference.py` (greedy determinism, negative-temperature raise, train-mode restore, `max_seq_len` cap, top-k/top-p shapes).

---

## 5. Supporting API used by the entry points

### 5.1 `MTPModule` attach contract (`models/mtp.py`, full API in [R5](../references/R5_mtp_api.md))

```python
def __init__(self, config: dict, depth: int = 1)          # models/mtp.py:MTPModule.__init__
def set_output_head(self, head: nn.Linear) -> None        # models/mtp.py:MTPModule.set_output_head
def forward(self, prev_hidden: torch.Tensor, target_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

- `output_head` starts `None`; `models/mtp.py:MTPModule.set_output_head(model.head)` must run before the first forward or `MTPModule.forward` raises `RuntimeError(f"MTPModule(depth={self.depth}): output_head not set.")`.
- Wiring order in `main()`: `set_output_head` **before** `load_state_dict` — the shared head's weights are stored under the main model's `head.*` keys (never under `mtp.*`), so `strict=False` tolerates the absent `output_head.weight` key.
- Shapes: `prev_hidden`, `target_emb` both `(bsz, seq, dim)`; returns `(draft_logits, hidden)` with `draft_logits` of shape `(bsz, seq, vocab_size)`; `SpeculativeDecoder` uses only `[:, -1, :]` of the logits and discards the hidden.
- Callers: `inference/generate.py:main` (attach), `models/mtp.py:MultiTokenPrediction.__init__` (shared-head wiring), tests.

### 5.2 `CheckpointManager` (`utils/checkpoint.py`, full API in [R8](../references/R8_utils_api.md))

```python
def load(self, model: torch.nn.Module, step: int, device: str = "cuda",
         optimizer: Optional[torch.optim.Optimizer] = None, strict: bool = True) -> dict
def latest_step(self) -> Optional[int]
```

- Files: `model_step_N.safetensors` (+ `optim_step_N.pt`, `meta_step_N.json` for training). `load` returns the metadata dict.
- `main()` passes `device=args.device` (default `"cuda"` — the CLI must override it on CPU) and `strict=False` because of weight tying (§2.3). `latest_step()` returns the newest *complete* checkpoint's step or `None`.
- Callers: `inference/generate.py:main`; `training/pretrain.py` (save/load/resume).

---

## 6. Pitfalls

- **Negative temperature is only guarded on one path.** `Transformer.generate` raises; `SpeculativeDecoder.generate` and bare `_sample` do not — a negative `temperature` silently produces `logits / negative`, flipping the ranking.
- **`temperature == 0.0` is an exact-equality test.** `1e-9` takes the stochastic path (`logits / 1e-9` → effectively argmax but still `torch.multinomial`).
- **Speculative mode ignores `--top_p`.** The CLI's `0.9` default applies only to the standard path; the decoder hardcodes `top_p=1.0, top_k=0`.
- **`output_head not set`.** Any `MTPModule` used for drafting must have `set_output_head(model.head)` first; the error surfaces on the first draft forward, not at construction.
- **`[warn] No MTP weights in checkpoint` is a warning, not an error.** The draft head proposes near-random tokens, acceptance collapses, and speculative decode silently degenerates to one token per trunk-forward — check the warning if speculative mode seems slower than standard decode. (With an untrained MTP head this repo *always* warns — there is no trained checkpoint yet, and every acceptance figure is an estimate.)
- **Two trunk forwards per speculative step.** The win requires acceptance to clear roughly 50% — otherwise the doubled per-step cost dominates; see [MTP](../concepts/moe-mtp.md) §11.5 for the break-even math.
- **Cache hygiene is caller-owned outside `generate`.** `Transformer.generate` and `SpeculativeDecoder.generate` both reset the cache; a bare `model(tokens, use_cache=True)` in a notebook does not — stale prefixes leak between prompts (see [Inference & Serving](../inference.md) "stale context" pitfall).
- **`uint32` tokens are accepted only through `Transformer.forward`** (and `forward_with_hidden`), which casts to `long` at the boundary; `SpeculativeDecoder.generate` passes them straight to `nn.Embedding` and will fail on a non-Long dtype.
- **Device mismatch fails inside the embedding**, not at the API surface: `input_ids` and the model must share a device; the CLI's `--device` moves the model and the tokenizer tensors to the same target.
- **Weight tying means `head.weight` is absent from checkpoints** — the expected `strict=False` load path in `main()`; a strict load fails on the missing key.

---

## References
- [Inference & Serving](../inference.md) — sampling theory, KV-cache decode lifecycle, speculative decode walkthrough, the same `generate_step` trace with shapes.
- [MTP](../concepts/moe-mtp.md) — MTP loss/target algebra, acceptance math, the position-by-position cache-consistency trace (§13), threshold-rule bias analysis (§11.4–11.5).
- [Model Architecture](../concepts/foundations.md) — the trunk these entry points drive.
- [R2 — Transformer API](../references/R2_transformer_api.md) — `Transformer`, `TransformerBlock`, `forward`/`forward_with_hidden`/`reset_cache` contracts.
- [R5 — MTP API](../references/R5_mtp_api.md) — `MTPModule` / `MultiTokenPrediction` full contracts.
- [R8 — Utils API](../references/R8_utils_api.md) — `CheckpointManager` save/load/resume/atomicity.
- [Triton Kernels](../concepts/kernels-and-ops.md) — fused MLA/MoE kernels behind the SDPA/stacked inference paths; canonical config falls back to `moe_dispatch="stacked"` (register cap at `moe_inter_dim=384`).


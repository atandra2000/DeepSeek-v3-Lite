# Inference — Generation and Speculative Decoding

## A Comprehensive Technical Reference

> **Covers**: `inference/generate.py`, `inference/speculative.py`, and `Transformer.generate()` — from checkpoint to tokens.

> **Read this if** you're debugging generation, KV cache, or speculative decode. **Skip if** you're training only → [training.md](training.md).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Complexity Analysis](#complexity-analysis)
3. [Training vs Inference](#training-vs-inference)
4. [KV Cache — Prefill and Decode](#kv-cache--prefill-and-decode)
5. [Standard Generation](#standard-generation)
6. [Sampling — Temperature, Top-K, Top-P](#sampling--temperature-top-k-top-p)
7. [Interactive REPL and CLI](#interactive-repl-and-cli)
8. [Checkpoint Loading](#checkpoint-loading)
9. [Speculative Decoding](#speculative-decoding)
10. [Programmatic API](#programmatic-api)
11. [Inference Memory Budget](#inference-memory-budget)
12. [Known Limitations](#known-limitations)
13. [Debugging Checklist](#debugging-checklist)
14. [Appendix A — Decode timeline](#appendix-a--decode-timeline)
15. [Appendix B — FAQ](#appendix-b--faq)
16. [Appendix C — Glossary](#appendix-c--glossary)
17. [References](#references)

---

## Abstract

Inference in DeepSeek-V3-Lite is **single-GPU autoregressive generation** with MLA KV-cache acceleration. Two paths exist:

1. **Standard decode** — `Transformer.generate()` with temperature / top-k / top-p sampling
2. **Speculative decode** — `SpeculativeDecoder` using the MTP draft head for ~1.7–1.9× throughput

Entry point:

```bash
python -m inference.generate --config configs/pretrain_a100_422m.yaml --checkpoint checkpoints/pretrain_a100
```

### Inference as conditional generation

Training learns $p_\theta(x_t \mid x_{<t})$ for all positions in parallel. Inference **samples** one token at a time:

$$
x_{t+1} \sim p_\theta(\cdot \mid x_1, \ldots, x_t)
$$

**Without KV cache:** $O(T^2)$ per sequence. **With MLA cache:** amortised $O(T)$ per generated token after prefill.

**Prerequisites:** [foundations.md](foundations.md) §11, [MLA.md](MLA.md) §KV Cache Management.

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

### MLA cache compression

Per layer, per token cached bytes (BF16):

$$
\text{bytes} = 2 \times (\text{kv_lora_rank} + \text{qk_rope_head_dim}) = 2 \times (192 + 24) = 432 \text{ bytes}
$$

At $S = 2048$, $L = 18$: $432 \times 2048 \times 18 \approx 15.9$ MB — vs $\sim 150$ MB for standard MHA at same dims.

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

---

## KV Cache — Prefill and Decode

MLA caches two tensors per layer (`models/mla.py`):

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

**Mask optimisation:** When `seqlen == 1`, causal mask is `None` — the single query cannot attend to future positions that do not exist in the cache slice.

**Why prefill is fast:** One forward over `prompt_len` positions amortises attention. Each decode step computes one new query against all cached keys.

---

## Standard Generation

`Transformer.generate()` in `models/transformer.py`:

```python
@torch.inference_mode()
def generate(self, input_ids, max_new_tokens=512, temperature=1.0,
             top_p=0.9, top_k=0, eos_token_id=None) -> torch.Tensor
```

### Algorithm (line-by-line)

| Step | Code | Notes |
|---|---|---|
| 1 | `reset_cache()` | Clears MLA caches on all 18 layers |
| 2 | `self.eval()` | Disables dropout (none in this repo, but safe) |
| 3 | `prefill_logits = forward(output, start_pos=0, use_cache=True)` | Full prompt forward |
| 4 | `next_logits = prefill_logits[:, -1, :]` | Only last position sampled |
| 5 | Loop `max_new_tokens` times | See below |
| 6 | `restore training mode` if was training | Preserves caller state |

**Decode loop body:**

```python
next_token = self._sample(next_logits, temperature, top_p, top_k)
output = torch.cat([output, next_token], dim=1)
if eos_token_id and all finished: break
if output.size(1) >= max_seq_len: break
decode_logits = self.forward(next_token, start_pos=prompt_len + step, use_cache=True)
next_logits = decode_logits[:, -1, :]
```

**`start_pos` invariant:** After prefill of length `prompt_len`, decode step `step` writes position `prompt_len + step`. Off-by-one here causes cache corruption and gibberish output.

### `forward_with_hidden` (MTP / speculative only)

Returns `(logits, h)` where `h` is **pre-final-norm** trunk hidden. Used by `SpeculativeDecoder.generate_step` to feed MTP after main model commits token $t_1$.

---

## Sampling — Temperature, Top-K, Top-P

`Transformer._sample(logits, temperature, top_p, top_k)`:

### Temperature

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

| $\tau$ | Behaviour |
|---|---|
| $\to 0$ | Greedy argmax (deterministic) |
| $1.0$ | Native model distribution |
| $> 1$ | Flatter (higher entropy) |

### Top-k

Keep only the $k$ largest logits; mask others to $-\infty$. `top_k=0` disables.

### Top-p (nucleus)

(Holtzman et al., 2020) Sort probabilities descending; find smallest set $V^{(p)}$ with $\sum_{i \in V^{(p)}} p_i \geq p$. Sample only from $V^{(p)}$.

### Implementation order

```
temperature scaling → top-k mask → softmax → top-p filter → multinomial
```

**Source:** `models/transformer.py:Transformer._sample`.

---

## Interactive REPL and CLI

`generate_interactive()` in `inference/generate.py`:

```
Commands:
  /exit   — quit
  /clear  — reset conversation history (does NOT reset KV cache)

Flow:
  user input → chat template → tokenise → generate → decode → append to history
```

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
| `--temperature` | 0.7 | Passed to generate |
| `--top_p` | 0.9 | **Parsed but not forwarded in REPL** |
| `--use_speculative` | off | Requires MTP weights in checkpoint |
| `--acceptance_threshold` | 0.8 | Draft acceptance bar |
| `--device` | auto | `cuda` if available else `cpu` |

### `main()` bootstrap sequence

1. `load_config(args.config)` — validates `model` key exists
2. `Transformer(model_cfg).to(device)` + `eval()`
3. `CheckpointManager.load(model, step, strict=False)`
4. If `--use_speculative`: build `MTPModule`, load `mtp.*` keys, `set_output_head(model.head)`
5. `AutoTokenizer.from_pretrained(tokenizer_path)`
6. `generate_interactive(...)`

**`transformers` is optional** for imports — only required for CLI tokenizer. Tests import decode helpers without it.

---

## Checkpoint Loading

```python
ckpt_mgr = CheckpointManager(ckpt_dir)
step = ckpt_mgr.latest_step()
ckpt_mgr.load(model, step, device=device, strict=False)
```

**Weight tying:** `head.weight` may be absent from safetensors — restored via shared `embed.weight` storage.

**MTP weights (speculative mode):**

```python
from safetensors.torch import load_file
state = load_file("model_step_N.safetensors", device=device)
mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
mtp_module.load_state_dict(mtp_state, strict=False)
mtp_module.set_output_head(model.head)  # re-share after load
```

---

## Speculative Decoding

`SpeculativeDecoder` (`inference/speculative.py`) uses the MTP head trained with $\lambda = 0.3$ as a **draft model** without a separate checkpoint.

### `generate_step` algorithm

```python
def generate_step(last_token, start_pos):
    main_logits = main_model(last_token, start_pos=start_pos, use_cache=True)
    main_probs = softmax(main_logits[:, -1, :])
    token_main = argmax(main_probs)

    t1_pos = start_pos + 1
    _, hidden = main_model.forward_with_hidden(token_main.unsqueeze(0), start_pos=t1_pos, use_cache=True)
    token_main_emb = main_model.embed(token_main.unsqueeze(-1))
    draft_logits, _ = mtp(hidden[:, -1:], token_main_emb)
    draft_probs = softmax(draft_logits[:, -1, :])
    token_draft = argmax(draft_probs)

    accept = p_main[draft] >= threshold * max(p_draft[draft], 1e-12)
    return token_main, token_draft, accept
```

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

Measured acceptance depends on MTP training quality and threshold.

### `generate()` outer loop

1. Prefill: `main_model(output, start_pos=0, use_cache=True)`
2. While `n_generated < max_new_tokens`:
   - `generate_step` from last token
   - Append `token_main`
   - If accepted, append `token_draft` (two tokens per iteration)
3. Greedy only — `temperature` parameter is accepted but not used in speculative path

**Cache coherence:** `forward_with_hidden` at `start_pos=t1_pos` must match the position after `token_main` is committed to the cache.

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
# ... load mtp weights ...
decoder = SpeculativeDecoder(model, mtp, acceptance_threshold=0.8)
output = decoder.generate(input_ids, max_new_tokens=100)
```

---

## Inference Memory Budget

For interactive generation ($B=1$, prompt 512, generating 256 tokens):

| Component | Approximate |
|---|---|
| Model weights (BF16) | 0.84 GB |
| KV cache at max_seq_len=2048 | ~16 MB |
| Activations (single-token decode) | negligible |
| CUDA context | 2–14 GB |

**Total:** ~1–2 GB for weights + overhead — fits on consumer GPUs for inference-only (no optimizer state).

Use `estimate_model_memory_gb(..., inference=True)` from `utils/memory.py` for analytical budget.

---

## Known Limitations

1. **`--top_p` not forwarded in REPL** — `generate_interactive` calls `model.generate` without `top_p`. Use programmatic API for nucleus sampling in REPL.
2. **No batch inference** — `SpeculativeDecoder` assumes batch size 1.
3. **No continuous batching** — serving-at-scale patterns not implemented.
4. **Triton MoE kernel not validated for decode** — use `moe_dispatch="stacked"` for inference.
5. **`transformers` required for CLI** — optional import; tests import helpers without it.
6. **Speculative path is greedy** — no temperature / top-p in `SpeculativeDecoder.generate`.
7. **Simplified speculative sampling** — threshold accept rule, not optimal Metropolis–Hastings (Leviathan et al.). Future work: full speculative sampling for higher acceptance at equal correctness.

---

## Debugging Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Gibberish after first token | `start_pos` off-by-one | Trace `prompt_len + step` in `generate()` |
| Same output every run | `temperature=0` or broken sampling | Check `_sample` args |
| Speculative never accepts | Random MTP weights | Train with `mtp_depth=1`, load `mtp.*` keys |
| OOM on long context | Cache exceeds `max_seq_len` | Truncate prompt or raise `max_seq_len` in config |
| Missing `head.weight` on load | Weight tying | Expected — `strict=False` |
| Stale context in REPL | KV cache not reset | Call `model.reset_cache()` between sessions |

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

**Q: Must I call `reset_cache` between REPL turns?** `/clear` resets messages but not KV cache. For independent prompts, create a new session or call `model.reset_cache()`.

**Q: Can I use speculative without MTP checkpoint?** The draft head will be random — acceptance rate near zero.

**Q: Why greedy in speculative path?** Simplifies acceptance check; MTP was trained with teacher forcing on ground-truth prefixes.

---

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| Prefill | Process full prompt in one forward |
| Decode | Single-token forwards with cache |
| `start_pos` | Cache write offset |
| `acceptance_threshold` | Min $p_{\text{main}} / p_{\text{draft}}$ ratio (0.8) |
| `generate_step` | One speculative decode iteration |

---

## References

- `inference/generate.py`, `inference/speculative.py`
- `models/transformer.py` — `generate()`, `_sample()`
- [mtp.md](mtp.md) — MTP theory
- [transformer.md](transformer.md) — wiring
- [MLA.md](MLA.md) — KV cache internals
- [utils.md](utils.md) — checkpoint format

## Worked Example — Prefill + 3 Decode Steps

**Setup:** $B=1$, prompt `[101, 2345, 678]` (length 3), greedy decode, `max_new_tokens=2`.

| Step | `input_ids` | `start_pos` | Cache positions written | Sampled token |
|---|---|---|---|---|
| Prefill | `[101,2345,678]` | 0 | 0,1,2 | (logits at pos 2 used) |
| Decode 0 | `[9012]` | 3 | 3 | argmax of last logits |
| Decode 1 | `[5555]` | 4 | 4 | argmax |

**Tensor shapes:**
- Prefill logits: $(1, 3, 100018)$
- Decode logits: $(1, 1, 100018)$ each step

**Common bug:** Using `start_pos=2` on first decode step (off by one from prompt length 3).

---

## Comparison — Standard vs Speculative

| Property | `Transformer.generate` | `SpeculativeDecoder.generate` |
|---|---|---|
| Sampling | temp / top-k / top-p | Greedy only |
| Tokens per forward | 1 | 1–2 |
| MTP required | No | Yes (trained weights) |
| KV cache | Main model only | Main model only |
| Batch size | $B \ge 1$ | $B = 1$ |
| Acceptance rule | N/A | $p_{\text{main}} \ge 0.8 \cdot p_{\text{draft}}$ |

---

## Integration with Chat Template

`generate_interactive` uses:

```python
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)
```

The template inserts role tokens (`user`, `assistant`) per DeepSeek-Coder format. **Tokenizer path** comes from `data.tokenizer_path` in YAML (`deepseek-ai/deepseek-coder-v2-lite`).

Multi-turn: each turn appends to `messages`; KV cache grows with full history unless you call `model.reset_cache()`.

<!-- docs:verified 2026-07-31 · 5a880d2 -->

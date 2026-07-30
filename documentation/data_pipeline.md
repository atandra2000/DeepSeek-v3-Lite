# Data Pipeline — Tokenization, Sharding, and Loading

## A Comprehensive Technical Reference

> **Covers**: How DeepSeek-V3-Lite obtains its 8.4B-token training corpus, from raw HuggingFace datasets to packed `uint32` shards consumed by `PretrainDataset`.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Why a Shared Pipeline?](#why-a-shared-pipeline)
3. [The DeepSeek Shim](#the-deepseek-shim)
4. [Tokenizer Deep Dive](#tokenizer-deep-dive)
5. [Pipeline Stages](#pipeline-stages)
6. [Data Mixture](#data-mixture)
7. [Shard Format](#shard-format)
8. [CLI Reference](#cli-reference)
9. [PretrainDataset Consumption](#pretraindataset-consumption)
10. [Environment Variables](#environment-variables)
11. [Learning Exercises](#learning-exercises)
12. [Appendix A — FAQ](#appendix-a--faq)
13. [Appendix B — Glossary](#appendix-b--glossary)
14. [References](#references)

---

## Abstract

DeepSeek-V3-Lite does **not** vendor its own data pipeline. `data/prepare_data.py` is a thin shim that configures the **universal 8.0B-token LLM pipeline** in `LLM/shared_data/` with DeepSeek-specific tokenizer settings (`vocab_size=100,018`). The output is a directory of memory-mapped `uint32` token shards consumed directly by `PretrainDataset` in `training/pretrain.py`.

---

## Why a Shared Pipeline?

All seven LLM projects in CoreProjects share:

- The same source datasets and mixture weights
- The same cleaning and deduplication logic
- The same shard packing format

Only the **tokenizer** differs per project. This means:

- One place to fix bugs in download/clean/dedup
- Fair cross-project comparisons (same text, different token IDs)
- Bit-identical shards except for token values

---

## The DeepSeek Shim

`data/prepare_data.py` does three things:

1. **Path injection** — adds project root and `LLM/` to `sys.path`
2. **Config materialisation** — writes `data/data_config.yaml` with DeepSeek overrides:
   ```python
   tokenizer.name = "deepseek-coder-v2-lite"
   vocab_size = 100_018
   eos_token_id = 100_017
   pad_token_id = 100_016
   ```
3. **Delegation** — calls `shared_data.prepare_data.run_pipeline(...)`

```bash
python3 data/prepare_data.py --stage pretrain
```

---

## Tokenizer Deep Dive

| Property | Value | Why it matters |
|---|---|---|
| Name | `deepseek-ai/deepseek-coder-v2-lite` | Public HF tokenizer |
| `vocab_size` | **100,018** | Must match `model.vocab_size` and embedding dim |
| `eos_token_id` | 100,017 | Appended at document boundaries |
| `pad_token_id` | 100,016 | Padding (rarely used in packed pretrain) |
| `byte_fallback` | yes | Unusual vs GPT-2/LLaMA — handles arbitrary bytes |

**The byte_fallback footgun:** Tokens 0–255 may represent raw bytes. The embedding table must have exactly 100,018 rows. Using 100,000 or 102,400 will silently misalign or crash.

**Verification:**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-v2-lite")
assert len(tok) == 100018
```

---

## Pipeline Stages

`run_pipeline` executes four subprocess stages (crash-isolated):

### Stage 1 — Download

Fetches HuggingFace datasets listed in `mixture.yaml`. Each source downloaded to a cache directory under `LLM_DATA_ROOT`.

### Stage 2 — Clean

- Quality filter: per-source `min_chars` / `max_chars`
- **SHA-256 deduplication** across the union of all sources
- Normalises whitespace, strips empty documents

### Stage 3 — Tokenize

- BPE encode with project tokenizer
- **EOS appended** at every document boundary
- Output: intermediate token files per source

### Stage 4 — Pack

- Concatenate tokenized documents into **50M-token shards**
- Storage dtype: **`uint32`** (4 bytes/token, supports vocab up to 4B)
- Atomic write: temp file → `os.rename`
- `manifest.json`: per-source provenance, SHA-256 checksums, token counts

Output directory (422M default): `data/pretrain_chinchilla/`

---

## Data Mixture

Canonical weights live in `LLM/shared_data/config/mixture.yaml` (sum to 1.0):

| Source | Weight | Content type |
|---|---|---|
| FineWeb-Edu | 0.50 | High-quality web text |
| FineWeb | 0.20 | General web crawl |
| the-stack-python | 0.15 | Code |
| OpenMathInstruct-2 | 0.10 | Math reasoning |
| arxiv | 0.05 | Scientific papers |

**Total budget:** 8.0B tokens (Chinchilla-optimal for ~422M params ≈ 20 tokens/param).

Always read the canonical YAML — do not rely on copied weights in docs.

Override per run: `--mixture PATH`

---

## Shard Format

### Why uint32 on disk, int64 in training

| Stage | dtype | Reason |
|---|---|---|
| Disk shards | uint32 | 4 bytes/token; vocab 100018 < 2³² |
| `PretrainDataset` | uint32 mmap | Memory-efficient random access |
| `train_step` boundary | int64 | `nn.Embedding` requires Long indices |

Casting at the train boundary (`training/pretrain.py:train_step`) keeps 8.4B tokens at ~32 GB mmap'd instead of ~64 GB.

```
data/pretrain_chinchilla/
  shard_000.bin    # torch-serialised 1D uint32 tensor, ~50M tokens
  shard_001.bin
  ...
  manifest.json
```

Each shard is a contiguous 1D tensor loaded with:

```python
torch.load(path, weights_only=True, mmap=True)
```

**No per-sample headers** — samples are created by sliding windows in `PretrainDataset`:

```python
start = idx * max_seq_len
chunk = data[start : start + max_seq_len + 1]
x, y = chunk[:-1], chunk[1:]
```

Documents are separated by EOS tokens in the packed stream. A window may span multiple documents — the model learns to predict across EOS boundaries (standard GPT-style pretraining).

---

## CLI Reference

```bash
python3 data/prepare_data.py --stage pretrain \
    [--mixture PATH] \
    [--data-config PATH] \
    [--data-root PATH] \
    [--source NAME] \
    [--skip-download] \
    [--skip-clean] \
    [--skip-tokenize] \
    [--skip-pack]
```

| Flag | Use case |
|---|---|
| `--skip-download` | Re-run from cached raw data |
| `--skip-clean` | Re-tokenize only |
| `--skip-tokenize` | Re-pack only |
| `--source NAME` | Process single mixture source |

---

## PretrainDataset Consumption

See [training.md](training.md) for full details.

| Layout | Detection | Sample count |
|---|---|---|
| Single file | `os.path.isfile(data_path)` | `(N-1) // max_seq_len` |
| Sharded dir | `shard_*.bin` glob | same formula on total tokens |

**Cross-shard windows:** Binary search `_locate` finds shard, stitches with `torch.cat`. Slower than single-file but necessary at 8B scale.

**LRU shard cache:** Not implemented in current code — all shards mmap'd at init. At 8B tokens (~64 GB uint32), ensure sufficient RAM or use streaming loader improvements.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `LLM_DATA_ROOT` | Shared cache root for all LLM projects |
| `HF_HOME` / `HF_DATASETS_CACHE` | HuggingFace download cache |
| `TOKENIZERS_PARALLELISM=false` | Set in launch script to avoid fork warnings |

---

## Learning Exercises

1. **Token count audit:** Load one shard, count EOS tokens, estimate documents per shard.
2. **Vocab spot-check:** Decode tokens 100000–100017 — observe byte_fallback and special tokens.
3. **Mixture \nablation:** Re-run with `--source fineweb_edu` only, compare loss curve.
4. **Shard integrity:** Verify `manifest.json` SHA-256 against on-disk shard hashes.
5. **Window boundary:** Find a sample index where the window crosses a shard boundary; verify `__getitem__` stitches correctly.

---

## Appendix A — FAQ

**Q: Can I use GPT-2 tokenizer?** Only for the 1650 smoke config (`vocab_size=50257`). The 422M config requires DeepSeek tokenizer.

**Q: Where is DATA_PIPELINE.md?** The universal pipeline README is at `LLM/shared_data/README.md`.

**Q: How long does full prep take?** Hours to days depending on bandwidth and CPU cores. Use `--skip-*` flags for incremental reruns.

---

## Appendix B — Glossary

| Term | Meaning |
|---|---|
| `mixture.yaml` | Source weights for the 8B corpus |
| `data_config.yaml` | Per-project tokenizer + shard settings |
| `manifest.json` | Shard metadata and checksums |
| `uint32` | 4-byte token storage dtype |
| `byte_fallback` | BPE fallback to raw byte tokens |

---



---

## Mathematical Properties of Packed Pretraining

### Next-token alignment

For chunk `chunk = data[start : start + S + 1]`:

```
tokens  = chunk[:-1]   # positions 0..S-1
targets = chunk[1:]    # positions 1..S   (shifted by 1)
```

At position $t$, the model predicts $x_{t+1}$ from context $x_{\le t}$. This is **teacher forcing** — targets are ground truth, not model predictions.

### Cross-document boundaries

Packed streams concatenate documents with EOS between them. A window $[start, start+S)$ may span:

```
... doc_A ... EOS | doc_B ...
         ^^^^^^^^^ window boundary
```

The model learns to predict the first token of doc_B after EOS — standard GPT-style pretraining. No special "document start" token beyond EOS.

### Sample count formula

$$
N_{\text{samples}} = \left\lfloor \frac{N_{\text{tokens}} - 1}{S} \right\rfloor
$$

Final $(N \bmod S)$ tokens are **dropped** — not padded. Padding would introduce fake targets.

---

## Shard Format — Byte-Level Detail

Each `shard_*.bin` is a PyTorch-serialised 1D `torch.uint32` tensor:

```python
torch.load(path, weights_only=True, mmap=True)
```

| Property | Value |
|---|---|
| dtype | uint32 (4 bytes/token) |
| endianness | platform-native (usually little-endian) |
| shard size | ~50M tokens (~200 MB/shard) |
| total shards | ~168 for 8.4B tokens |

**Why uint32?** Vocab 100,018 < $2^{32}$. Halves storage vs int64.

**Why mmap?** `PretrainDataset` memory-maps shards — OS pages in data on demand. Multiple workers share read-only pages.

---

## Cross-Shard Window Algorithm

When sample `idx` needs tokens spanning shard boundary:

```python
def __getitem__(self, idx):
    start = idx * self.max_seq_len
    needed = self.max_seq_len + 1
    pieces = []
    cursor = start
    while len(concatenated) < needed:
        shard_idx, offset = self._locate(cursor)
        take = min(needed - len_so_far, shard_size - offset)
        pieces.append(shard[offset:offset+take])
        cursor += take
```

`_locate` uses `bisect` on `shard_offsets` — $O(\log K)$ for $K$ shards.

**Performance:** Cross-shard windows are rare (~1/50M per shard boundary) but slower due to `torch.cat`.

---

## Mixture Design Rationale

| Source | Weight | Rationale |
|---|---|---|
| FineWeb-Edu | 0.50 | High-quality educational web text — backbone |
| FineWeb | 0.20 | Diversity + long-tail knowledge |
| the-stack-python | 0.15 | Code structure for DeepSeek-Coder tokenizer |
| OpenMathInstruct-2 | 0.10 | Reasoning patterns |
| arxiv | 0.05 | Technical/scientific register |

**Always read canonical** `LLM/shared_data/config/mixture.yaml` — weights may change.

### Stratified sampling intuition

If source $i$ has weight $w_i$ and total budget $T$:

$$
T_i = w_i \cdot T
$$

Documents are sampled proportional to $w_i$ after cleaning/dedup. Under-represented sources (arxiv at 5%) still contribute $\approx 420$M tokens — substantial for a 422M model.

---


## `PretrainDataset.__getitem__` — Step-by-Step

Source: `training/pretrain.py:108-125`.

For sample index `idx`:

1. `start = idx * max_seq_len`
2. Need `max_seq_len + 1` contiguous tokens (input + shifted target)
3. **Single-file layout:** slice `data[start : start+needed]`
4. **Sharded layout:** while tokens remain, `_locate(cursor)` → `(shard_idx, offset)`; append slice; advance cursor
5. Return `(chunk[:-1], chunk[1:])` — classic next-token prediction pair

**Why `+1` token?** Causal LM aligns input position $t$ with target $t+1$; last input token's target is the extra token.

---

## `_locate(global_idx)` — Binary Search on Shards

Shard metadata:
- `shard_offsets[i]` — starting global index of shard `i`
- `shard_sizes[i]` — number of uint32 tokens in shard `i`

`bisect.bisect_right(shard_offsets, global_idx) - 1` finds the shard in $O(\log N_{shards})$.

**Bounds:** `global_idx < 0` or `>= _total_tokens` raises `IndexError` — prevents silent wrap on corrupt indices.

---

## Sample Count Formula — Edge Cases

$$
N_{\text{samples}} = \left\lfloor \frac{N_{\text{tokens}} - 1}{S} \right\rfloor
$$

- The `-1` accounts for the extra target token in each window
- Partial tail shorter than `S+1` is **dropped** — not padded
- `drop_last=True` in DataLoader drops partial **batches**, not samples

**422M:** $N_{tokens} \approx 8.4 \times 10^9$, $S=2048$ → $\approx 4.1 \times 10^6$ samples/epoch.

---

## Tokenizer Contract (DeepSeek Coder V2 Lite)

| Property | Value |
|---|---|
| `vocab_size` | 100,018 |
| Special tokens | Includes EOS at index 100,017 |
| Chat template | Used by `inference/generate.py` REPL |
| Embedding rows | Must equal `model.vocab_size` exactly |

Changing tokenizer without regenerating shards **and** updating `vocab_size` breaks training at embedding lookup.

---

## Data Mixture Rationale (8.4B Chinchilla Pack)

The universal pipeline under `LLM/shared_data/` blends:

| Source class | Role |
|---|---|
| Web / general | Broad language prior |
| Code | Syntax, APIs, reasoning patterns |
| Math | Symbolic manipulation |
| Books / long-form | Discourse structure |

Exact ratios live in `shared_data/config/mixture.yaml`. This project consumes the **pre-mixed** uint32 shards — you do not re-blend at train time.

---

## Operational Runbook

```bash
# Full pipeline (hours, hundreds of GB download)
python data/prepare_data.py

# Verify shards exist
ls data/pretrain_chinchilla/shard_*.bin | head

# Quick train connectivity test
python -m training.pretrain --config configs/pretrain_1650_2m.yaml --no-compile
```

**CI / dev:** use tiny generated shards in `tests/` fixtures — never require 8.4B tokens in unit tests.



## Pipeline Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Wrong vocab_size in config | Embedding crash at step 0 | Regenerate `data_config.yaml` |
| Incomplete download | Missing source in manifest | Re-run without `--skip-download` |
| Dedup too aggressive | Tiny corpus | Check clean stage logs |
| Disk full during pack | Partial shard | Delete incomplete shard, re-pack |
| HF auth for tokenizer | Download fail | Use cached tokenizer or mirror |

---

## Storage Budget

| Item | Size |
|---|---|
| Raw downloads | ~500 GB+ (cached under `LLM_DATA_ROOT`) |
| Cleaned text | ~200 GB |
| Tokenized uint32 | ~32 GB (8.4B × 4 bytes) |
| Per-project copy | Same shards shared across LLM projects |

Only token IDs differ per project (tokenizer); raw text is shared via `shared_data`.

---

## Learning Exercises (Extended)

6. **Compression ratio:** Tokenize 1 MB of raw text — compute bytes/token ratio.
7. **EOS frequency:** Count EOS token (100017) per shard — estimate documents per 50M tokens.
8. **Boundary test:** Find `idx` where window crosses shard — verify `__getitem__` matches manual stitch.
9. **Mixture \nablation:** Train 1000 steps on single-source data — compare loss to full mixture.
10. **Manifest audit:** Verify SHA-256 in `manifest.json` matches on-disk shards.


## References

- `data/prepare_data.py` — project shim
- `LLM/shared_data/README.md` — universal pipeline
- `LLM/shared_data/config/mixture.yaml` — canonical mixture
- [training.md](training.md) — `PretrainDataset` loader

## Tokenizer Contract — Special Tokens

From `data/prepare_data.py`:

| Constant | Value | Role |
|---|---|---|
| `DEEPSEEK_VOCAB_SIZE` | 100,018 | Embedding rows |
| `DEEPSEEK_EOS_TOKEN_ID` | 100,017 | Document boundary |
| `DEEPSEEK_PAD_TOKEN_ID` | 100,016 | Padding (rare in pretrain) |

**byte_fallback:** Bytes 0–255 are valid tokens. Never assume all tokens are printable Unicode.

### Verification script

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-v2-lite")
assert len(tok) == 100018
assert tok.eos_token_id == 100017
```

---

## `_locate` — Binary Search Detail

```python
def _locate(self, global_idx: int) -> Tuple[int, int]:
    lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
    return lo, global_idx - self.shard_offsets[lo]
```

`shard_offsets[i]` = cumulative token count before shard `i`.

**Complexity:** $O(\log K)$ per `_locate` call; cross-shard windows call it in a loop until `needed` tokens collected.

**Edge cases tested:** `test_locate_out_of_range_raises`, `test_sharded_cross_boundary`.

---

## Data Prep Runbook

### First-time full pipeline

```bash
export LLM_DATA_ROOT=/path/to/shared_data   # optional
python3 data/prepare_data.py --stage pretrain
```

Stages: download → clean → tokenize → pack. Each stage is subprocess-isolated — failure in tokenize does not corrupt packed shards.

### Incremental reruns

| Goal | Flags |
|---|---|
| Re-download one source | `--source NAME` (no skip flags) |
| Re-tokenize after tokenizer change | `--skip-download --skip-clean` |
| Re-pack only | `--skip-download --skip-clean --skip-tokenize` |

### Output verification

```bash
ls data/pretrain_chinchilla/shard_*.bin | wc -l   # expect ~168 shards
python -c "
import torch; from pathlib import Path
p = sorted(Path('data/pretrain_chinchilla').glob('shard_*.bin'))[0]
t = torch.load(p, weights_only=True, mmap=True)
print(t.dtype, t.numel())
"
# torch.uint32 ~50000000
```

---

## Pipeline Failure Modes (Extended)

| Failure | Symptom | Fix |
|---|---|---|
| Wrong vocab in config | Embedding crash step 0 | Regenerate `data_config.yaml` |
| HF auth failure | Download stage error | `huggingface-cli login` |
| Disk full during pack | Partial shard | Delete incomplete shard, re-pack |
| Mixture weights ≠ 1.0 | Pipeline warning / error | Fix `mixture.yaml` |
| Stale manifest | SHA mismatch | Re-run pack stage |

## Token Budget Derivation

Chinchilla-optimal unique tokens for 422M params:

$$
N_{\text{tokens}} \approx 20 \times N_{\text{params}} = 20 \times 4.22 \times 10^8 \approx 8.44 \times 10^9
$$

Pipeline targets **8.0–8.4B** tokens after cleaning/dedup. Multi-epoch training (512k steps × 65k tokens/step ≈ 33.5B exposures) is intentional — rare sources (arxiv, math) benefit from repeated exposure.

### Shard count estimate

$$
\text{shards} \approx \frac{8.4 \times 10^9}{50 \times 10^6} \approx 168
$$

Each shard ~200 MB uint32 on disk.

---

## `prepare_data.py` Shim Architecture

```
DeepSeek-v3-Lite/data/prepare_data.py
    → writes data/data_config.yaml (vocab 100018)
    → calls shared_data.prepare_data.run_pipeline()
        → download / clean / tokenize / pack
```

**Shared corpus:** Raw text cached under `LLM_DATA_ROOT` — multiple LLM projects can share downloads; only token IDs differ per tokenizer.

---

## Learning Exercises (Advanced)

11. **Shard boundary:** Find `idx` where `idx * max_seq_len` lands exactly on shard edge; verify `__getitem__` matches manual concat.
12. **EOS rate:** Count token 100017 per shard; estimate documents per 50M tokens.
13. **Compression:** Measure bytes/token for packed uint32 vs raw UTF-8 on 1 MB sample.
14. **Mixture ablation:** Train 1000 steps on single-source data; compare loss to full mixture.
15. **Manifest audit:** Verify SHA-256 in `manifest.json` matches on-disk shards.

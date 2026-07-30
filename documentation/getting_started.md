# Getting Started — From Zero to a Running DeepSeek-V3-Lite

> **Purpose:** Onboard a strong ML student who has never seen this repo. You will learn *what* the project is, *why* each design choice exists, and *how* to verify your understanding with smoke tests — before diving into component chapters.

---

## Table of Contents

1. [What Problem Does This Project Solve?](#what-problem-does-this-project-solve)
2. [The DeepSeek-V3 Architecture in One Page](#the-deepseek-v3-architecture-in-one-page)
3. [Canonical Numbers — 422M Config](#canonical-numbers--422m-config)
4. [Mental Model — Three Execution Modes](#mental-model--three-execution-modes)
5. [Recommended Reading Order](#recommended-reading-order)
6. [Environment Setup](#environment-setup)
7. [Quick Smoke Test (CPU)](#quick-smoke-test-cpu)
8. [Quick GPU Smoke Test](#quick-gpu-smoke-test)
9. [Full Training Run (A100)](#full-training-run-a100)
10. [Interactive Inference](#interactive-inference)
11. [How to Read the Codebase](#how-to-read-the-codebase)
12. [Common Pitfalls — Theory and Fixes](#common-pitfalls--theory-and-fixes)
13. [Learning Exercises](#learning-exercises)
14. [FAQ](#faq)
15. [References](#references)

---

## What Problem Does This Project Solve?

Modern frontier LLMs (GPT-4, Claude, DeepSeek-V3) are **black boxes** — you can call an API, but you cannot inspect how Multi-Head Latent Attention compresses KV cache, how aux-loss-free MoE routing avoids expert collapse, or how Multi-Token Prediction enables speculative decoding.

**DeepSeek-V3-Lite** is a **pedagogical reproduction**: the full DeepSeek-V3 architecture at ~422M parameters, implemented in raw PyTorch with no HuggingFace Trainer abstraction. Every line is meant to be read.

### What you will learn

| Topic | Why it matters | Doc |
|---|---|---|
| **MLA** | 10× KV-cache compression via low-rank latent + absorption | [MLA.md](MLA.md) |
| **MoE** | Sparse FFN capacity without proportional compute | [moe.md](moe.md) |
| **MTP** | Denser training signal + inference draft head | [mtp.md](mtp.md) |
| **μP + Chinchilla** | Stable hyperparameter transfer across scales | [training.md](training.md) |
| **Triton kernels** | Optional fused paths for production throughput | [triton_kernels.md](triton_kernels.md) |

### What this project is NOT

- Not a production chatbot (422M is tiny by industry standards)
- Not a distributed training framework (single GPU only)
- Not a drop-in replacement for `transformers.AutoModel`

It **is** a complete, trainable, testable implementation you can run on one A100 in ~15 hours.

---

## The DeepSeek-V3 Architecture in One Page

```
                    ┌─────────────────────────────────────┐
  Token IDs ───────►│  Embedding (V=100018, d=768)        │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │  Layer 0–1 (DENSE)      │                         │
         │  RMSNorm → MLA → +      │  SwiGLU FFN (I=1536)    │
         └─────────────────────────┼─────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │  Layer 2–17 (MoE)       │                         │
         │  RMSNorm → MLA → +      │  DeepSeekMoE            │
         │                         │   top-4 of 20 experts   │
         │                         │   + 1 shared expert     │
         └─────────────────────────┼─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  RMSNorm → LM Head (weight-tied)      │
                    └──────────────┬──────────────────────┘
                                   │
                              Logits (B,S,V)

  Training-only parallel path:
  MTPModule(hidden_t, embed_{t+1}) → predicts token_{t+2}
```

**Key insight:** Attention (MLA) and FFN (MoE) are **independent innovations** glued by the standard pre-norm residual stack. Understanding each in isolation, then reading [architecture.md](architecture.md), gives you the full picture.

**Prerequisites:** If RMSNorm, RoPE, or SwiGLU are unfamiliar, read [foundations.md](foundations.md) first.

---

## Canonical Numbers — 422M Config

These numbers appear everywhere. Memorise the orders of magnitude; exact values live in `configs/pretrain_a100_422m.yaml`.

| Quantity | Value | Formula / note |
|---|---|---|
| Parameters | ~422M | MoE dominates (~70%) |
| Training tokens | 8.4B | Chinchilla: $20 \times N_{\text{params}}$ |
| Layers | 18 | 2 dense + 16 MoE |
| Hidden dim $d$ | 768 | |
| Heads $H$ | 12 | MLA per head: 48 nope + 24 rope QK, 64 V |
| KV latent rank $R$ | 192 | MLA compression |
| MoE experts | 20 routed + 1 shared | Top-4 activated |
| Expert width | 384 | `moe_inter_dim` |
| Vocab $V$ | 100,018 | DeepSeek tokenizer |
| Seq length $S$ | 2,048 | |
| Micro-batch | 8 | |
| Grad accum | 4 | Effective batch = 32 seqs |
| Tokens / optim step | 65,536 | $32 \times 2048$ |
| Optim steps | 512,000 | |
| LR (μP-scaled) | ~8.07e-4 | From ref 6e-4 @ 757M params |
| Wall time (A100) | 13–15 h | Target MFU 35–40% |
| Training VRAM | ~35 GB | grad_checkpoint + compile |

---

## Mental Model — Three Execution Modes

The same codebase serves three purposes. Confusing them causes the most common bugs.

### Mode 1 — CPU correctness (development)

```bash
python -m pytest tests/ -q
```

- No CUDA, no Triton, no data download required
- Tests use `small_cfg` (dim=64, vocab=1024)
- **Purpose:** Prove math invariants (MLA absorption, MoE bias, MTP alignment)

### Mode 2 — GPU smoke (validation)

```bash
python scripts/smoke_forward.py
python scripts/microbench_a100.py
python scripts/step_time_a100.py
```

- Builds full 422M model on GPU
- One forward/backward, VRAM measurement, throughput
- **Purpose:** Confirm hardware headroom before a 15-hour run

### Mode 3 — Production training

```bash
python3 data/prepare_data.py --stage pretrain   # once
bash scripts/launch_a100.sh
```

- 8.4B-token corpus, 512K steps, checkpointing, WandB
- **Purpose:** Train a real checkpoint

| Mode | `use_cache` | `torch.compile` | Data |
|---|---|---|---|
| CPU tests | False | No | Random / tiny fixture |
| GPU smoke | False | Optional | Random |
| Training | **False** | Yes | 8.4B shards |
| Inference | **True** | No | Prompt tokens |

---

## Recommended Reading Order

### Track A — Theory first (recommended for learning)

| Step | Doc | Time | You will understand |
|---|---|---|---|
| 0 | [foundations.md](foundations.md) | 2–3 h | Transformer basics, RoPE, Chinchilla, μP |
| 1 | [architecture.md](architecture.md) | 1 h | How components connect |
| 2 | [MLA.md](MLA.md) | 3–4 h | KV compression + absorption trick |
| 3 | [moe.md](moe.md) | 2–3 h | Aux-loss-free routing |
| 4 | [mtp.md](mtp.md) | 1–2 h | Multi-token prediction |
| 5 | [transformer.md](transformer.md) | 1 h | Wiring + generation |
| 6 | [training.md](training.md) | 2 h | Full train loop |
| 7 | [inference.md](inference.md) | 1 h | KV cache + speculative decode |

### Track B — Code first (recommended for debugging)

| Step | Start here | Question answered |
|---|---|---|
| 1 | `models/transformer.py` | What is the forward path? |
| 2 | `models/mla.py` | How does attention work? |
| 3 | `models/moe.py` | How does routing work? |
| 4 | `training/pretrain.py` | How is loss computed? |
| 5 | `tests/test_models.py` | What invariants must hold? |

### Track C — Operations only

| Doc | When |
|---|---|
| [configs.md](configs.md) | Tuning YAML |
| [scripts.md](scripts.md) | Launch / benchmark |
| [data_pipeline.md](data_pipeline.md) | Preparing tokens |
| [utils.md](utils.md) | Checkpoints / VRAM |
| [testing.md](testing.md) | Writing tests |

---

## Environment Setup

### Dependencies

```bash
cd LLM/DeepSeek-v3-Lite
pip install torch safetensors pyyaml tqdm pytest
# Optional: transformers (inference CLI), wandb (logging), triton (GPU kernels)
```

### Hardware expectations

| Hardware | What works |
|---|---|
| Mac / Linux CPU | All `pytest` tests |
| GTX 1650 4GB | `configs/pretrain_1650_2m.yaml` smoke train |
| A100 80GB | Full 422M Chinchilla run |

### Environment variables (common)

```bash
export TOKENIZERS_PARALLELISM=false   # avoid fork warnings in DataLoader
export ENABLE_TRITON_KERNELS=1      # only when benchmarking Triton paths
export WANDB_PROJECT=deepseek-v3-lite-a100
```

---

## Quick Smoke Test (CPU)

```bash
python -m pytest tests/ -q --ignore=tests/test_moe_triton.py::TestMoeTritonKernelGPU
```

**Expected:** All tests pass in ~1–3 minutes on a modern laptop.

**What this proves:**
- MLA SDPA and manual paths agree (absorption correctness)
- MoE gate bias is a buffer, not a Parameter
- MTP shared head and alignment logic
- Checkpoint atomic save/load with MTP prefix
- Triton env-var guard forces PyTorch fallback

**If tests fail:** Read the failing test name in [testing.md](testing.md) — each test documents an invariant.

---

## Quick GPU Smoke Test

```bash
python scripts/smoke_forward.py          # builds 422M, one forward
python scripts/microbench_a100.py        # VRAM estimate vs measured
python scripts/step_time_a100.py         # ms/step, MFU
```

### Interpreting microbench output

```
estimated peak   ≈ 30.5 GB   # from utils/memory.py formulas
measured peak    ≈ 33-35 GB # torch.cuda.max_memory_allocated
```

**Theory:** The gap is CUDA allocator overhead + `torch.compile` workspace spikes. If measured > 72 GB on an 80GB card, reduce `micro_batch_size` or `max_seq_len` before launching training.

### Interpreting step_time output

```
median ms/step ≈ 270-320 ms
MFU ≈ 35-40%
```

**MFU** (Model FLOPs Utilisation) = fraction of peak Tensor Core throughput used. Below 25% usually means memory-bound (MoE dispatch, activation recomputation).

---

## Full Training Run (A100)

### Step 1 — Prepare data (~8.4B tokens)

```bash
python3 data/prepare_data.py --stage pretrain
```

Output: `data/pretrain_chinchilla/shard_*.bin` (uint32 tokens). See [data_pipeline.md](data_pipeline.md).

### Step 2 — Verify GPU headroom

```bash
python scripts/microbench_a100.py
```

### Step 3 — Launch

```bash
bash scripts/launch_a100.sh
# or manually:
python -u training/pretrain.py --config configs/pretrain_a100_422m.yaml
```

### Step 4 — Monitor

```bash
tail -f checkpoints/pretrain_a100/train.log
# step=  4000 | loss=2.85 | ppl=17.3 | lr=7.9e-04 | tps=128000
```

### Resume from checkpoint

```bash
python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume 4000
```

---

## Interactive Inference

```bash
python -m inference.generate \
  --config configs/pretrain_a100_422m.yaml \
  --checkpoint checkpoints/pretrain_a100 \
  --use_speculative
```

**Theory:** Standard decode generates one token per main-model forward. Speculative decode uses the MTP head to **draft** a second token, verified by the main model — potentially 2 tokens per forward. See [inference.md](inference.md) and [mtp.md](mtp.md).

---

## How to Read the Codebase

### File priority map

```
models/transformer.py   ← start here (wiring)
models/mla.py           ← hardest, most important
models/moe.py           ← routing invariants
models/mtp.py           ← training + speculative bridge
training/pretrain.py    ← train loop
tests/test_models.py    ← executable specification
```

### Config → code routing

YAML keys are consumed at different depths. `Pretrainer` stores the **entire** YAML in `TrainingConfig.model_config`; sub-modules call `config.get("model", config)`:

| YAML key | Consumer | Effect |
|---|---|---|
| `model.n_dense_layers` | `TransformerBlock` | Layers 0–1 dense, 2–17 MoE |
| `model.attn_impl` | `MultiHeadLatentAttention` | sdpa / manual / triton |
| `model.mtp_depth` | `Pretrainer` | Wrap with `MultiTokenPrediction` |
| `training.mup_lr` | `Pretrainer.__init__` | LR scaling |
| `training.nan_guard` | `train()` loop | Rollback on NaN streak |

Full reference: [configs.md](configs.md).

---

## Common Pitfalls — Theory and Fixes

| Mistake | Symptom | Root cause | Fix |
|---|---|---|---|
| `vocab_size` ≠ 100018 | Shape error at embed | Embedding rows must match tokenizer | Use DeepSeek tokenizer everywhere |
| `use_cache=True` in training | Silent wrong loss / NaN | Cache breaks full-sequence backward | `use_cache=False` in `train_step` |
| Forget `reset_cache()` | Garbage generation | Stale KV from prior prompt | Call before each `generate()` |
| MoE aux loss instead of bias | Expert collapse | Gradient fights task loss | Read [moe.md](moe.md) §Why Not Aux Loss |
| Triton without env var | No speedup, warning once | `enforce_triton_env_var` guard | `ENABLE_TRITON_KERNELS=1` |
| μP before MTP wrap | Wrong LR | Param count excludes MTP | Scaling after wrap in `Pretrainer` |
| Load ckpt with `strict=True` | Missing `head.weight` | Weight tying drops duplicate key | `strict=False` (default) |

---

## Learning Exercises

1. **Parameter audit:** Run `Pretrainer` init with `log_per_component_params: true`; verify MoE > 65% of params.
2. **MLA trace:** Set `attn_impl: manual` vs `sdpa` in a tiny config; confirm identical outputs (`test_sdpa_and_manual_agree`).
3. **MoE routing histogram:** Log `gate.get_expert_counts()` over 1000 steps; confirm no expert gets 0% load.
4. **MTP alignment:** Manually verify `usable = S - depth - 1` for depth=1, S=2048.
5. **Chinchilla check:** Compute $512000 \times 65536$ total token exposures vs 8.4B unique tokens — how many epochs?
6. **Speculative accept rate:** Log `was_accepted` in `SpeculativeDecoder.generate_step`; relate to threshold 0.8.

---

## FAQ

**Q: I only have a Mac. Can I learn this project?**
A: Yes. All correctness tests run on CPU. You cannot train 422M locally, but you can read every line and run `small_cfg` forwards.

**Q: How does this compare to nanoGPT / minGPT?**
A: Those teach classic GPT. This teaches **modern** DeepSeek-V3 mechanisms (MLA, aux-loss-free MoE, MTP) that nanoGPT omits.

**Q: Should I read the paper or the docs first?**
A: Read [foundations.md](foundations.md) + [architecture.md](architecture.md), then the DeepSeek-V3 paper (arXiv:2412.19437) with this repo open. The docs explain *this implementation*; the paper explains *the full 671B system*.

**Q: Where do I ask questions in code?**

| Question | Start here |
|---|---|
| "How does attention work?" | `models/mla.py` + [MLA.md](MLA.md) |
| "How does routing work?" | `models/moe.py` + [moe.md](moe.md) |
| "How is loss computed?" | `models/mtp.py:compute_loss` + [training.md](training.md) |
| "How do checkpoints work?" | `utils/checkpoint.py` + [utils.md](utils.md) |
| "What do tests verify?" | [testing.md](testing.md) |

---

## References

- [foundations.md](foundations.md) — prerequisite theory
- [architecture.md](architecture.md) — system map
- [README.md](README.md) — documentation index
- DeepSeek-V3 paper — arXiv:2412.19437
- `configs/pretrain_a100_422m.yaml` — canonical recipe

## Day-1 Checklist

| Step | Command | Success criterion |
|---|---|---|
| 1 | `pip install torch safetensors pyyaml tqdm pytest` | No import errors |
| 2 | `python -m pytest tests/ -q` | All CPU tests pass |
| 3 | `python scripts/smoke_forward.py` (GPU) | `(B,S,V)` logits shape |
| 4 | `python scripts/microbench_a100.py` (GPU) | measured < 72 GB |
| 5 | Read [foundations.md](foundations.md) §1–6 | Can explain CE loss + RoPE |

---

## Loss Curve — What to Expect

422M Chinchilla training on mixed corpus (rough guide, not guarantees):

| Step | Loss (nats) | PPL | Phase |
|---|---|---|---|
| 0–2k | 10–11 | ~50k | Warmup, random init |
| 10k | 4–5 | ~150 | Learning subword structure |
| 100k | 2.8–3.2 | ~16–25 | Mid-training |
| 400k+ | 2.3–2.7 | ~10–15 | Late cosine tail |

**Red flags:** loss flat after 50k steps (data path wrong), sudden spike to NaN (LR / shard corruption), PPL stuck > 100 at 200k (vocab mismatch).

---

## Checkpoint Inspection

```python
from safetensors.torch import load_file
state = load_file("checkpoints/pretrain_a100/model_step_4000.safetensors")
print(len(state), "keys")
print([k for k in state if k.startswith("mtp.")][:5])
```

Expect ~hundreds of keys; `head.weight` may be absent when `weight_tying: true`.

---

## Extending the Project — Safe Order

1. **Change depth/width** — edit YAML, run `pytest`, then `microbench`
2. **Add data source** — edit `mixture.yaml`, re-run `prepare_data.py`
3. **New attention path** — add test in `test_models.py` first, then implement
4. **Triton kernel** — set `ENABLE_TRITON_KERNELS=1`, compare against stacked in `test_moe_triton.py`

Never skip step 1 (tests) when touching `mla.py`, `moe.py`, or `mtp.py`.

<!-- docs:verified 2026-07-31 · 5a880d2 -->

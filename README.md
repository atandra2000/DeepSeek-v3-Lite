<<<<<<< HEAD
# DeepSeek-v3-Lite
=======
# DeepSeek-V3-Lite — Chinchilla-Scale Reproduction

A from-scratch reimplementation of the DeepSeek-V3 architecture in PyTorch,
**sized for Chinchilla-style training runs on a single A100 80GB**:

| Config | Parameters | Chinchilla-20 tokens | Wall time (A100) | Peak VRAM | Status |
|---|---|---|---|---|---|
| `configs/pretrain_800m.yaml`     | 757M | 15.1B  | ~9 days  | ~14 GB  | ready (microbench + dry-run needed first) |

BF16 forward, `F.scaled_dot_product_attention` attention, `torch.compile`,
zero custom CUDA kernels. The 800M config is the **maximum recommended single-A100 tier** —
validated end-to-end on CPU smoke tests, ready for the A100 launch.

![Architecture Overview](assets/architecture_overview.png)

> **Status:** Single A100 80GB SXM, BF16, no custom CUDA. Architecture and
> training code complete; pre-training has not started yet.

---

## Why Chinchilla scale?

The DeepSeek-V3 paper describes a 671B-parameter model trained on 14.8T
tokens — a compute-optimal ratio of ~22 tokens/param. Reproducing that
end-to-end is impossible on one GPU. We instead follow Hoffmann et al.'s
**Chinchilla scaling law** and target a small model that is *actually
compute-optimal* for the hardware we have:

| Quantity | Value | Source |
|---|---|---|
| **Parameters** | 757,226,496 (~757M) | This config |
| **Chinchilla-optimal tokens** | 15.1B (20:1) | Hoffmann et al. 2022 |
| **Compute-upper tokens** | 75.7B (100:1) | Same |
| **Sequence length** | 1 024 | Fits one A100 80GB with grad-ckpt |
| **Peak VRAM (bs=4, seq=1 024, grad-ckpt)** | ~14 GB | `utils/memory.py` estimate |
| **Hardware** | 1 × NVIDIA A100 80GB SXM | — |

The architecture follows the DeepSeek-V3 paper verbatim (MLA, MoE, MTP,
aux-loss-free balancing) at a larger scale. Every component is exercised
end-to-end on a real model — we are not stubbing or faking the
algorithmic ideas.

---

## Architecture at a Glance

```
Input tokens (vocab = 14 336)
    │
    ▼
  Embedding  (14 336 × 1024)
    │
    ├─ Layers 0–1: Dense Transformer Blocks ──────────────────┐
    │      MLA  →  SwiGLU FFN (no experts)                     │
    │                                                           │
    ├─ Layers 2–23: MoE Transformer Blocks ── × 22 ────────────┤
    │      MLA  →  DeepSeekMoE FFN                              │
    │                │                                          │
    │                ├─ 1 Shared Expert  (always active)        │
    │                └─ 16 Routed Experts  (top-2 per token)    │
    │                                                           │
    └─ RMSNorm  →  Linear head  →  logits                      │
                                                             │
  MTP Module (depth = 1) ─────────────────────────────────────┘
      Shared output head · predicts token t+2 alongside t+1
```

### Multi-Head Latent Attention (MLA)

Standard MHA caches `n_heads × head_dim` floats per token per layer.
MLA instead projects keys/values down to a **latent vector of rank 256**,
achieving a ~5× KV-cache reduction while recovering full attention quality
via the absorption trick.

```
  x  ──W_DKV──►  c_kv  (B, T, kv_lora_rank=256)   # compressed latent
                   │
           ┌───────┴──────┐
         W_UK           W_UV
           │               │
         k_C  (×n_heads)  v  (×n_heads)
           │
         + k_R  (decoupled RoPE, qk_rope_head_dim=32)
```

**Absorption trick**: at inference, `W_UK` is folded into `W_Q` so the
expanded key matrix never materialises — only `c_kv` is cached per token.
The score compute uses `F.scaled_dot_product_attention`, not einsum.

### DeepSeekMoE (16 routed + 1 shared, top-2)

```
token → AuxLossFreeGate → top-2 routed expert indices + 1 shared expert
                              │
                  Σ  gated SwiGLU expert outputs
```

Load balancing via a **bias term on each expert's gate logit**, updated
after every N optimiser steps proportional to the deviation from the
target token rate — no auxiliary loss gradient needed. Grouped forward
is the default; per-expert loop is available via `use_grouped: false`.

### BF16 + SDPA + torch.compile (no custom kernels)

```
BF16 activation
      │
  F.linear  (cuBLAS SGEMM, BF16)
      │
  F.scaled_dot_product_attention   ← Flash-Attention-2 on A100
      │
BF16 output  +  AdamW FP32 state  →  BF16 master weights
```

- `F.scaled_dot_product_attention` is the only attention op — no einsum,
  no custom Triton, no FlashAttn-2 build dependency.
- The MLA absorption math is preserved; only the softmax(QK^T)V step
  routes through SDPA.
- `torch.compile(model, mode="reduce-overhead")` is enabled by default.
- Gradient checkpointing is on by default (peak activation memory drops
  by ~3× at the cost of ~33% extra FLOPs on the backward pass).

### Multi-Token Prediction (MTP)

An auxiliary `MTPBlock` shares the output embedding head and predicts
**token t+2** in parallel with the main head's prediction of **token t+1**.
Improves training signal density and enables single-step speculative
decoding at inference.

```python
loss = main_loss + mtp_loss_weight * mtp_loss   # weight = 0.3
```

---

## Repository Structure

```
DeepSeek-V3-Lite/
│
├── configs/
│   ├── pretrain_800m.yaml         # 800M model + Chinchilla schedule
│   └── post-train_config.yaml     # SFT, GRPO, distillation
│
├── models/
│   ├── transformer.py             # Top-level Transformer + generate()
│   ├── mla.py                     # Multi-Head Latent Attention + YaRN RoPE
│   ├── moe.py                     # AuxLossFreeGate + DeepSeekMoE (grouped)
│   └── mtp.py                     # MTPBlock, MTPModule, MultiTokenPrediction
│
├── kernels/
│   └── bf16_linear.py            # BF16Linear drop-in for nn.Linear
│
├── training/
│   ├── pretrain.py               # LambdaLR, PretrainDataset, BF16 Pretrainer
│   ├── sft.py                    # SFTDataset (sample-isolation mask), SFTTrainer
│   ├── rl.py                     # GRPOConfig, rule_based_reward, GRPOTrainer
│   └── distillation.py           # ReasoningDistillation (KL + CE, frozen teacher)
│
├── inference/
│   ├── generate.py                # Autoregressive generation + top-p sampling
│   └── speculative.py             # SpeculativeDecoder (MTP draft + acceptance)
│
├── utils/
│   ├── checkpoint.py              # CheckpointManager: atomic safetensors
│   ├── distributed.py             # Single-GPU device() helper
│   ├── logging.py                 # TrainingLogger with env-var WandB
│   └── memory.py                  # estimate_model_memory_gb, assert_fits_in_a100_80gb
│
├── data/
│   └── prepare_data.py            # Download + tokenise FineWeb-Edu, Stack v2, MATH
│
├── assets/
│   ├── generate_plots.py          # 6-panel dark architecture chart
│   └── architecture_overview.png  # Generated overview figure
│
├── results/
│   └── training_status.md         # Live status: pending pre-training
│
├── .github/workflows/ci.yml       # Lint + model forward pass smoke tests
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Chinchilla scale, not 21B** | One A100 80GB; a 671B-param model is not reproducible here. Chinchilla-style scaling keeps the architecture authentic while making training feasible in days, not months. |
| **MLA over GQA** | 5–10× KV-cache reduction at our scale; absorption trick removes key expansion at decode time. |
| **Aux-loss-free MoE balancing** | Bias updates achieve balance without affecting the task loss; the original CE-load-balancing term is kept in code (default `balance_loss_alpha=0.0`) for paper-faithful re-runs. |
| **SDPA in MLA, no einsum** | `F.scaled_dot_product_attention` on A100 emits Flash-Attn-2; all attention scores are computed via `torch.matmul` / `torch.bmm` / SDPA — no `torch.einsum` anywhere in the model. |
| **BF16 + torch.compile** | A100 SXM has BF16 tensor cores; `torch.compile(mode="reduce-overhead")` removes Python overhead from the per-block forward. No FP8 — A100 SXM does not have FP8 tensor cores. |
| **Grouped MoE forward** | Per-expert Python loop is the worst bottleneck; a single index-sorted batched forward keeps the same math but runs in one Python trip. Per-expert loop available as `use_grouped: false`. |
| **MTP depth = 1** | Single extra prediction head shares weights with the output head; boosts training signal; directly enables 1-step speculative decoding at inference. |
| **Decoupled RoPE** | RoPE applied only to a 32-dim subspace; content keys (96 dim) remain unrotated, preserving low-rank KV projection accuracy. |
| **Gradient checkpointing on by default** | Peak activation memory drops from ~14 GB → ~7 GB at `bs=4, seq=1024`; the ~33% extra FLOPs are cheap on A100. |

---

## Model Configuration

```yaml
# From configs/pretrain_800m.yaml
model:
  vocab_size:          14336
  dim:                 1024
  n_layers:            24           # 2 dense + 22 MoE
  n_heads:             16
  n_routed_experts:    16
  n_shared_experts:    1
  n_activated_experts: 2           # top-2 routing
  kv_lora_rank:        256
  qk_nope_head_dim:    96
  qk_rope_head_dim:    32
  v_head_dim:          96
  max_seq_len:         1024
  mtp_depth:           1
  mtp_loss_weight:     0.3
  dtype:               bf16
  attn_impl:           "sdpa"       # F.scaled_dot_product_attention
  use_grouped:         "stacked"    # grouped MoE forward

# Resulting parameter count
total_params:         757_226_496  # ~757M
chin_optimal_tokens:   15_144_531_968  # 20 × params
chin_upper_tokens:   75_722_649_840  # 100 × params
```

---

## Training Pipeline

### 1. Pre-training

```bash
python training/pretrain.py --config configs/pretrain_800m.yaml
```

- **Schedule:** 46,300-step linear warmup → cosine decay to 8.4e-6
- **Total tokens:** 15.1B (matches Chinchilla 20:1)
- **Single A100 80GB;** `torch.compile(mode="reduce-overhead")` and gradient
  checkpointing on by default.
- **Attention:** `F.scaled_dot_product_attention` (Flash-Attn-2 on A100).
- **MoE dispatch:** stacked forward (default); `--no-grouped` for the
  per-expert Python loop.
- **Checkpoints:** safetensors weights + `.pt` optimiser state, atomic
  temp-rename.

### 2. Supervised Fine-Tuning

```bash
python training/sft.py \
    --config configs/pretrain_800m.yaml \
    --model-path checkpoints/pretrain_800m
```

- Chat-template formatting; **sample-isolation loss mask** zeroes prompt
  tokens so loss is computed on assistant completions only.
- `loss_mask` is built from the prompt-only chat-template offset, not
  hard-coded to all-ones.

### 3. GRPO Reinforcement Learning

```bash
python training/rl.py
```

- `group_size=4` (8 completions × 2 policies doesn't fit at 512 tokens on
  a single 80 GB GPU).
- Rule-based reward (boxed answer, reasoning keywords, length).
- PPO clip ε = 0.2, KL penalty = 0.04 against frozen reference.
- `seq_log_ratio` clamped to ±20 to prevent `exp()` overflow early in
  training.

### 4. R1 Reasoning Distillation

```bash
python training/distillation.py \
    --config configs/pretrain_800m.yaml \
    --student-path checkpoints/sft_800m \
    --teacher-path checkpoints/pretrain_800m
```

- `DataLoader`-based loop over `(prompt, teacher_response)` JSON.
- Loss: `0.7 × KL(T=2) + 0.3 × CE` on teacher tokens only.

---

## Inference

```python
from models.transformer import Transformer
from inference.generate import generate_interactive

model = Transformer.from_pretrained("checkpoints/pretrain")
generate_interactive(model, tokenizer, args, max_new_tokens=512, temperature=0.7, top_p=0.9)
```

### Speculative decoding (via MTP draft)

```python
from inference.speculative import SpeculativeDecoder

decoder = SpeculativeDecoder(model, mtp_module, acceptance_threshold=0.8)
tokens = decoder.generate(prompt_ids, max_new_tokens=512)
```

The MTP head produces a draft for token `t+2`; if the main model's
probability ratio exceeds 0.8 the draft is accepted, doubling throughput
in the best case.

---

## Data Preparation

```bash
python data/prepare_data.py --stage all \
    --tokenizer deepseek-ai/deepseek-coder-v2-lite
```

Stages:
- `pretrain` — download FineWeb-Edu, Stack v2, MATH, tokenise, pack into
  a flat `.bin` tensor.
- `sft` — seed `data/sft_data.json` with a few instruction/response
  examples.
- `distill` — seed `data/distill_data.json` with one reasoning example.

`tokenize_and_pack` requires a HuggingFace tokenizer — there is no
character-encoding fallback.

For Chinchilla's 15.1B tokens at this model size, plan for ~60 GB of
uncompressed text per shard. Use `--shard-size-tokens 1e9` to keep each
shard at ~1 GB on disk.

---

## Setup

```bash
git clone https://github.com/atandra2000/DeepSeek-V3-Lite
cd DeepSeek-V3-Lite
pip install -r requirements.txt
```

Runs on a single NVIDIA A100 80GB SXM (BF16 + SDPA + `torch.compile`).
CPU fallback is supported for smoke tests only — `torch.compile` is
disabled when CUDA is unavailable.

---

## References

- [Chinchilla scaling laws (Hoffmann et al. 2022)](https://arxiv.org/abs/2203.15556) — the 20 tokens/param rule we follow
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — architecture, MLA, MoE design
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — fine-grained expert decomposition
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO + reasoning distillation
- [Multi-Token Prediction](https://arxiv.org/abs/2404.19737) — training efficiency via auxiliary prediction heads
- [YaRN](https://arxiv.org/abs/2309.00071) — efficient long-context RoPE extension
- [Flash-Attention-2](https://arxiv.org/abs/2205.14135) — the kernel that backs `F.scaled_dot_product_attention` on A100

---

## 757M Config (`configs/pretrain_800m.yaml`)

Same architecture (MLA, MoE, MTP, aux-loss-free), scaled up to ~757M params.

| Property | 757M config |
|---|---|---|
| Parameters | 757,226,496 |
| Layers (dense / MoE) | 2 / 22 |
| `dim` | 1024 |
| `n_heads` | 16 |
| `n_routed_experts` | 16 |
| `kv_lora_rank` | 256 |
| `moe_inter_dim` | 512 |
| Micro batch × grad accum | 4 × 8 (effective 32) |
| LR (µP-scaled) | 8.4e-5 |
| Total steps (Chinchilla-20) | 463,000 |
| Wall time on A100 | ~8.5 days |
| Peak VRAM (predicted) | ~14 GB |

**Launch sequence** (run from repo root on the A100 host):

```bash
# 1. Data: download + tokenise ~15B tokens into data/pretrain_800m/
python data/prepare_data.py --stage pretrain \
    --tokenizer deepseek-ai/deepseek-coder-v2-lite \
    --shard-size-tokens 1000000000 --max-tokens 15100000000 \
    --data-mix deepseek-v3 --include-extra

# 2. Microbench — measure peak VRAM, must be < 70 GB
python scripts/microbench_800m.py

# 3. Step time — must hit 25%+ MFU (35-45% is the target)
python scripts/step_time_800m.py --steps 20 --warmup 5

# 4. (Optional) 1k-step dry-run on a 100M-token subset

# 5. Launch the full Chinchilla-20 run
bash scripts/launch_800m.sh
```

The launch script sets up the env, starts the run in the background, and
tails the first 50 lines of the log so you can spot-check the loss curve
before walking away. The full run is ~8-9 days wall-clock; checkpoints
saved every 5,000 steps (93 of them) plus the final step.
>>>>>>> 8ce61c8 (Initial commit: DeepSeek-V3-Lite Chinchilla-scale reproduction)

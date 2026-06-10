# DeepSeek-V3-Lite

A faithful, from-scratch reimplementation of the DeepSeek-V3 architecture, scaled for compute-optimal single-GPU training.

| Config | Parameters | Chinchilla tokens | Wall time (A100) | Peak VRAM | Status |
|---|---|---|---|---|---|
| `configs/pretrain_800m.yaml` | 757M | 15.1B | ~9 days | ~14 GB | Code complete, pre-training pending |

BF16 forward, `F.scaled_dot_product_attention` (Flash-Attn-2), `torch.compile`, zero custom CUDA. Fits one A100 80GB.

---

![DeepSeek-V3-Lite Architecture](assets/architecture_overview.png)

## Architecture

The model follows the DeepSeek-V3 technical report exactly — every component is implemented end-to-end, no stubs.

```
Input tokens (vocab = 14,336)
    │
    ▼
  Embedding (14,336 × 1,024)
    │
    ├─ Layers 0–1: Dense Transformer Blocks
    │     MLA → SwiGLU FFN
    │
    ├─ Layers 2–23: MoE Transformer Blocks (×22)
    │     MLA → DeepSeekMoE FFN
    │             ├─ 1 shared expert (always active)
    │             └─ 16 routed experts (top-2 per token)
    │
    └─ RMSNorm → Linear head → logits

  MTP Module (depth = 1) ──────────────────────┘
      Shared output head · predicts token t+2 alongside t+1
```

### Multi-Head Latent Attention (MLA)

MLA projects keys and values into a low-rank latent space (`kv_lora_rank=256`), then recovers full multi-head K and V via up-projection. The **absorption trick** folds the K up-projection into the query weight at inference, so only the compressed latent is cached — a ~5× KV-cache reduction. RoPE is applied to a decoupled 32-dim subspace, keeping the content keys rotation-free.

### DeepSeekMoE

16 routed experts with top-2 routing plus 1 always-active shared expert. Load balancing uses **aux-loss-free bias updates**: a per-expert bias on the gate logit is adjusted periodically based on observed token count deviation, with no auxiliary gradient term contaminating the task loss. Three dispatch modes available — `stacked` (single-bmm), `grouped` (per-expert bmm), and `loop` (per-expert Python, debug only).

### Multi-Token Prediction (MTP)

An auxiliary prediction head shares the output embedding and predicts token `t+2` in parallel with the main head. This densifies the training signal and enables single-step speculative decoding at inference.

---

## Training Pipeline

### 1. Pre-training

```bash
python training/pretrain.py --config configs/pretrain_800m.yaml
```

- Chinchilla-optimal schedule: 2,000-step warmup → cosine decay over 463,000 steps
- BF16 forward, FP32 AdamW master weights
- Gradient checkpointing on by default (~3× activation memory reduction)
- `torch.compile(mode="reduce-overhead")`
- Safetensors checkpoints with atomic temp-rename

### 2. Supervised Fine-Tuning

```bash
python training/sft.py \
    --config configs/pretrain_800m.yaml \
    --model-path checkpoints/pretrain_800m
```

Sample-isolation loss masking — the CE loss is computed only on assistant completion tokens, with prompt tokens zeroed via `loss_mask`.

### 3. GRPO Reinforcement Learning

```bash
python training/rl.py
```

- Group size 4 (fits single A100 80GB at 512 tokens)
- Rule-based reward (boxed answers, reasoning keywords, length)
- PPO clip ε = 0.2, KL penalty β = 0.04
- Log-ratio clamped to ±20 to prevent `exp()` overflow

### 4. R1 Reasoning Distillation

```bash
python training/distillation.py \
    --config configs/pretrain_800m.yaml \
    --student-path checkpoints/sft_800m \
    --teacher-path checkpoints/pretrain_800m
```

Loss = 0.7 × KL(T=2) + 0.3 × CE, computed on teacher-generated tokens only.

---

## Inference

```python
from models.transformer import Transformer
from inference.generate import generate_interactive

model = Transformer(cfg).to("cuda")
generate_interactive(model, tokenizer, args, max_new_tokens=512)
```

### Speculative Decoding

The MTP draft head produces a candidate for token `t+2`. If the main model's probability ratio exceeds the acceptance threshold (default 0.8), the draft is accepted — up to 2× throughput in the best case.

```python
from inference.speculative import SpeculativeDecoder

decoder = SpeculativeDecoder(model, mtp_module, acceptance_threshold=0.8)
tokens = decoder.generate(prompt_ids, max_new_tokens=512)
```

---

## Quick Start

```bash
git clone https://github.com/atandra2000/DeepSeek-V3-Lite
cd DeepSeek-V3-Lite
pip install -r requirements.txt

# Prepare data (tokenizer required)
python data/prepare_data.py --stage all --tokenizer deepseek-ai/deepseek-coder-v2-lite
```

Runs on a single A100 80GB SXM. CPU fallback is available for smoke tests only — `torch.compile` is disabled when CUDA is absent.

### Launch Sequence (A100)

```bash
# 1. Data — download and tokenise ~15B tokens
python data/prepare_data.py --stage pretrain \
    --tokenizer deepseek-ai/deepseek-coder-v2-lite \
    --shard-size-tokens 1000000000 --max-tokens 15100000000 \
    --data-mix deepseek-v3 --include-extra

# 2. Microbench — measure peak VRAM
python scripts/microbench_800m.py

# 3. Step-time benchmark — validate MFU
python scripts/step_time_800m.py --steps 20 --warmup 5

# 4. Launch the full run (~8-9 days)
bash scripts/launch_800m.sh
```

---

## Project Structure

```
├── configs/
│   ├── pretrain_800m.yaml         # 757M model + Chinchilla schedule
│   └── post-train_config.yaml     # SFT, GRPO, distillation
├── models/
│   ├── transformer.py             # Top-level Transformer + generate()
│   ├── mla.py                     # Multi-Head Latent Attention
│   ├── moe.py                     # AuxLossFreeGate + DeepSeekMoE
│   └── mtp.py                     # MTPBlock, MTPModule, MultiTokenPrediction
├── training/
│   ├── pretrain.py                # Pre-training (BF16, LambdaLR, sharded dataset)
│   ├── sft.py                     # SFT with sample-isolation masking
│   ├── rl.py                      # GRPO reinforcement learning
│   └── distillation.py            # R1-style knowledge distillation
├── inference/
│   ├── generate.py                # Autoregressive generation
│   └── speculative.py             # MTP speculative decoding
├── utils/
│   ├── checkpoint.py              # Atomic safetensors checkpoint manager
│   ├── distributed.py             # Single-GPU device helper
│   ├── logging.py                 # WandB-capable training logger
│   └── memory.py                  # VRAM estimator + A100 guard
├── data/
│   └── prepare_data.py            # Download, tokenise, pack datasets
├── scripts/
│   ├── microbench_800m.py         # Peak VRAM measurement
│   ├── step_time_800m.py          # MFU benchmark
│   └── launch_800m.sh             # Full run launcher
├── kernels/
│   └── bf16_linear.py             # BF16Linear drop-in
└── configs/pretrain_800m.yaml     # Primary configuration
```

---

## Configuration

```yaml
# configs/pretrain_800m.yaml — key model hyperparameters
model:
  vocab_size:          14336
  dim:                 1024
  n_layers:            24           # 2 dense + 22 MoE
  n_heads:             16
  n_routed_experts:    16
  n_shared_experts:    1
  n_activated_experts: 2
  kv_lora_rank:        256
  qk_rope_head_dim:    32
  v_head_dim:          96
  max_seq_len:         1024
  attn_impl:           "sdpa"
  use_grouped:         "stacked"

training:
  micro_batch_size:              4
  gradient_accumulation_steps:   8
  total_steps:                   463000
  lr:                            2.2e-4    # µP-scaled to 8.4e-5
```

**Parameter count: 757,226,496** (`~757M`). Chinchilla-optimal at 15.1B tokens (20:1 ratio).

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Chinchilla scale, not 671B | One A100 80GB; authentic architecture at feasible training time |
| MLA over GQA | 5× KV-cache reduction; absorption trick removes key expansion at decode |
| Aux-loss-free MoE balancing | Bias updates don't contaminate task loss gradient |
| SDPA over einsum | Flash-Attn-2 on A100; zero custom CUDA dependencies |
| BF16 + torch.compile | A100 BF16 tensor cores; no FP8 hardware available |
| Stacked MoE forward | One Python trip per layer, not per expert |
| Decoupled RoPE | 32-dim rotation preserves low-rank KV compression accuracy |
| Gradient checkpointing | ~3× activation reduction at 33% extra backward FLOPs |

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — architecture, MLA, MoE
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) — 20 tokens/param rule
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — fine-grained expert decomposition
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO + reasoning distillation
- [Multi-Token Prediction](https://arxiv.org/abs/2404.19737) — auxiliary prediction heads
- [YaRN](https://arxiv.org/abs/2309.00071) — efficient RoPE extension

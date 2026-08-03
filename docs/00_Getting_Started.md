# 00 — Getting Started

> **Canonical** for DeepSeek-V3 onboarding, environment setup, smoke tests, and canonical numbers. Educational textbook chapter.

> Your first stop: what DeepSeek-v3-Lite is, how to run it, and where to read next. Architecture, MLA math, DeepSeekMoE routing, MTP, and FP8 precision depth live in the chapter sequence starting at [[Docs/01_Foundations]].

**Depends on:** (none) · **Read next:** [[Docs/01_Foundations]]

---

## 1. Overview & Core Innovations

**DeepSeek-v3-Lite** is a from-scratch PyTorch reproduction of **DeepSeek-V3** (DeepSeek-AI, 2024) at Chinchilla-optimal scale (~422M params). It incorporates four seminal architectural breakthroughs from DeepSeek:

1. **Multi-Head Latent Attention (MLA)** — Low-rank KV compression ($d_c = 192$) with decoupled RoPE ($d_R = 24$) reduces KV cache memory consumption by over $75\%$ while outperforming standard Multi-Head Attention.
2. **DeepSeekMoE with Aux-Loss-Free Balancing** — 20 fine-grained routed experts (4 activated per token) + 1 always-active shared expert. Load balancing is achieved without auxiliary loss penalties by dynamically adjusting expert bias terms ($b_i \leftarrow b_i + \gamma \cdot \text{sign}(\text{target} - \text{load})$).
3. **Multi-Token Prediction (MTP)** — Sequential MTP modules (depth 1) predict next-token targets $t+1$ and $t+2$ simultaneously, densifying training signals and enabling zero-overhead speculative decoding.
4. **$\mu\text{P}$ Learning Rate Scaling & optional fused Triton kernels** — Maximal Update Parameterization ($\mu\text{P}$) scaling transfers the LR across widths; optional fused Triton kernels accelerate MLA attention and grouped MoE GEMM in BF16. *(DeepSeek-V3's FP8 mixed precision and DualPipe parallelism are paper techniques documented in [[Docs/06_FP8_Mixed_Precision]] and [[Docs/07_DualPipe_Parallelism]] but not implemented in this single-GPU BF16 repo.)*

---

## 2. Model Architecture at a Glance

```
Input Tokens (vocab = 100,018)
    │
    ▼
Embedding (dim = 768) ──────────────────── [ Weight-tied with LM Head ]
    │
    ▼
18 × Transformer Blocks (Gradient Checkpointing enabled):
    ├─ Layer 0–1:  RMSNorm → MLA (c_KV=192, qk_rope=24) → RMSNorm → Dense SwiGLU (d_ff=1536)
    └─ Layer 2–17: RMSNorm → MLA (c_KV=192, qk_rope=24) → RMSNorm → DeepSeekMoE (20 Routed + 1 Shared)
    │
    ▼
Final RMSNorm → Linear Output Head → Cross-Entropy Loss (Main Branch)
    │
    ▼
MTP Module (Depth 1) ───> Predicts t+2 Token ───> MTP Loss (Weight = 0.3)
```

---

## 3. Canonical Parameters & Configuration

| Parameter | Canonical Value | Description |
|---|---|---|
| **Total Parameters** | **~422M** | ~280M active params per token |
| `dim` ($d_{\text{model}}$) | **768** | Hidden embedding dimension |
| `n_layers` | **18** | 2 Dense layers + 16 MoE layers |
| `n_heads` | **12** | Query attention heads |
| `kv_lora_rank` ($d_c$) | **192** | Compressed KV rank dimension |
| `qk_nope_head_dim` | **48** | Non-positional query/key head dimension |
| `qk_rope_head_dim` ($d_R$) | **24** | Decoupled RoPE head dimension |
| `v_head_dim` | **64** | Value head dimension |
| `n_routed_experts` | **20** | Fine-grained routed experts |
| `n_shared_experts` | **1** | Always-active shared expert |
| `n_activated_experts` | **4** | Routed experts selected per token |
| `inter_dim` / `moe_inter_dim` | **1536 / 384** | SwiGLU FFN expansion / Expert FFN dimension |
| `vocab_size` | **100,018** | DeepSeek BPE Tokenizer |
| `mtp_depth` / `mtp_loss_weight` | **1 / 0.3** | MTP depth & auxiliary loss weight |

> [!NOTE]
> Configuration file: [`configs/pretrain_a100_422m.yaml`](../configs/pretrain_a100_422m.yaml).

---

## 4. Environment & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/atandra2000/DeepSeek-v3-Lite.git
cd DeepSeek-v3-Lite

# 2. Install requirements
pip install -r requirements.txt

# 3. Verify CPU smoke tests (< 2s)
python3 -m pytest tests/ -v

# 4. Launch pretraining on A100 GPU
python3 training/pretrain.py --config configs/pretrain_a100_422m.yaml
```

---

## 5. Learning Path Curriculum

Follow the 14-chapter sequence under `Docs/`:

| Step | Chapter | Focus Area |
|---|---|---|
| 00 | [[Docs/00_Getting_Started]] | Onboarding, canonical numbers, smoke test execution |
| 01 | [[Docs/01_Foundations]] | DeepSeek lineage (V1 $\rightarrow$ V2 $\rightarrow$ V3) & design choices |
| 02 | [[Docs/02_Model_Architecture]] | Full topology, tensor shapes, parameter accounting |
| 03 | [[Docs/03_Multi_Head_Latent_Attention]] | MLA low-rank compression, matrix absorption, RoPE |
| 04 | [[Docs/04_DeepSeekMoE]] | Fine-grained experts, shared expert, aux-loss-free bias updates |
| 05 | [[Docs/05_Multi_Token_Prediction]] | MTP module depth, training loss, speculative decoding |
| 06 | [[Docs/06_FP8_Mixed_Precision]] | E4M3/E5M2 precision, tile-wise & block-wise FP8 scaling |
| 07 | [[Docs/07_DualPipe_Parallelism]] | DualPipe bidirectional pipeline parallelism & overlap |
| 08 | [[Docs/08_Training_Pipeline]] | Pretraining loop, AdamW, $\mu\text{P}$ scaling, NaN guards |
| 09 | [[Docs/09_Data_Pipeline]] | Tokenizer, dataset mixture, mmap binary sharding |
| 10 | [[Docs/10_Inference_and_Serving]] | Autoregressive sampling, MLA KV decompression, speculative decode |
| 11 | [[Docs/11_Operations_and_Testing]] | Pytest suite, safetensors checkpointing, VRAM budget |
| 12 | [[Docs/12_Triton_Kernels]] | Custom Triton kernels (MLA, MoE dispatch, FP8 GEMM) |
| 13 | [[Docs/13_Portfolio_Comparison]] | Architecture comparison vs LLaMA-3, Mamba-3, HyMo |

---

> **Next:** [[Docs/01_Foundations]] — DeepSeek architectural lineage from V1 to V3.

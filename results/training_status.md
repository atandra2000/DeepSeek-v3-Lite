# Training Status

## Current Stage: 800M config ready, awaiting A100 launch sequence

Architecture and training code are complete for the new 800M Chinchilla-20 config. CPU smoke tests pass for all paths. Pre-training has **not started** yet (awaiting the A100 microbench + 1k-step dry-run before the long run).

**Targets:**
- **757M parameters** (Chinchilla-20, 15.1B tokens) — `configs/pretrain_800m.yaml`

---

### Implementation Checklist (post-800M upgrade)

| Component | File | Status |
|---|---|---|
| Transformer backbone (24 layers, dim=1024) | `models/transformer.py` | ✅ Verified |
| Multi-Head Latent Attention (SDPA, no einsum) | `models/mla.py` | ✅ Verified — incl. bsz>1 SDPA vs manual bit-exact (1e-7) |
| DeepSeekMoE + AuxLossFree gate — `loop` / `grouped` / `stacked` | `models/moe.py` | ✅ Verified — all three paths bit-exact (0.0e+00) at 4 and 16 experts |
| Multi-Token Prediction (depth=1) | `models/mtp.py` | ✅ Verified |
| BF16 linear drop-in | `kernels/bf16_linear.py` | ✅ Verified |
| Pre-training (BF16, LambdaLR, SDPA, stacked MoE) | `training/pretrain.py` | ✅ Verified on CPU |
| **Phase D1** — µP LR scaling (`lr ∝ P^(-0.5)`) | `training/pretrain.py` | ✅ Verified |
| **Phase D2** — per-component param breakdown at startup | `training/pretrain.py` | ✅ Verified |
| **Phase D3** — NaN/Inf guard with checkpoint restore | `training/pretrain.py` | ✅ Verified (injected-NaN test) |
| **Phase B4** — `seq_len` plumbed through logger | `utils/logging.py`, `training/pretrain.py` | ✅ Verified |
| **Phase B3** — memory estimator uses real `dim`/`n_layers` + 13.7 GB overhead | `utils/memory.py` | ✅ Verified (800M → 14.82 GB) |
| SFT trainer (sample-isolation mask) | `training/sft.py` | ✅ Verified |
| GRPO trainer (rule-based reward, log-ratio clamp) | `training/rl.py` | ✅ Verified |
| R1 distillation (DataLoader over JSONL) | `training/distillation.py` | ✅ Verified |
| Autoregressive inference (KV cache) | `inference/generate.py` | ✅ Verified |
| Speculative decoding (MTP draft) | `inference/speculative.py` | ✅ Verified |
| Checkpoint manager (atomic safetensors) | `utils/checkpoint.py` | ✅ Verified |
| Single-GPU device helper | `utils/distributed.py` | ✅ Verified |
| Training logger (env-var WandB) | `utils/logging.py` | ✅ Verified |
| Data prep — sharded output (800M lineage), `--data-mix deepseek-v3` | `data/prepare_data.py` | ✅ Verified |
| **Phase C3** — `PretrainDataset` auto-detects single vs sharded layout | `training/pretrain.py` | ✅ Verified (5 cases) |
| 800M config | `configs/pretrain_800m.yaml` | ✅ Verified (757M params) |
| Post-training config | `configs/post-train_config.yaml` | ✅ Verified |
| **Phase E** — microbench + step-time scripts | `scripts/` | ✅ Written (awaiting A100) |
| **Phase F** — launch script | `scripts/launch_800m.sh` | ✅ Written (awaiting A100) |

### Removed (non-functional on a single A100 80GB BF16 setup)

| Component | Reason |
|---|---|
| `kernels/fp8_kernel.py` (Triton FP8 kernels) | A100 SXM has no FP8 tensor cores |
| `utils/communication.py` (MoE / pipeline comms stubs) | Single-GPU |
| FSDP / DDP wrappers | Single-GPU |
| `torch.einsum` in MLA | Replaced with `torch.matmul` / `torch.bmm` / SDPA |
| WandB API key in YAML | Replaced with `WANDB_PROJECT` env var |

---

### Expected Pre-training Configurations

#### 800M Chinchilla-20 (max-recommended single-A100)

| Hyperparameter | Value |
|---|---|
| **Parameters** | **757,226,496 (~757M)** |
| **Hardware** | 1 × NVIDIA A100 80GB SXM |
| **Chinchilla-20 tokens** | 15.1B |
| **Schedule** | 463 000 steps × 4 micro × 8 grad_acc × 1 024 ctx = 15.1B tokens |
| **Sequence length** | 1 024 |
| **Effective batch** | 4 × 8 = 32 |
| **Learning rate (µP-scaled)** | 8.4e-5 (auto-derived from 6e-4) |
| **Total steps** | 463 000 |
| **Precision** | BF16 forward + BF16 master, FP32 AdamW state |
| **Attention** | `attn_impl: "sdpa"` (default) — `attn_impl: "absorption"` available |
| **MoE dispatch** | `use_grouped: "stacked"` (Phase B1 fused path) |
| **Compile** | `torch.compile(mode="reduce-overhead")` |
| **Gradient checkpointing** | On by default |
| **NaN/Inf guard** | On by default (restores from latest checkpoint after 5 fires) |
| **Bias update** | every 20 optimiser steps, speed 0.002 |
| **MTP loss weight** | 0.3 |
| **Peak VRAM (bs=4, seq=1024, grad-ckpt)** | ~14.8 GB (estimated, awaiting A100 measurement) |
| **Predicted wall time (A100, 40% MFU)** | ~8.5 days |

### Token-budget guidance

For 15.1B tokens of pre-training data (800M Chinchilla-20), use the
`deepseek-v3` data mix with `--shard-size-tokens 1000000000`. Plan for
~100 GB of disk.

---

### Launch sequence (800M run, on the A100 host)

```bash
# 1. Build the data (one-time, ~1-2 days of downloads + tokenisation)
python data/prepare_data.py --stage pretrain \
    --tokenizer deepseek-ai/deepseek-coder-v2-lite \
    --shard-size-tokens 1000000000 --max-tokens 15100000000 \
    --data-mix deepseek-v3 --include-extra

# 2. Microbench — measure peak VRAM, must be < 70 GB
python scripts/microbench_800m.py

# 3. Step time — must hit 25%+ MFU (35-45% is the target)
python scripts/step_time_800m.py --steps 20 --warmup 5

# 4. (Optional but recommended) 1k-step dry-run on a 100M-token subset

# 5. Launch the full Chinchilla-20 run
bash scripts/launch_800m.sh
```

---

*This file will be updated with loss curves and benchmark results once training begins.*

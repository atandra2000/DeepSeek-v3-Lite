# DeepSeek-v3-Lite — Code Documentation

Conceptual notes extracted from the source tree during the codebase
cleanup. These files explain the *why* behind components whose rationale
was previously inlined as comments; the code itself stays clean.

## Index

| File | Component(s) | Source |
|------|--------------|--------|
| [mla.md](mla.md) | Multi-Head Latent Attention | `models/mla.py` |
| [moe.md](moe.md) | AuxLossFreeGate + DeepSeekMoE | `models/moe.py` |
| [mtp.md](mtp.md) | Multi-Token Prediction + speculative decoder | `models/mtp.py`, `inference/speculative.py` |
| [transformer.md](transformer.md) | Top-level Transformer wiring | `models/transformer.py` |
| [training.md](training.md) | Pretrain loop, μP, NaN guard | `training/pretrain.py` |
| [data_pipeline.md](data_pipeline.md) | Data mixture + tokenizer | `data/prepare_data.py` |
| [inference.md](inference.md) | Generate + speculative decoder | `inference/` |
| [utils.md](utils.md) | Checkpoint, distributed, logging, memory | `utils/` |

## Authoritative MLA reference

The deep theory of MLA — KV-cache problem, low-rank compression, the
absorption trick algebra, decoupled RoPE, dimension breakdown, comparison
against MHA/GQA/MQA, and a full implementation walkthrough — lives in the
top-level **[`../MLA.md`](../MLA.md)** (643 lines). That file is the
canonical reference; `mla.md` here only records project-specific notes and
defers to it for theory.

## Load-bearing invariants (do not break)

- **AuxLossFreeGate bias** is a **buffer**, not a `Parameter`. It receives
  no autograd updates and persists via `state_dict()`. Replacing the
  bias-update mechanism with a standard auxiliary loss silently breaks MoE
  load balance. See [moe.md](moe.md).
- **MLA absorption trick** — `W^{UQ}·W^{UKᵀ}` is folded so full K/V are
  never materialised at inference. The SDPA path materialises for FA2
  efficiency; the manual path demonstrates true absorption. See
  [mla.md](mla.md) and `../MLA.md`.
- **μP LR scaling** — `new_lr = mup_lr_reference * (mup_lr_reference_params / total) ** 0.5`,
  applied after MTP-wrap param counting (so total may include MTP heads).
  Reference: 6e-4 @ 757M → ~8.07e-4 @ 422M. See [training.md](training.md).
- **NaN guard** — after `nan_guard_max_consecutive=5` consecutive NaN/Inf
  steps, the run auto-rolls back to the last good checkpoint and resumes
  scheduler/optimizer state. Never disable without explicit user consent.
  See [training.md](training.md).
- **Vocab 100,018** — `deepseek-coder-v2-lite` tokenizer with
  `byte_fallback` tokens. Embedding dim must equal `vocab_size`. See
  [data_pipeline.md](data_pipeline.md).
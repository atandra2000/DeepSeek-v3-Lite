# Archify delivery evidence: DeepSeek-v3-Lite

The [interactive guide](deepseek_visual_guide.html) links four standalone diagrams.

Content repair pass, 2026-09-17. Source revision `2edd9cdbf6b6708f103001c1d50fff2b05b32c85`, configuration `configs/pretrain_a100_422m.yaml`.

All four: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed** on the corrected artifacts.

Perceptual review: **skipped (image reader unavailable)**. No screenshot-based polish judgment is claimed; the 12px/11px typography target remains a proposal.

### Model Architecture

- Output: [architecture-model.html](architecture-model.html)
- Specification: `docs/diagrams/architecture-model.json`
- Specification SHA-256: `26166b0281749c0712832cf3bb0ab4bd50f1fa42b67c5a301a0f15d0d4d3b7c7` (7,284 bytes)
- Artifact SHA-256: `2e8002c5212f30e45c8913129437c7640913c53c917cf01c392a0b8dc84c1a76` (719,386 bytes)
- [Browser receipt](architecture-model.visual-check.json) - status `pass`
- `browser_evidence: passed` - `visual_review: skipped (image reader unavailable)`

### Optimization Stack

- Output: [architecture-optimizations.html](architecture-optimizations.html)
- Specification: `docs/diagrams/architecture-optimizations.json`
- Specification SHA-256: `8ecda9086bd7a508ca1f120bf7dcee138c3a85d01941f0c760affed4227c5fdb` (8,352 bytes)
- Artifact SHA-256: `42eb980a3fe9e5d9f10b351078aeab3daff3584db92c0125cfc8198d951a6857` (720,619 bytes)
- [Browser receipt](architecture-optimizations.visual-check.json) - status `pass`
- `browser_evidence: passed` - `visual_review: skipped (image reader unavailable)`

### Data Pipeline

- Output: [dataflow-data-pipeline.html](dataflow-data-pipeline.html)
- Specification: `docs/diagrams/dataflow-data-pipeline.json`
- Specification SHA-256: `cca0ef5ed581ebed95fde961d97e05ec01673742f1ef808a65a4fad8281abd88` (5,914 bytes)
- Artifact SHA-256: `c6b84807c5d7678271f5a753a070ab93ac1dbecafdac27e893741fdaa0406682` (714,113 bytes)
- [Browser receipt](dataflow-data-pipeline.visual-check.json) - status `pass`
- `browser_evidence: passed` - `visual_review: skipped (image reader unavailable)`

### Training Loop

- Output: [workflow-training-loop.html](workflow-training-loop.html)
- Specification: `docs/diagrams/workflow-training-loop.json`
- Specification SHA-256: `33be479ee421f9d5626ec1a9fa259375e6b8b0eb481a91d9694f9fbdd8b15f70` (6,686 bytes)
- Artifact SHA-256: `1317c4bc4b9acff381b17f223c25ca3d5ec3d46b17a2014c61d93dd65f077f4c` (716,954 bytes)
- [Browser receipt](workflow-training-loop.visual-check.json) - status `pass`
- `browser_evidence: passed` - `visual_review: skipped (image reader unavailable)`

## Content corrections in this pass

- KV cache: 192 latent + 24 positional = 216 values/token/layer (models/mla.py:MultiHeadLatentAttention._ensure_cache); the ~7x ratio is arithmetic against its stated baseline.
- Manual eager absorption vs SDPA K/V materialization distinguished; acceptance 0.8 labeled configured, throughput unmeasured.
- Dataflow discloses the real format gap: shared pipeline writes raw uint32 streams while training/pretrain.py:PretrainDataset reads torch.load shards.
- Training loop: MTP loss silently disabled (main() reads mtp_loss_weight from training: while the YAML defines it under model:); micro-step vs optimizer-step clocks; sampler state not checkpointed.
- Optimizations: FP32 master weights sized ~1.6 GB (earlier 0.8 GB was BF16-sized); A100 targets kept separate from measurements.

## Verification limits

No model code changed. Parameter counts are counts; memory figures are arithmetic; no MFU, throughput, cache-savings or corpus-inventory number is claimed as measured. Speculative verification conditioning remains a separate investigation.

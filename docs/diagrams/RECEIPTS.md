# Archify delivery evidence — DeepSeek-v3-Lite

The [interactive visual guide](deepseek_visual_guide.html) links four standalone showcase architecture, dataflow, and workflow diagrams.

All four: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed.**

Chrome checked at 1440×900, 1600×1000, 1920×1080 and 2048×1320 in light/dark. All required viewport measurements passed horizontal/vertical containment, minimum projected text size and viewer-control clearance.

## Artifact bindings

### Model Architecture (MLA + DeepSeekMoE + MTP)

- Diagram type: `architecture`
- Output: [architecture-model.html](architecture-model.html)
- Specification: `docs/diagrams/architecture-model.json`
- Artifact SHA-256: `dbb11c2f8bfe4343cdd6e875065c2df73ea6fdcf040d18dcca7cb84b4f96ef8d` (719,147 bytes)
- [Browser receipt](architecture-model.visual-check.json) · [Screenshot contact sheet](architecture-model.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Architecture Optimizations & Memory

- Diagram type: `architecture`
- Output: [architecture-optimizations.html](architecture-optimizations.html)
- Specification: `docs/diagrams/architecture-optimizations.json`
- Artifact SHA-256: `c1966dedf68a23a6fa0259d25d3ca80c552d83ecb1868b94cf8c4d2661017957` (720,355 bytes)
- [Browser receipt](architecture-optimizations.visual-check.json) · [Screenshot contact sheet](architecture-optimizations.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Data Pipeline & Token Streaming

- Diagram type: `dataflow`
- Output: [dataflow-data-pipeline.html](dataflow-data-pipeline.html)
- Specification: `docs/diagrams/dataflow-data-pipeline.json`
- Artifact SHA-256: `51bbb595ab9ed02be215fe062bfc2d0f802e05cdb6715bebba08ea5eeb5f2373` (713,579 bytes)
- [Browser receipt](dataflow-data-pipeline.visual-check.json) · [Screenshot contact sheet](dataflow-data-pipeline.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Training Loop & MTP Supervision

- Diagram type: `workflow`
- Output: [workflow-training-loop.html](workflow-training-loop.html)
- Specification: `docs/diagrams/workflow-training-loop.json`
- Artifact SHA-256: `86388f70a78e21ef8012d5c124ea6752e58f2829a957b1dbc51d22a7a28ca05e` (716,844 bytes)
- [Browser receipt](workflow-training-loop.visual-check.json) · [Screenshot contact sheet](workflow-training-loop.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

## Verification limits

Parameter count: ~412M total / ~185M active parameters per token (418.7M with depth-1 MTP head). Implemented in raw PyTorch with Multi-Head Latent Attention (MLA), DeepSeekMoE fine-grained routing with aux-loss-free bias updates, and μP learning rate transfer. Single A100 80GB baseline budget estimated at ~30–45 hours for pretraining.

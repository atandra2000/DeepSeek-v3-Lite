# SKILLS.md — DeepSeek-v3-Lite

> Skills for the faithful V3 reproduction. Pair with `.agents/skills/llm-architecture/SKILL.md`.

---

## Skill 1: Run a smoke test on the architecture

```bash
cd LLM/DeepSeek-v3-Lite
python -c "
from models.transformer import Transformer, ModelConfig
import torch
cfg = ModelConfig()
m = Transformer(cfg).cuda().to(dtype=torch.bfloat16)
x = torch.randint(0, cfg.vocab_size, (2, 128), device='cuda')
y, aux = m(x)
print(y.shape, len(aux))   # expected: torch.Size([2, 128, 100018]) 16
"
```

A working forward + a non-NaN loss is the minimum.

## Skill 2: Add a new MLA hyperparameter

`kv_lora_rank` (default 192), `qk_nope_head_dim` (48), `qk_rope_head_dim` (24),
`v_head_dim` (64), `q_lora_rank` (0 in 422M). Changes:

1. Update `configs/pretrain_a100_422m.yaml`.
2. Update `ModelConfig` defaults in `models/transformer.py`.
3. Re-init affected weights (the absorption trick must be re-applied if
   `q_lora_rank` or `kv_lora_rank` changes).
4. Re-run μP LR scaling.

**Pitfall:** changing `qk_rope_head_dim` invalidates existing KV cache shape.

## Skill 3: Tune AuxLossFreeGate

The bias update happens every step:
```python
# models/moe.py — _update_bias()
self.bias[expert] -= lr_bias * sign(load[expert] - target_load)
```

- `bias_update_speed=0.001` default (per `configs/pretrain_a100_422m.yaml`).
- `balance_loss_alpha=0.0` is intentional — the framework is **aux-loss-free**.
- Don't add a load-balancing loss term without disabling the bias update
  (they conflict).

## Skill 4: Use MTP for speculative decoding

```bash
python inference/generate.py --checkpoint checkpoints/dsv3_step_50000.pt \
  --prompt "The capital of France is" --max_new_tokens 200

python inference/speculative.py --checkpoint checkpoints/dsv3_step_50000.pt \
  --prompt "The capital of France is" --draft_depth 2 --acceptance 0.8
```

`draft_depth=2` uses the MTP head as a cheap draft. Expected ~1.5–2× speedup
at acceptance ≈ 0.8.

## Skill 5: Validate μP LR scaling

```bash
python -c "
import yaml
cfg = yaml.safe_load(open('configs/pretrain_a100_422m.yaml'))['training']
ref_params = cfg['mup_lr_reference_params']
target_params = 422_000_000
scale = (target_params / ref_params) ** 0.5
print(f'μP scale: {scale:.3f}')
print(f'μP LR: {cfg[\"mup_lr_reference\"] * scale:.3e}')
"
```

Expected: μP scale ≈ 0.745, μP LR ≈ 4.5e-4. (The 8.07e-4 in
`mup_lr_reference` is for the full ~757M reference model.)

## Skill 6: Add a new data source to the mixture

Edit the **universal** mixture at `data/shared_data/config/mixture.yaml`
(this is the single source of truth shared by all 5 LLM projects).
Re-run the pipeline:
```bash
python3 data/prepare_data.py --stage pretrain
# Or restrict to the new source for a dry-run:
python3 data/prepare_data.py --stage pretrain --source <new-source-id>
```

The mixture weights must sum to 1.0. To override per-project, pass
`--mixture <path-to-yaml>` to the shim.

## Pitfalls (cross-cutting)
- **NaN guard** is `nan_guard_max_consecutive=5` — after 5 consecutive NaN
  steps the run auto-rolls back to the last good checkpoint.
- **Speculative decoding** acceptance rate is prompt-dependent. Measure
  per-batch on a held-out set; do not rely on a single prompt.
- **Embedding tied?** Yes (`weight_tying: true`). Removing tying breaks
  generation quality.


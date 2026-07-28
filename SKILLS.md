# SKILLS.md — DeepSeek-v3-Lite

> Skills for the faithful V3 reproduction. Pair with `.agents/skills/llm-architecture/SKILL.md`.

---

## Skill 1: Run a smoke test on the architecture

```bash
cd LLM/DeepSeek-v3-Lite
python -c "
import torch
from models.transformer import Transformer
cfg = {
  'vocab_size': 100018, 'dim': 768, 'n_layers': 18, 'n_heads': 12,
  'n_dense_layers': 2, 'n_routed_experts': 20, 'n_shared_experts': 1,
  'n_activated_experts': 4, 'inter_dim': 1536, 'moe_inter_dim': 384,
  'kv_lora_rank': 192, 'q_lora_rank': 0, 'qk_nope_head_dim': 48,
  'qk_rope_head_dim': 24, 'v_head_dim': 64, 'max_seq_len': 2048,
  'rope_theta': 10000, 'rope_factor': 1.0, 'mscale': 1.0,
  'attn_impl': 'sdpa', 'moe_dispatch': 'stacked', 'weight_tying': True,
  'dtype': 'bf16', 'mtp_depth': 1, 'mtp_loss_weight': 0.3,
}
m = Transformer(cfg).to('cuda').to(dtype=torch.bfloat16)
x = torch.randint(0, cfg['vocab_size'], (2, 128), device='cuda')
y = m(x)
print(y.shape)  # expected: torch.Size([2, 128, 100018])
"
```

A working forward + a non-NaN loss is the minimum.

## Skill 2: Add a new MLA hyperparameter

`kv_lora_rank` (default 192), `qk_nope_head_dim` (48), `qk_rope_head_dim` (24),
`v_head_dim` (64), `q_lora_rank` (0 in 422M). Changes:

1. Update `configs/pretrain_a100_422m.yaml`.
2. Update the config dict passed to `Transformer` (no separate `ModelConfig` class exists — the dataclass defaults live in `TrainingConfig` in `training/pretrain.py`).
3. Re-init affected weights (the absorption trick must be re-applied if
   `q_lora_rank` or `kv_lora_rank` changes).
4. Re-run μP LR scaling.

**Pitfall:** changing `qk_rope_head_dim` invalidates existing KV cache shape.

## Skill 3: Tune AuxLossFreeGate

The bias update rule (in `models/moe.py::AuxLossFreeGate.update_bias`):

```python
# Deadband rule — no change inside ±bias_upper/lower_threshold of the mean.
self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
self.bias[counts < avg * (1.0 - self.bias_lower)] += speed
```

- `bias_update_speed=0.001` default (per `configs/pretrain_a100_422m.yaml`).
- `bias_update_every=1` in the canonical config (the dataclass default is 10; yaml wins).
- The bias is a `register_buffer` (not a Parameter) and is not in the gradient.
- Do not add a load-balancing loss term — the framework is **aux-loss-free**; a separate
  loss would double-count load balancing and break convergence.
- Defaults: `bias_upper_threshold=0.10`, `bias_lower_threshold=0.10` (10% deadband).

## Skill 4: Use MTP for speculative decoding

```bash
# Load a checkpoint and run interactive generation with the MTP draft head.
python inference/generate.py \
  --config configs/pretrain_a100_422m.yaml \
  --checkpoint checkpoints/pretrain \
  --use_speculative --acceptance_threshold 0.8 \
  --max_new_tokens 200
```

The speculative decoder is constructed by `inference/generate.py` when `--use_speculative`
is passed. Acceptance ~0.8 is the expected rate; `acceptance_threshold=1.0` means "always
accept the draft" (useful for sanity-checking the draft head). `inference/speculative.py`
is a library — it has no CLI of its own.

## Skill 5: Validate μP LR scaling

The formula in `training/pretrain.py`:

```python
new_lr = mup_lr_reference * (mup_lr_reference_params / total) ** 0.5
```

```bash
python -c "
# Reference: 6e-4 at 757,226,496 params. Scale = sqrt(ref / total).
ref_lr, ref_params = 6.0e-4, 757_226_496
target = 422_000_000
scale = (ref_params / target) ** 0.5
print(f'μP scale: {scale:.3f}')
print(f'μP LR:    {ref_lr * scale:.3e}')
"
# Expected: μP scale ≈ 1.34, μP LR ≈ 8.04e-4 at 422M params.
```

## Skill 6: Add a new data source to the mixture

Edit the **universal** mixture at `data/shared_data/config/mixture.yaml` (in the LLM
umbrella — this project imports it via `sys.path`, it is not vendored here). Re-run:

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
- **`inference/generate.py --device`** defaults to `cuda` if available,
  else `cpu`. Pass `--device cpu` to run on a Mac or any non-CUDA host.


# DeepSeek-v3-Lite — Working Context

> Build target: `/Users/atandrabharati/Desktop/CoreProjects/LLM/DeepSeek-v3-Lite`
> Status snapshot: 2026-06-27 (per session clock).

## Scoping note

`MLA.md` (643-line MLA deep-dive) is a **human study doc**, not an agent artifact.
- Do **not** preload it into context.
- Do **not** read it on every MLA question — derive MLA answers from the code in `models/mla.py`.
- Read it only if the user explicitly asks for the conceptual deep-dive and is willing to spend the tokens.

Everything in this file is derived from code/configs/tests, not from `MLA.md`.

## Project snapshot

| Field | Value |
|---|---|
| Repo | DeepSeek-v3-Lite (single-A100 80GB faithful V3 reproduction) |
| Scale | ~422M params, 8.4B Chinchilla-optimal tokens |
| Wall budget | 13-15h A100 80GB SXM, 35-40% MFU target |
| Stack | PyTorch 2.x, TF32, SDPA/FA2, `torch.compile(max-autotune)`, dataclasses (not pydantic), safetensors ckpt |
| Vocab | 100,018 (`deepseek-coder-v2-lite`, has `byte_fallback`) |
| Topology | dim=768, n_layers=18 (2 dense + 16 MoE), 12 heads |
| MLA dims | `kv_lora_rank=192`, `qk_rope_head_dim=24`, `qk_nope_head_dim=48`, `v_head_dim=64`, `q_lora_rank=0` |
| MoE | 20 routed (top-4), 1 shared, `inter_dim=1536` (dense), `moe_inter_dim=384` |
| RoPE | θ=10000, factor=1.0 (no YaRN scaling at training length) |
| Optim | AdamW(fused), betas=(0.9, 0.95), wd=0.1, µP LR scaling (ref 6e-4 @ 757M → ~8.07e-4 @ 422M) |
| Steps | 512,000 total (with grad_accum=4 ≈ 2,048,000 micro-steps), 2000-step warmup, cosine to 5% |
| MTP | depth=1, loss_weight=0.3, shared output head, speculative decoder in `inference/speculative.py` |

## Hard rules (never violate)

1. **Never** suggest HF Trainer.
2. **Never** replace AuxLossFreeGate with an aux loss — it silently breaks MoE balance.
3. **Verify** vocab matches embedding dim and is 100,018 — tokenizer has `byte_fallback` (don't derive MLA answers from `MLA.md`, derive from `models/mla.py`).
4. **Never** disable the NaN guard without explicit user consent.

## Directory map

```
AGENTS.md, SKILLS.md, README.md, MLA.md, requirements.txt, opencode.json
configs/pretrain_a100_422m.yaml          # canonical 422M A100 recipe
data/prepare_data.py                     # download+tokenise+pack; mix weights in DATA_MIXES
models/
  __init__.py (empty)
  transformer.py                         # SwiGLUFFN, TransformerBlock, ParallelEmbedding, Transformer (+generate, _sample), count_parameters
  mla.py                                 # MultiHeadLatentAttention (SDPA + manual absorption paths, RoPE, kv/pe cache, _extend_rope, prefill_cache)
  moe.py                                 # AuxLossFreeGate, Expert, DeepSeekMoE (stacked bmm dispatch, _update_bias)
  mtp.py                                 # MTPBlock, MTPModule, MultiTokenPrediction (shared head + embed)
training/pretrain.py                     # TrainingConfig dataclass, PretrainDataset (single+sharded), Pretrainer (BF16, μP, NaN guard, MTP wrap, ckpt mgmt)
inference/
  generate.py                            # interactive generation + optional MTP speculative
  speculative.py                         # SpeculativeDecoder: accept ratio = min(1, p_main/p_draft), gated by threshold
utils/
  checkpoint.py                          # CheckpointManager (atomic safetensors+torch+json, dedup, latest_step, keep_last_n)
  distributed.py                         # single-GPU device helper
  logging.py                             # TrainingLogger (rolling-window print + optional wandb via WANDB_PROJECT env)
  memory.py                              # estimate_model_memory_gb, assert_fits_in_available_gpu
scripts/
  launch_a100.sh                         # pre-flight + nohup pretrain (PID, log tail)
  microbench_a100.py                     # measure peak VRAM
  step_time_a100.py                      # measure ms/step + MFU (target 30-45%)
tests/                                   # conftest.py (cfg/small_cfg/nested_cfg/training_cfg/tokens/targets + tmp helpers); CPU-only; 2021 lines
  test_models.py 766 lines (Transformer, MLA, SwiGLU, Expert, DeepSeekMoE, AuxLossFreeGate, MTP, Generation, CountParameters)
  test_training.py 592 lines (TrainingConfig, scheduler, dataset, pretrainer construction, ckpt roundtrip, train_step, MoE metric, YAML parse)
  test_inference.py 314 lines (generate_tokens, SpeculativeDecoder, generate_interactive, helpers)
  test_utils.py 349 lines (CheckpointManager save/load/MTP, memory estimation)
.github/workflows/ci.yml                 # CPU smoke (imports + forward); references nonexistent configs.pretrain_a100_422m.get_config (BUG)
assets/architecture_overview.png         # 200 KB diagram
graphify-out/                            # prior graphify run artefacts (gitignored, but present)
```

## Architectural invariants & quirks (load-bearing)

- **Weight tying**: `head.weight = embed.weight` (storage shared). Removing breaks generation quality. Affects param counting (handled in `count_parameters`).
- **Config shape**: `Pretrainer` accepts flat OR `{"model": {...}}`. MLA/MoE expect flat; Transformer unwraps if nested.
- **Causal mask cache**: `_mask_cache` keyed by `seqlen + device` on `Transformer`. Skipped when `seqlen == 1`.
- **Forward contracts**:
  - `Transformer.forward(x, start_pos=0, use_cache=True)` → `(B, S, V)`.
  - `Transformer.forward_with_hidden(x, start_pos, use_cache=False)` → `(logits, h_norm)` — used by MTP.
  - `MultiHeadLatentAttention.forward(x, start_pos, mask, use_cache)` returns `(B, S, dim)`.
  - `DeepSeekMoE.forward(x)` returns same shape; `_last_weights/_last_indices` stashed for `update_gate_bias` and balance metrics.
- **MLA KV cache**: `kv_cache (B, max_seq_len, kv_lora_rank)` + `pe_cache (B, max_seq_len, qk_rope_head_dim)`. `_ensure_cache` doubles capacity (min 16). `.detach()` on writes prevents autograd leakage. Cache reset via `reset_cache()`.
- **Two MLA paths**: `attn_impl="sdpa"` (default, FA2-friendly) materialises K_nope/V via bmm; manual path implements true absorption (per-batch bmm, 4× FLOPs, debug/ref). Cache is identical.
- **YaRN/RoPE**: `_extend_rope` doubles to `max_seq_len`; `rope_factor > 1.0` divides inv_freq and adds `mscale`. With factor=1.0 everything is bypassed.
- **MoE bias (load-bearing)**: `self.bias` is a **buffer** (not Parameter), so it does not receive autograd updates and is not in `state_dict()`. Updates are gated by `bias_update_every` (config: 1). Init zero. Update rule: subtract `speed` if count > avg·(1+upper); add if count < avg·(1-lower). Default upper/lower=0.10.
- **Stacked MoE dispatch** (`use_grouped="stacked"`): builds `_stacked_w1/w2/w3` lazily on first forward, one Python loop over experts (no segment-CuMoE / group-gemm). `_forward_grouped` exists for reference; both should agree per `test_stacked_and_grouped_agree`.
- **MTP wiring**: `MultiTokenPrediction(main_model)` registers `embed = main_model.embed`, shares `main_model.head` via `set_output_head`. MTP states saved under `mtp.` prefix in safetensors, optim state intentionally skipped.
- **NaN guard**: rolls back to last good checkpoint after `nan_guard_max_consecutive=5` consecutive NaN/Inf; resumes that step's scheduler/optimizer state.
- **µP LR**: `new_lr = mup_lr_reference * (mup_lr_reference_params / total) ** 0.5`. Note: scaling factor applied AFTER counting total params (post MTP-wrap), so may use mtp_total if wrapping is enabled.
- **CheckpointManager**: dedups by `data_ptr` (shared embed/head cloned); atomic write via mkstemp + os.replace; restores strict=False with warning. Lists complete checkpoints only (all 3 files present).
- **PretrainDataset**: single-file layout vs sharded layout (binary-search `_locate`, LRU 2-shard cache, cross-shard stitching). 2-shard eviction LRU is small — may thrash on small RAM.

## Known issues / sharp edges

- **CI is broken**: `.github/workflows/ci.yml` imports `from configs.pretrain_a100_422m import get_config` but that module doesn't exist (only `.yaml`). Smoke step will fail in CI.
- **MoE inter_dim vs shared expert count**: yaml says `n_shared_experts: 1` but README says 2 — single shared expert is what the code actually builds. README text outdated.
- **`n_dense_layers` semantics**: Transformer swaps to MoE for `layer_id >= n_dense_layers`. With `n_dense_layers=2`, layers 0-1 are dense SwiGLU, layers 2-17 are MoE (16 MoE layers). README claims "20 routed + 2 shared" but config+code have 20 routed + 1 shared.
- **`mtp_depth` in cfg**: `Pretrainer.__init__` looks up `config.model_config.get("model", config.model_config).get("mtp_depth", 0)` — must be at the top level of the model section (works with both flat and nested YAML).
- **Pretrainer mup_lr vs MTP**: µP scales by post-MTP total, which inflates ref count slightly. Document this if user asks about exact 8.07e-4.
- **`MTPModule` gradient through main model**: `forward_with_hidden` uses `use_cache=False` (correct for training); in `SpeculativeDecoder.generate_step` it uses `use_cache=True` — cache grows during draft.
- **`SpeculativeDecoder`**: no separate cache for MTP — relies on main model's KV cache. Acceptance ratio is computed once per step (greedy argmax comparison, not weighted rejection sampling).
- **Pretrainer `bias_update_every=10` vs `1`**: `pretrain_a100_422m.yaml` sets it to 1; default dataclass value is 10 — make sure yaml-derived config wins (it does).
- **Forward-mode check**: `Transformer.forward(...)` returns logits; when `use_cache=True` it caches for all positions in `start_pos:start_pos+seqlen`. Be careful when slicing.
- **`models/__init__.py` is empty** — imports are explicit (`from models.x import Y`).

## Test corpus (2021 lines, CPU-only, Mac-friendly)

- `tests/conftest.py` provides `cfg` (n_layers=2, dim=640), `small_cfg` (n_layers=2, dim=64, vocab=1024), `training_cfg`, `nested_cfg`, `tokens`, `targets`, `tmp_ckpt_dir`, `tmp_data_file`, `tmp_shard_dir`.
- Coverage highlights:
  - `TestMLA::test_sdpa_and_manual_agree` — absorption equivalence.
  - `TestDeepSeekMoE::test_stacked_and_grouped_agree` — dispatch equivalence.
  - `TestAuxLossFreeGate::test_bias_not_in_parameters` — bias is buffer (load-bearing invariant).
  - `TestAuxLossFreeGate::test_bias_in_state_dict` — bias is persisted via the buffer mechanism (registers survive state_dict via safetensors).
  - `TestPretrainerConstruction::test_optimizer_deduplicates` — handles tied weights.
  - `TestPretrainerConstruction::test_mup_lr_scaling` — verifies µP math.
  - `TestCheckpointRoundtrip::test_save_load_with_mtp` — MTP `mtp.` prefix round-trip.
  - `TestSpeculativeDecoder::*` — accept/reject + cache coherence.
- No GPU required; uses `device = torch.device("cpu")` in fixtures.

## Reproduction status

- 8.4B-token run **not yet started** (no `checkpoints/pretrain_a100/`).
- `data/pretrain_chinchilla/` does not exist; `data/prepare_data.py` must be invoked first.
- Speculative acceptance rate measured at ~0.8 on smoke tests.
- Pre-flight `scripts/launch_a100.sh` enforces ≥75 GB VRAM, ≥1 shard present.

## Open questions / TODOs user may ask about

1. Why is README "2 shared experts" when code has 1? (config vs prose drift)
2. Why is the CI workflow referencing a non-existent `configs.pretrain_a100_422m.get_config`?
3. µP scaling math: 6e-4 × sqrt(757_226_496 / total_params) — for ~422M → ~8.07e-4.
4. NaN guard threshold semantics — config flag default in dataclass differs from yaml.
5. Loss = main + 0.3 × mtp_loss (mean across depths); only `loss` is divided by `gradient_accumulation_steps` in the train_step.
6. `MTPBlock.attn` uses `nn.MultiheadAttention` (SDPA under the hood), not the MLA module — separate causal mask buffer, no KV cache.
7. `data/prepare_data.py` reads json/jsonl; for `lighteval/MATH` it looks for `problem` field — matches real schema.
8. `PretrainDataset` shard cache is LRU 2 (small). Cross-shard stitching uses `tolist()` on chunks — slow at scale.
9. `compute_loss` ignores mtp_pairs that are empty (e.g., seq too short) — returns main_loss for both auxiliary slots in that case.

## Useful commands

```bash
# Smoke test
python -c "
from models.transformer import Transformer
import torch
cfg = {'vocab_size':100018,'dim':768,'n_layers':2,'n_heads':12,'n_dense_layers':1,
       'n_routed_experts':4,'n_shared_experts':1,'n_activated_experts':2,
       'inter_dim':1024,'moe_inter_dim':256,'kv_lora_rank':64,'q_lora_rank':0,
       'qk_nope_head_dim':32,'qk_rope_head_dim':16,'v_head_dim':32,'max_seq_len':64,
       'rope_theta':10000,'rope_factor':1.0,'mscale':1.0,'attn_impl':'sdpa',
       'use_grouped':'stacked','weight_tying':True,'dtype':'bf16','mtp_depth':1,
       'mtp_loss_weight':0.3}
m = Transformer(cfg).cuda().to(torch.bfloat16)
x = torch.randint(0, cfg['vocab_size'], (2, 64), device='cuda')
y, aux = m.forward_with_hidden(x)
print(y.shape)  # (2, 64, 100018)
"

# Run tests
python -m pytest tests/ -q

# Step-time benchmark
python scripts/step_time_a100.py --steps 20 --warmup 5

# Launch full run
bash scripts/launch_a100.sh
```

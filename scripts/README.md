# Scripts

Run-once scripts for the 800M upgrade path. Each script is standalone and
self-contained.

| Script | What it does | When to run |
|---|---|---|
| `microbench_800m.py` | Build the 800M model, run one forward + backward, print peak VRAM. | Before the 1k-step dry-run. Catches OOMs. |
| `step_time_800m.py`  | Build + compile the 800M model, measure ms/step and MFU. | After microbench passes. Validates the Phase B1+B2 refactors. |
| `launch_800m.sh`     | Launch the full Chinchilla-20 run in the background, tail the first 50 log lines. | After both above pass and `data/pretrain_800m/` exists. |

## Order of operations

```bash
# 1. Build the data (one-time, ~1-2 days of downloads + tokenisation)
python data/prepare_data.py --stage pretrain \
    --tokenizer deepseek-ai/deepseek-coder-v2-lite \
    --shard-size-tokens 1000000000 \
    --max-tokens 15100000000 \
    --data-mix deepseek-v3 --include-extra

# 2. Microbench (5 min, A100)
python scripts/microbench_800m.py
# Expected: "measured peak = ~14 GB"
# If "peak > 70 GB": halve micro_batch_size and re-run

# 3. Step time (10 min, A100 — includes torch.compile warmup)
python scripts/step_time_800m.py --steps 20 --warmup 5
# Expected: "MFU 35-45%"
# If MFU < 25%: investigate (see script output for hints)

# 4. Launch (~8-9 days, A100)
bash scripts/launch_800m.sh
# - Sets up env, starts the run in background
# - Tails first 50 lines of log
# - Run continues in background; check `nvidia-smi` and `tail -f checkpoints/pretrain_800m/train.log`
```

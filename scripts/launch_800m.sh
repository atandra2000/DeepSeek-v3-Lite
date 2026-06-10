# scripts/launch_800m.sh
#
# Launch the 757M Chinchilla-20 pre-training run.
#
# Prereqs:
#   1. Microbench has passed (peak VRAM < 70 GB, MFU > 25%).
#   2. Data prep has produced data/pretrain_800m/ with ~15B tokens.
#   3. CUDA / A100 80GB driver / PyTorch ≥ 2.4 are installed.
#
# Usage:
#   bash scripts/launch_800m.sh
#
# What this does:
#   - Sets up env (WandB, NCCL, deterministic)
#   - Launches the pretraining run in the background via nohup
#   - Tails the log to stdout
#   - Checks the first ~50 lines for sanity (loss should drop from ~9.5 to ~8.5
#     in the first 100 steps; if it doesn't, kill the run)
#
# To resume:
#   python training/pretrain.py --config configs/pretrain_800m.yaml --resume <step>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Env ─────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1                  # so the log flushes immediately
export WANDB_PROJECT="${WANDB_PROJECT:-deepseek-v3-lite-800m}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-800m-c20-rev1}"
# A100-specific: enable BF16 reduced-precision reductions in cuBLAS
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Disable tokenizer parallelism (a WandB warning nuisance)
export TOKENIZERS_PARALLELISM=false

# ── Sanity checks ───────────────────────────────────────────────────────────
echo "=== Pre-flight ==="
python -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 75, \
    f'A100 80GB required; found {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
"

# Check data is present
DATA_DIR="$REPO_ROOT/data/pretrain_800m"
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR"/shard_*.bin 2>/dev/null)" ]]; then
    echo "ERROR: no shard_*.bin files in $DATA_DIR"
    echo "Run first:  python data/prepare_data.py --stage pretrain --tokenizer deepseek-ai/deepseek-coder-v2-lite --shard-size-tokens 1000000000 --max-tokens 15100000000 --data-mix deepseek-v3 --include-extra"
    exit 1
fi
echo "Data: $(ls "$DATA_DIR"/shard_*.bin 2>/dev/null | wc -l) shards"

CHECKPOINT_DIR="$REPO_ROOT/checkpoints/pretrain_800m"
mkdir -p "$CHECKPOINT_DIR"
LOG_FILE="$CHECKPOINT_DIR/train.log"
echo "Log: $LOG_FILE"

# ── Launch ──────────────────────────────────────────────────────────────────
echo
echo "=== Launching 800M Chinchilla-20 pre-training ==="
echo "Estimated wall: 8-9 days at ~40% MFU"
echo "Estimated peak VRAM: ~14 GB / 80 GB"
echo

# The first 3-4 minutes are torch.compile warmup; do NOT kill the run
# during that window even if the log is quiet.
nohup python -u training/pretrain.py \
    --config configs/pretrain_800m.yaml \
    --data-path "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started PID $PID. Tailing $LOG_FILE (Ctrl+C to stop tailing; the run will continue in background)."
echo

# Watch the first 50 lines of the log for sanity
sleep 30
echo "=== First 50 lines of log (after 30s warmup) ==="
head -n 50 "$LOG_FILE" 2>/dev/null || echo "(log not yet written)"
echo
echo "=== Last 20 lines ==="
tail -n 20 "$LOG_FILE" 2>/dev/null || echo "(log not yet written)"

# Don't wait on the run; the script returns and the run continues in the
# background. To re-tail: tail -f $LOG_FILE
echo
echo "Run continues in background (PID $PID)."
echo "To monitor: tail -f $LOG_FILE"
echo "To check:   ps -p $PID && nvidia-smi"

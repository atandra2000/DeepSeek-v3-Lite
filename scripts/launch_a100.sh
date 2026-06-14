# scripts/launch_a100.sh
#
# Launch the 422M Chinchilla-style pre-training run on 1× A100 80GB SXM.
#
# Prereqs:
#   1. Microbench has passed (peak VRAM < 40 GB, MFU > 30%).
#   2. Data prep has produced data/pretrain_chinchilla/ with ~8.4B tokens.
#   3. CUDA / A100 / PyTorch ≥ 2.4 are installed.
#
# Usage:
#   bash scripts/launch_a100.sh
#
# Estimated wall: 13-15 h on A100 80GB (35-40% MFU).
#
# To resume:
#   python training/pretrain.py --config configs/pretrain_a100_422m.yaml --resume <step>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Env ─────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1                  # log flushes immediately
export WANDB_PROJECT="${WANDB_PROJECT:-deepseek-v3-lite-a100}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-422m-chinchilla}"
export TOKENIZERS_PARALLELISM=false
# A100 max-autotune compilation mode
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}"

# ── Sanity checks ───────────────────────────────────────────────────────────
echo "=== Pre-flight ==="
python -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available'
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert mem_gb >= 75, f'Need at least 75 GB VRAM; found {mem_gb:.0f} GB'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {mem_gb:.0f} GB')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
"

# Check data is present
DATA_DIR="$REPO_ROOT/data/pretrain_chinchilla"
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR"/shard_*.bin 2>/dev/null)" ]]; then
    echo "ERROR: no shard_*.bin files in $DATA_DIR"
    echo "Run first:  python data/prepare_data.py --stage pretrain --tokenizer deepseek-ai/deepseek-coder-v2-lite --shard-size-tokens 50000000 --max-tokens 8400000000 --data-mix deepseek-v3 --include-extra --output-dir data/pretrain_chinchilla"
    exit 1
fi
echo "Data: $(ls "$DATA_DIR"/shard_*.bin 2>/dev/null | wc -l) shards"
echo "Data size: $(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)"

CHECKPOINT_DIR="$REPO_ROOT/checkpoints/pretrain_a100"
mkdir -p "$CHECKPOINT_DIR"
LOG_FILE="$CHECKPOINT_DIR/train.log"
echo "Log: $LOG_FILE"

# ── Launch ──────────────────────────────────────────────────────────────────
echo
echo "=== Launching 422M Chinchilla pre-training ==="
echo "Estimated wall: 13-15 hours on A100 80GB"
echo "Estimated peak VRAM: ~30-35 GB / 80 GB"
echo

nohup python -u training/pretrain.py \
    --config configs/pretrain_a100_422m.yaml \
    --data-path "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started PID $PID. Tailing $LOG_FILE (Ctrl+C to stop tailing; the run will continue in background)."
echo

# Watch the first 50 lines
sleep 30
echo "=== First 50 lines of log (after 30s warmup) ==="
head -n 50 "$LOG_FILE" 2>/dev/null || echo "(log not yet written)"
echo
echo "=== Last 20 lines ==="
tail -n 20 "$LOG_FILE" 2>/dev/null || echo "(log not yet written)"

echo
echo "Run continues in background (PID $PID)."
echo "To monitor: tail -f $LOG_FILE"
echo "To check:   ps -p $PID && nvidia-smi"

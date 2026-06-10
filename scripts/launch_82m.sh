# scripts/launch_82m.sh
#
# Launch the 82M code-model pre-training run.
#
# Prereqs:
#   1. Microbench has passed (peak VRAM < 18 GB, MFU > 15%).
#   2. Data prep has produced data/pretrain_code-82m/ with ~1.2B tokens.
#   3. CUDA / RTX 4090 / PyTorch >= 2.4 are installed.
#
# Usage:
#   bash scripts/launch_82m.sh
#
# What this does:
#   - Sets up env (WandB, NCCL, deterministic)
#   - Launches the pretraining run in the background via nohup
#   - Tails the log to stdout
#   - Checks the first ~50 lines for sanity (loss should drop from ~9.5 to ~8.5
#     in the first 100 steps; if it doesn't, kill the run)
#
# To resume:
#   python training/pretrain.py --config configs/pretrain_82m.yaml --resume <step>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Env ─────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1                  # so the log flushes immediately
export WANDB_PROJECT="${WANDB_PROJECT:-deepseek-v3-lite-82m}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-82m-code-c1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false

# ── Sanity checks ───────────────────────────────────────────────────────────
echo "=== Pre-flight ==="
python -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available'
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert mem_gb >= 20, f'Need at least 20 GB VRAM; found {mem_gb:.0f} GB'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {mem_gb:.0f} GB')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
"

# Check data is present
DATA_DIR="$REPO_ROOT/data/pretrain_code-82m"
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR"/shard_*.bin 2>/dev/null)" ]]; then
    echo "ERROR: no shard_*.bin files in $DATA_DIR"
    echo "Run first:  python data/prepare_data.py --stage pretrain --tokenizer deepseek-ai/deepseek-coder-v2-lite --shard-size-tokens 50000000 --max-tokens 1176000000 --data-mix code-82m --output-dir data/pretrain_code-82m"
    exit 1
fi
echo "Data: $(ls "$DATA_DIR"/shard_*.bin 2>/dev/null | wc -l) shards"
echo "Data size: $(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)"

CHECKPOINT_DIR="$REPO_ROOT/checkpoints/pretrain_82m"
mkdir -p "$CHECKPOINT_DIR"
LOG_FILE="$CHECKPOINT_DIR/train.log"
echo "Log: $LOG_FILE"

# ── Launch ──────────────────────────────────────────────────────────────────
echo
echo "=== Launching 82M code-model pre-training ==="
echo "Estimated wall: 6-10 hours on RTX 4090 24GB"
echo "Estimated peak VRAM: ~3-5 GB / 24 GB"
echo

nohup python -u training/pretrain.py \
    --config configs/pretrain_82m.yaml \
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

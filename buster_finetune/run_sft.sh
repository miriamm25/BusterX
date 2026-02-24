#!/usr/bin/env bash
# run_sft.sh
# =============================================================================
# Launch script for BusterX++ LoRA SFT on Deepfake-Eval-2024.
#
# Prerequisites:
#   bash setup_env.sh
#   python prepare_deepfake_eval.py
#
# Modes:
#   bash run_sft.sh           →  single H200 (recommended)
#   bash run_sft.sh --2gpu    →  2× H200, DeepSpeed ZeRO-2
#   bash run_sft.sh --smoke   →  10-step smoke test (no GPU quota needed)
#
# NOTE: In the 3-stage DAPO pipeline this script serves as Stage 2
# (thinking-mode fusion).  Run it AFTER Stage 1 DAPO converges.
# For standalone SFT (without DAPO) run it directly.
#
# Live log monitoring:
#   tail -f checkpoints/sft/logs/train_*.log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_DATA="data/train.jsonl"
VAL_DATA="data/val.jsonl"
OUTPUT_DIR="checkpoints/sft"
NUM_EPOCHS=3
EVAL_STEPS=200
SAVE_STEPS=200
MAX_PIXELS=589824   # ≈768². Raise to 1048576 (1024²) if VRAM allows.

# ---- Parse flags -----------------------------------------------------------
MODE="single"
for arg in "$@"; do
    case $arg in
        --2gpu)   MODE="2gpu"  ;;
        --smoke)  MODE="smoke" ;;
    esac
done

# ---- Guard: data must exist ------------------------------------------------
if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
    echo "[ERROR] JSONL data not found. Run first:"
    echo "  python prepare_deepfake_eval.py"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")/sft/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/logs/train_${TS}.log"
echo "Log: $LOG_FILE"

# ---- Launch ----------------------------------------------------------------
case $MODE in

  smoke)
    echo "=== Smoke test (10 steps, 20 samples) ==="
    python train_sft_lora.py \
        --train_data  "$TRAIN_DATA"        \
        --val_data    "$VAL_DATA"          \
        --output_dir  "checkpoints/smoke"  \
        --max_steps   10                   \
        --max_samples 20                   \
        --eval_steps  5                    \
        --save_steps  5                    \
        2>&1 | tee "checkpoints/smoke/logs/train_${TS}.log"
    echo "Smoke test passed."
    ;;

  2gpu)
    echo "=== 2× H200, DeepSpeed ZeRO-2 ==="
    deepspeed --num_gpus=2 train_sft_lora.py \
        --train_data  "$TRAIN_DATA"                   \
        --val_data    "$VAL_DATA"                     \
        --output_dir  "$OUTPUT_DIR"                   \
        --num_epochs  $NUM_EPOCHS                     \
        --eval_steps  $EVAL_STEPS                     \
        --save_steps  $SAVE_STEPS                     \
        --max_pixels  $MAX_PIXELS                     \
        --deepspeed   configs/ds_config_zero2.json    \
        2>&1 | tee "$LOG_FILE"
    ;;

  single|*)
    echo "=== Single H200 ==="
    python train_sft_lora.py \
        --train_data  "$TRAIN_DATA"   \
        --val_data    "$VAL_DATA"     \
        --output_dir  "$OUTPUT_DIR"   \
        --num_epochs  $NUM_EPOCHS     \
        --eval_steps  $EVAL_STEPS     \
        --save_steps  $SAVE_STEPS     \
        --max_pixels  $MAX_PIXELS     \
        2>&1 | tee "$LOG_FILE"
    ;;

esac

echo ""
echo "Training complete. Outputs:"
echo "  checkpoints/sft/lora_adapter/  ← LoRA weights (~50-100 MB)"
echo "  checkpoints/sft/merged/        ← full merged model (~15 GB)"
echo ""
echo "Monitor training:"
echo "  tensorboard --logdir $OUTPUT_DIR --port 6006"

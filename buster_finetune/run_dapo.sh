#!/usr/bin/env bash
# run_dapo.sh
# =============================================================================
# Launch script for BusterX++ DAPO training on Deepfake-Eval-2024.
#
# Prerequisites:
#   bash setup_env.sh
#   python prepare_deepfake_eval.py        # creates data/train.jsonl etc.
#
# Modes:
#   bash run_dapo.sh                       # Stage 1 only, single H200
#   bash run_dapo.sh --stage all           # Full 3-stage pipeline
#   bash run_dapo.sh --stage 3             # Only Stage 3 (needs existing stage2/)
#   bash run_dapo.sh --2gpu                # 2× H200 with DeepSpeed ZeRO-2
#   bash run_dapo.sh --smoke               # 5-step smoke test (no GPU quota needed)
#
# Live log monitoring (in another terminal):
#   tail -f checkpoints/dapo/logs/train_*.log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_DATA="data/train.jsonl"
VAL_DATA="data/val.jsonl"
OUTPUT_DIR="checkpoints/dapo"
NUM_EPOCHS=3
SAVE_STEPS=500
EVAL_STEPS=500
MAX_PIXELS=589824    # ≈768². Raise to 1048576 (1024²) on more VRAM.
MAX_NEW_TOKENS=1500

# ---- Parse flags -----------------------------------------------------------
MODE="single"
STAGE="1"
for arg in "$@"; do
    case $arg in
        --2gpu)         MODE="2gpu"   ;;
        --smoke)        MODE="smoke"  ;;
        --stage)        ;;            # handled below
        --stage=*)      STAGE="${arg#*=}" ;;
    esac
done
# Manual two-arg parsing for --stage VALUE
for i in "$@"; do
    if [[ "$PREV" == "--stage" ]]; then
        STAGE="$i"
    fi
    PREV="$i"
done

# ---- Guard: data must exist ------------------------------------------------
if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
    echo "[ERROR] JSONL data not found. Run first:"
    echo "  python prepare_deepfake_eval.py"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"

# ---- Timestamp for log file ------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/logs/dapo_${MODE}_stage${STAGE}_${TS}.log"

echo "=== BusterX++ DAPO | mode=$MODE | stage=$STAGE ==="
echo "Log: $LOG_FILE"
echo ""

# ---- Launch ----------------------------------------------------------------
case $MODE in

  smoke)
    echo "=== Smoke test (5 steps, 10 samples) ==="
    python train_dapo.py \
        --stage       1              \
        --train_data  "$TRAIN_DATA"  \
        --val_data    "$VAL_DATA"    \
        --output_dir  "checkpoints/dapo_smoke" \
        --max_steps   5              \
        --max_samples 10             \
        --save_steps  5              \
        --eval_steps  5              \
        2>&1 | tee "$OUTPUT_DIR/logs/dapo_smoke_${TS}.log"
    echo "Smoke test passed."
    ;;

  2gpu)
    echo "=== 2× H200, DeepSpeed ZeRO-2 ==="
    deepspeed --num_gpus=2 train_dapo.py \
        --stage           "$STAGE"          \
        --train_data      "$TRAIN_DATA"     \
        --val_data        "$VAL_DATA"       \
        --output_dir      "$OUTPUT_DIR"     \
        --num_epochs      $NUM_EPOCHS       \
        --save_steps      $SAVE_STEPS       \
        --eval_steps      $EVAL_STEPS       \
        --max_pixels      $MAX_PIXELS       \
        --max_new_tokens  $MAX_NEW_TOKENS   \
        --deepspeed       configs/ds_config_zero2.json \
        2>&1 | tee "$LOG_FILE"
    ;;

  single|*)
    echo "=== Single H200 ==="
    python train_dapo.py \
        --stage           "$STAGE"        \
        --train_data      "$TRAIN_DATA"   \
        --val_data        "$VAL_DATA"     \
        --output_dir      "$OUTPUT_DIR"   \
        --num_epochs      $NUM_EPOCHS     \
        --save_steps      $SAVE_STEPS     \
        --eval_steps      $EVAL_STEPS     \
        --max_pixels      $MAX_PIXELS     \
        --max_new_tokens  $MAX_NEW_TOKENS \
        2>&1 | tee "$LOG_FILE"
    ;;

esac

echo ""
echo "Training complete. Outputs in $OUTPUT_DIR/"
echo ""
echo "Stage checkpoints:"
echo "  $OUTPUT_DIR/stage1/lora_adapter/   ← Stage 1 LoRA"
echo "  $OUTPUT_DIR/stage2/lora_adapter/   ← Stage 2 LoRA (mode fusion)"
echo "  $OUTPUT_DIR/stage3/lora_adapter/   ← Stage 3 LoRA (FINAL)"
echo ""
echo "Evaluate final model:"
echo "  python eval_on_test.py \\"
echo "      --lora  $OUTPUT_DIR/stage3/lora_adapter \\"
echo "      --data  data/test.jsonl"
echo ""
echo "Monitor training:"
echo "  tail -f $LOG_FILE"

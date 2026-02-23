#!/usr/bin/env bash
# setup_env.sh
# =============================================================================
# Environment setup for BusterX++ SFT on Deepfake-Eval-2024.
# Run once on the server before any training or evaluation.
#
# Usage:
#   bash setup_env.sh
#
# Assumes: Python >= 3.10, CUDA 12.1 (adjust --index-url for other CUDA vers.)
# =============================================================================

set -euo pipefail

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing HuggingFace stack ==="
# Pin transformers to a release that ships Qwen2.5-VL support.
# 4.49+ is required; 4.51.x is stable as of early 2026.
pip install "transformers>=4.49.0,<4.53.0" accelerate

echo "=== Installing PEFT (LoRA) ==="
pip install "peft>=0.10.0"

echo "=== Installing DeepSpeed (optional – only needed for multi-GPU) ==="
pip install deepspeed

echo "=== Installing video / data utils ==="
pip install decord Pillow tqdm pandas

echo "=== Installing logging ==="
pip install tensorboard

echo "=== Verifying imports ==="
python - <<'EOF'
import torch, transformers, peft, decord, pandas
print(f"  torch       : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"  transformers: {transformers.__version__}")
print(f"  peft        : {peft.__version__}")
print(f"  decord      : {decord.__version__}")
print(f"  pandas      : {pandas.__version__}")
try:
    import deepspeed
    print(f"  deepspeed   : {deepspeed.__version__}")
except ImportError:
    print("  deepspeed   : not installed (single-GPU mode only)")
print("OK – all imports successful.")
EOF

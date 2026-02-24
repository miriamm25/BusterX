#!/usr/bin/env bash
#SBATCH --job-name=busterx_sft
#SBATCH --output=checkpoints/sft/logs/slurm_%j.out
#SBATCH --error=checkpoints/sft/logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1        # single H200; use gpu:h200:2 + --ntasks=2 for 2-GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# =============================================================================
# SLURM wrapper for BusterX++ LoRA SFT (Stage 2 of the DAPO pipeline,
# or standalone fine-tuning).
#
# Submit:
#   sbatch slurm_sft.sh
#   sbatch slurm_sft.sh --2gpu
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPUs      : $(nvidia-smi --list-gpus | head -4)"
echo "Start time: $(date)"
echo ""

mkdir -p checkpoints/sft/logs

# Pass remaining CLI flags (e.g. --2gpu) through to run_sft.sh
bash run_sft.sh "$@"

echo ""
echo "Done. End time: $(date)"

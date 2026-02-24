#!/usr/bin/env bash
#SBATCH --job-name=busterx_dapo
#SBATCH --output=checkpoints/dapo/logs/slurm_%j.out
#SBATCH --error=checkpoints/dapo/logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1        # single H200; use gpu:h200:2 for 2-GPU run
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G                # DAPO needs more RAM (generates 4× per clip)
#SBATCH --time=48:00:00
# =============================================================================
# SLURM wrapper for BusterX++ DAPO training.
#
# Submit (single stage):
#   sbatch slurm_dapo.sh --stage 1
#   sbatch slurm_dapo.sh --stage 3
#
# Submit (full pipeline, ~30 h on single H200):
#   sbatch slurm_dapo.sh --stage all
#
# Submit (2-GPU):
#   sbatch --gres=gpu:h200:2 slurm_dapo.sh --2gpu --stage 1
#
# Submit (smoke test, 5 steps):
#   sbatch slurm_dapo.sh --smoke
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPUs      : $(nvidia-smi --list-gpus | head -4)"
echo "Start time: $(date)"
echo ""

mkdir -p checkpoints/dapo/logs

# Pass all CLI flags through to run_dapo.sh
bash run_dapo.sh "$@"

echo ""
echo "Done. End time: $(date)"

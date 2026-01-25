# BusterX++ Retraining Pipeline

Scripts for retraining BusterX++ to improve TikTok deepfake detection.

## Problem Statement

- Current BusterX++ performance on TikTok videos: **43.8% accuracy**
- Critical failure on lip-sync detection: **27.5%**
- 82% of errors are false negatives (predicts REAL when FAKE)
- Model trained on text-to-video generators, but TikTok deepfakes use face-swaps/lip-sync

## Solution

Retrain BusterX++ with:
1. Mixed data sources (GenBuster + face-swap datasets + TikTok videos)
2. Watermarks intact to train model to ignore them
3. Both SFT and DAPO training approaches

## Files

```
busterx_retrain/
├── __init__.py              # Package init
├── prepare_data.py          # Data preparation & standardization
├── reward_functions.py      # DAPO reward functions
├── train_sft.py             # Supervised Fine-Tuning with LoRA
├── train_dapo.py            # Full DAPO 3-stage pipeline
├── evaluate.py              # Model evaluation
├── configs/
│   ├── sft_config.yaml      # SFT configuration
│   ├── dapo_config.yaml     # DAPO configuration
│   └── ds_config.json       # DeepSpeed config
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers peft trl accelerate bitsandbytes
pip install torch torchvision decord
pip install deepspeed  # For multi-GPU training
pip install pyyaml tqdm
```

### 2. Prepare Data

```bash
# From GenBuster-200K-mini
python prepare_data.py \
    --source ../data/GenBuster-200K-mini/GenBuster-200K-mini \
    --output ./training_data \
    --workers 8

# Quick test (limited videos)
python prepare_data.py \
    --source ../data/GenBuster-200K-mini/GenBuster-200K-mini \
    --output ./training_data \
    --max-videos 100 \
    --skip-processing
```

### 3. Train with SFT (Recommended First)

```bash
# Single GPU
python train_sft.py --config configs/sft_config.yaml

# Quick test run
python train_sft.py \
    --config configs/sft_config.yaml \
    --max_samples 100 \
    --num_epochs 1

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=4 train_sft.py \
    --config configs/sft_config.yaml \
    --deepspeed configs/ds_config.json
```

### 4. Train with DAPO (Better Results)

```bash
# Stage 1 only (Foundation RL)
python train_dapo.py --stage 1 --config configs/dapo_config.yaml

# Full 3-stage pipeline
python train_dapo.py --stage all --config configs/dapo_config.yaml
```

### 5. Evaluate

```bash
# Evaluate on test set
python evaluate.py \
    --model ./checkpoints/sft/final \
    --data ./training_data/test.jsonl \
    --output results_sft.json

# Evaluate on TikTok videos
python evaluate.py \
    --model ./checkpoints/sft/final \
    --tiktok ../deepfakes/ \
    --output results_tiktok.json

# Compare baseline vs trained
python evaluate.py --model l8cv/BusterX_plusplus --data ./training_data/test.jsonl --name baseline
python evaluate.py --model ./checkpoints/sft/final --data ./training_data/test.jsonl --name trained
```

## Training Approaches

### Approach 1: SFT with LoRA (Simpler, Faster)
- Standard supervised fine-tuning
- Uses LoRA (trains ~0.1% of parameters)
- Good for validating the pipeline
- Expected improvement: 10-20% accuracy gain

### Approach 2: DAPO (Better Results)
3-stage pipeline from BusterX++ paper:

| Stage | Purpose | % of Training |
|-------|---------|---------------|
| 1 | Foundation RL (classification) | 70% |
| 2 | Thinking Mode Fusion (SFT) | 5% |
| 3 | Advanced RL (reasoning quality) | 25% |

Key DAPO innovations:
- **Clip-Higher**: Removes upper PPO clipping
- **Dynamic Sampling**: Ensures mixed correct/wrong per batch
- **Token-Level Loss**: Masks negative advantage tokens
- **Overlong Reward Shaping**: Graduated penalty for long responses

## Reward Functions

| Reward | Stage | Purpose |
|--------|-------|---------|
| r_format | 1, 3 | Correct `<think>/<answer>` format |
| r_overlong | 1, 3 | Penalize responses > 600 tokens |
| r_accuracy | 1, 3 | Binary classification accuracy |
| r_hybrid | 3 | Think/no-think mode compliance |
| r_thinking | 3 | Reasoning quality (SophiaVL model) |

## Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| SFT (single GPU) | 24GB+ | A100 recommended |
| DAPO (single GPU) | 40GB+ | Generates G=4 samples |
| Multi-GPU | 4x A100 | Use DeepSpeed ZeRO-2 |

## Expected Results

| Metric | Baseline | After SFT | After DAPO |
|--------|----------|-----------|------------|
| TikTok Accuracy | 43.8% | ~60-70% | ~80%+ |
| False Negative Rate | 82% | ~40-50% | ~20-30% |
| Format Compliance | ~90% | ~95%+ | ~98%+ |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing: true`
- Use DeepSpeed with CPU offload

### Slow Training
- Reduce `num_frames` to 8
- Reduce `max_new_tokens` to 1000
- Use `max_samples` to limit data

### Poor Results
- Ensure balanced data (50% real, 50% fake)
- Check format compliance in outputs
- Verify data paths in config

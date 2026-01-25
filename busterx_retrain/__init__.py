"""
BusterX++ Retraining Pipeline
=============================
Scripts for retraining BusterX++ on TikTok deepfake detection.

Modules:
- prepare_data: Data preparation and standardization
- reward_functions: DAPO reward functions (format, accuracy, thinking, etc.)
- train_sft: Supervised Fine-Tuning with LoRA
- train_dapo: Full DAPO 3-stage training pipeline
- evaluate: Model evaluation and metrics
"""

from .reward_functions import (
    RewardConfig,
    FormatReward,
    OverlongReward,
    AccuracyReward,
    HybridReward,
    ThinkingReward,
    CombinedReward,
    compute_advantages,
    dynamic_sampling_filter,
)

__version__ = "1.0.0"
__all__ = [
    "RewardConfig",
    "FormatReward",
    "OverlongReward",
    "AccuracyReward",
    "HybridReward",
    "ThinkingReward",
    "CombinedReward",
    "compute_advantages",
    "dynamic_sampling_filter",
]

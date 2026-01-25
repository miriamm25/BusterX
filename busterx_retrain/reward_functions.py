#!/usr/bin/env python3
"""
BusterX++ Reward Functions for DAPO Training
=============================================
Implements the 5 reward functions from the BusterX++ paper:
1. r_format: Checks proper <think></think><answer></answer> format
2. r_overlong: Penalizes responses > L_max tokens (graduated penalty)
3. r_accuracy: Binary accuracy reward based on correct classification
4. r_hybrid: Rewards thinking/no-thinking mode compliance
5. r_thinking: Quality of reasoning (via SophiaVL thinking reward model)

Reference: BusterX++ paper Section 3.2
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    # Format reward
    format_reward: float = 1.0
    format_penalty: float = 0.0

    # Overlong reward shaping
    L_max: int = 600  # Maximum tokens before penalty
    L_cache: int = 256  # Tokens beyond L_max for graduated penalty

    # Accuracy reward
    accuracy_reward: float = 1.0
    accuracy_penalty: float = 0.0

    # Hybrid mode reward
    hybrid_think_reward: float = 1.0
    hybrid_nothink_reward: float = 1.0

    # Thinking reward model
    thinking_model_path: str = "SophiaVL/SophiaVL-R1-Thinking-Reward-Model-3B"
    use_thinking_reward: bool = True


class FormatReward:
    """
    Format Reward (r_format)

    Verifies that responses follow the BusterX++ format:
    <think> reasoning </think><answer> A or B </answer>

    Returns:
        1.0 if format is correct
        0.0 if format is incorrect
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
        self.answer_pattern = re.compile(r'<answer>\s*([AB])\s*\)?</answer>', re.IGNORECASE)
        self.full_pattern = re.compile(
            r'<think>.*?</think>\s*<answer>\s*[AB]\s*\)?</answer>',
            re.DOTALL | re.IGNORECASE
        )

    def __call__(self, response: str) -> float:
        """Calculate format reward for a single response."""
        if self.full_pattern.search(response):
            return self.config.format_reward
        return self.config.format_penalty

    def batch_compute(self, responses: List[str]) -> torch.Tensor:
        """Calculate format rewards for a batch of responses."""
        return torch.tensor([self(r) for r in responses])


class OverlongReward:
    """
    Overlong Reward Shaping (r_overlong)

    From paper: Graduated penalty for responses exceeding L_max tokens.

    For response length L:
    - If L <= L_max: reward = 0 (no penalty)
    - If L > L_max: reward = -min(1, (L - L_max) / L_cache)
    """

    def __init__(self, config: RewardConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: ~4 chars per token
            return len(text) // 4

    def __call__(self, response: str) -> float:
        """Calculate overlong reward for a single response."""
        L = self.count_tokens(response)

        if L <= self.config.L_max:
            return 0.0

        penalty = min(1.0, (L - self.config.L_max) / self.config.L_cache)
        return -penalty

    def batch_compute(self, responses: List[str]) -> torch.Tensor:
        """Calculate overlong rewards for a batch of responses."""
        return torch.tensor([self(r) for r in responses])


class AccuracyReward:
    """
    Accuracy Reward (r_accuracy)

    Binary reward based on correct classification:
    - 1.0 if prediction matches ground truth
    - 0.0 if prediction is wrong

    Labels: A (or 0) = Real, B (or 1) = Fake
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.answer_pattern = re.compile(r'<answer>\s*([AB])\s*\)?</answer>', re.IGNORECASE)

    def extract_prediction(self, response: str) -> Optional[str]:
        """Extract prediction (A or B) from response."""
        match = self.answer_pattern.search(response)
        if match:
            return match.group(1).upper()

        # Fallback: look for A) or B) near the end
        fallback = re.search(r'\b([AB])\)', response[-100:], re.IGNORECASE)
        if fallback:
            return fallback.group(1).upper()

        return None

    def __call__(self, response: str, label: Union[int, str]) -> float:
        """Calculate accuracy reward."""
        prediction = self.extract_prediction(response)

        if prediction is None:
            return self.config.accuracy_penalty

        if isinstance(label, int):
            expected = "A" if label == 0 else "B"
        else:
            expected = "A" if label.lower() == "real" else "B"

        return self.config.accuracy_reward if prediction == expected else self.config.accuracy_penalty

    def batch_compute(self, responses: List[str], labels: List[Union[int, str]]) -> torch.Tensor:
        """Calculate accuracy rewards for a batch."""
        return torch.tensor([self(r, l) for r, l in zip(responses, labels)])


class HybridReward:
    """
    Hybrid Mode Reward (r_hybrid)

    For Stage 2+, supports thinking mode switching:
    - /think mode: Response MUST include <think> tags
    - /no_think mode: Response should NOT include <think> tags
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

    def __call__(self, response: str, mode: str) -> float:
        """Calculate hybrid mode reward."""
        has_thinking = bool(self.think_pattern.search(response))

        if mode == "think":
            return self.config.hybrid_think_reward if has_thinking else 0.0
        elif mode == "no_think":
            return self.config.hybrid_nothink_reward if not has_thinking else 0.0
        return 0.0

    def batch_compute(self, responses: List[str], modes: List[str]) -> torch.Tensor:
        """Calculate hybrid rewards for a batch."""
        return torch.tensor([self(r, m) for r, m in zip(responses, modes)])


class ThinkingReward:
    """
    Thinking Quality Reward (r_thinking)

    Uses SophiaVL-R1-Thinking-Reward-Model-3B to evaluate reasoning quality.
    Falls back to heuristic scoring if model unavailable.
    """

    def __init__(self, config: RewardConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self):
        """Lazy load the thinking reward model."""
        if self._loaded or not self.config.use_thinking_reward:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            print(f"Loading thinking reward model: {self.config.thinking_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.thinking_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.config.thinking_model_path,
                trust_remote_code=True
            )
            self._loaded = True
            print("Thinking reward model loaded!")
        except Exception as e:
            print(f"Warning: Could not load thinking reward model: {e}")
            self._loaded = False

    def extract_thinking(self, response: str) -> str:
        """Extract thinking content from response."""
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def heuristic_reward(self, thinking: str) -> float:
        """Heuristic thinking quality reward when model unavailable."""
        if not thinking:
            return 0.0

        score = 0.0
        thinking_lower = thinking.lower()

        # Length check
        if len(thinking) >= 100:
            score += 0.2
        if len(thinking) >= 300:
            score += 0.2

        # Thinking indicators
        thinking_phrases = [
            "let me", "hmm", "wait", "oh", "i see", "interesting",
            "notice", "observe", "examine", "looking at", "checking", "analyzing"
        ]
        indicator_count = sum(1 for p in thinking_phrases if p in thinking_lower)
        score += min(0.2, indicator_count * 0.05)

        # Artifact/analysis terms
        artifact_terms = [
            "artifact", "boundary", "edge", "temporal", "motion",
            "lighting", "shadow", "texture", "inconsisten", "blur",
            "frame", "smooth", "unnatural", "synthetic"
        ]
        artifact_count = sum(1 for t in artifact_terms if t in thinking_lower)
        score += min(0.2, artifact_count * 0.04)

        # Self-reflection
        reflection_phrases = ["but", "however", "although", "on the other hand", "let me reconsider"]
        if any(p in thinking_lower for p in reflection_phrases):
            score += 0.2

        return min(1.0, score)

    def __call__(self, response: str) -> float:
        """Calculate thinking quality reward."""
        thinking = self.extract_thinking(response)
        if not thinking:
            return 0.0

        if self.config.use_thinking_reward and self._loaded and self.model is not None:
            try:
                inputs = self.processor(thinking, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    reward = torch.sigmoid(outputs.logits[:, -1]).item()
                return reward
            except Exception as e:
                print(f"Thinking model error: {e}, using heuristic")

        return self.heuristic_reward(thinking)

    def batch_compute(self, responses: List[str]) -> torch.Tensor:
        """Calculate thinking rewards for a batch."""
        return torch.tensor([self(r) for r in responses])


class CombinedReward:
    """
    Combined Reward Calculator

    Stage 1: r_total = r_format + r_overlong + r_accuracy
    Stage 2: Uses SFT, not RL rewards
    Stage 3: r_total = r_format + r_overlong + r_accuracy + r_hybrid + r_thinking
    """

    def __init__(self, config: RewardConfig, tokenizer=None, stage: int = 1, device: str = "cuda"):
        self.config = config
        self.stage = stage

        self.format_reward = FormatReward(config)
        self.overlong_reward = OverlongReward(config, tokenizer)
        self.accuracy_reward = AccuracyReward(config)
        self.hybrid_reward = HybridReward(config)
        self.thinking_reward = ThinkingReward(config, device)

        if stage == 3:
            self.thinking_reward.load_model()

    def compute(self, response: str, label: Union[int, str], mode: str = "think") -> Dict[str, float]:
        """Compute all applicable rewards for a single response."""
        rewards = {
            "r_format": self.format_reward(response),
            "r_overlong": self.overlong_reward(response),
            "r_accuracy": self.accuracy_reward(response, label),
        }

        if self.stage == 1:
            rewards["r_total"] = rewards["r_format"] + rewards["r_overlong"] + rewards["r_accuracy"]
        elif self.stage == 3:
            rewards["r_hybrid"] = self.hybrid_reward(response, mode)
            rewards["r_thinking"] = self.thinking_reward(response)
            rewards["r_total"] = sum(rewards.values())

        return rewards

    def batch_compute(
        self,
        responses: List[str],
        labels: List[Union[int, str]],
        modes: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for a batch of responses."""
        if modes is None:
            modes = ["think"] * len(responses)

        batch_rewards = {
            "r_format": self.format_reward.batch_compute(responses),
            "r_overlong": self.overlong_reward.batch_compute(responses),
            "r_accuracy": self.accuracy_reward.batch_compute(responses, labels),
        }

        if self.stage == 1:
            batch_rewards["r_total"] = (
                batch_rewards["r_format"] + batch_rewards["r_overlong"] + batch_rewards["r_accuracy"]
            )
        elif self.stage == 3:
            batch_rewards["r_hybrid"] = self.hybrid_reward.batch_compute(responses, modes)
            batch_rewards["r_thinking"] = self.thinking_reward.batch_compute(responses)
            batch_rewards["r_total"] = sum(batch_rewards.values())

        return batch_rewards


def compute_advantages(rewards: torch.Tensor, group_size: int, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute advantages for DAPO using group-wise normalization.

    For each group of G outputs, normalize rewards to have mean 0.
    """
    batch_size = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(batch_size, group_size)

    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True)

    advantages = (rewards_grouped - mean) / (std + epsilon)
    return advantages.view(-1)


def dynamic_sampling_filter(rewards: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DAPO Dynamic Sampling: Ensure at least one positive and one negative
    advantage per group to prevent degenerate cases.
    """
    batch_size = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(batch_size, group_size)

    has_correct = (rewards_grouped > 0).any(dim=1)
    has_wrong = (rewards_grouped <= 0).any(dim=1)
    valid_mask = has_correct & has_wrong

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    sample_mask = valid_mask.unsqueeze(1).expand(-1, group_size).reshape(-1)

    return sample_mask, valid_indices


if __name__ == "__main__":
    # Test reward functions
    config = RewardConfig()

    test_responses = [
        "<think>Let me analyze this video. I notice artifacts around the face boundary.</think><answer>B)</answer>",
        "<think>The video looks smooth but there might be issues.</think><answer>B)</answer>",
        "This video is fake because I see artifacts.",
        "<think>" + "Analysis " * 200 + "</think><answer>A)</answer>",
    ]
    labels = [1, 0, 1, 1]

    print("Testing reward functions:")
    print("=" * 60)

    combined = CombinedReward(config, stage=1)
    for i, (resp, label) in enumerate(zip(test_responses, labels)):
        rewards = combined.compute(resp, label)
        print(f"Response {i+1}: r_total = {rewards['r_total']:.3f}")

#!/usr/bin/env python3
"""
reward_functions.py
===================
Reward functions for BusterX++ DAPO training on Deepfake-Eval-2024.

Five reward functions from the BusterX++ paper (Section 3.2):
  r_format   – correct <think>/<answer> tag structure
  r_overlong – graduated penalty for responses > L_max tokens
  r_accuracy – binary reward: correct real/fake classification
  r_hybrid   – mode compliance (/think or /no_think) for Stage 3
  r_thinking – quality of reasoning (SophiaVL-R1 or heuristic fallback)

Stage 1 total: r_format + r_overlong + r_accuracy
Stage 3 total: above + r_hybrid + r_thinking
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    # Format reward
    format_reward: float = 1.0
    format_penalty: float = 0.0

    # Overlong shaping
    L_max: int = 600      # tokens before penalty starts
    L_cache: int = 256    # additional tokens for full penalty

    # Accuracy
    accuracy_reward: float = 1.0
    accuracy_penalty: float = 0.0

    # Hybrid mode
    hybrid_think_reward: float = 1.0
    hybrid_nothink_reward: float = 1.0

    # Thinking reward model
    thinking_model_path: str = "SophiaVL/SophiaVL-R1-Thinking-Reward-Model-3B"
    use_thinking_reward: bool = True


# ---------------------------------------------------------------------------
# Individual reward functions
# ---------------------------------------------------------------------------

class FormatReward:
    """r_format: 1.0 if response has <think>…</think><answer>[AB]</answer>, else 0.0"""

    def __init__(self, config: RewardConfig):
        self.config = config
        self.full_pattern = re.compile(
            r'<think>.*?</think>\s*<answer>\s*[AB]\s*\)?</answer>',
            re.DOTALL | re.IGNORECASE,
        )

    def __call__(self, response: str) -> float:
        return self.config.format_reward if self.full_pattern.search(response) else self.config.format_penalty

    def batch_compute(self, responses: List[str]) -> torch.Tensor:
        return torch.tensor([self(r) for r in responses])


class OverlongReward:
    """
    r_overlong: 0.0 when len(response) ≤ L_max tokens.
    Linear penalty reaching -1.0 at L_max + L_cache tokens.
    """

    def __init__(self, config: RewardConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4  # fallback: ~4 chars/token

    def __call__(self, response: str) -> float:
        L = self._count_tokens(response)
        if L <= self.config.L_max:
            return 0.0
        return -min(1.0, (L - self.config.L_max) / self.config.L_cache)

    def batch_compute(self, responses: List[str]) -> torch.Tensor:
        return torch.tensor([self(r) for r in responses])


class AccuracyReward:
    """
    r_accuracy: 1.0 if predicted label matches ground truth, else 0.0.
    A = real (label 0), B = fake (label 1).
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.answer_re = re.compile(r'<answer>\s*([AB])\s*\)?</answer>', re.IGNORECASE)

    def _extract(self, response: str) -> Optional[str]:
        m = self.answer_re.search(response)
        if m:
            return m.group(1).upper()
        # Fallback: bare A) or B) near end
        fb = re.search(r'\b([AB])\)', response[-100:], re.IGNORECASE)
        return fb.group(1).upper() if fb else None

    def __call__(self, response: str, label: Union[int, str]) -> float:
        pred = self._extract(response)
        if pred is None:
            return self.config.accuracy_penalty
        expected = "A" if (label == 0 or str(label).lower() == "real") else "B"
        return self.config.accuracy_reward if pred == expected else self.config.accuracy_penalty

    def batch_compute(self, responses: List[str], labels: List[Union[int, str]]) -> torch.Tensor:
        return torch.tensor([self(r, l) for r, l in zip(responses, labels)])


class HybridReward:
    """
    r_hybrid (Stage 3): 1.0 if response matches the requested think/no_think mode.
    /think   → must contain <think> tags
    /no_think → must NOT contain <think> tags
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.think_re = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

    def __call__(self, response: str, mode: str) -> float:
        has_thinking = bool(self.think_re.search(response))
        if mode == "think":
            return self.config.hybrid_think_reward if has_thinking else 0.0
        elif mode == "no_think":
            return self.config.hybrid_nothink_reward if not has_thinking else 0.0
        return 0.0

    def batch_compute(self, responses: List[str], modes: List[str]) -> torch.Tensor:
        return torch.tensor([self(r, m) for r, m in zip(responses, modes)])


class ThinkingReward:
    """
    r_thinking (Stage 3): quality of reasoning content.
    Uses SophiaVL-R1-Thinking-Reward-Model-3B when available,
    falls back to heuristic scorer otherwise.
    """

    def __init__(self, config: RewardConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self):
        if self._loaded or not self.config.use_thinking_reward:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import logging
            logging.getLogger(__name__).info(
                f"Loading thinking reward model: {self.config.thinking_model_path}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.thinking_model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.thinking_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self._loaded = True
            import logging
            logging.getLogger(__name__).info("Thinking reward model loaded.")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"Could not load thinking reward model ({exc}). Using heuristic fallback."
            )
            self._loaded = False

    def _extract_thinking(self, response: str) -> str:
        m = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _heuristic(self, thinking: str) -> float:
        if not thinking:
            return 0.0
        score = 0.0
        tl = thinking.lower()
        # Length score
        score += 0.2 if len(thinking) >= 100 else 0.0
        score += 0.2 if len(thinking) >= 300 else 0.0
        # Thinking indicators
        indicators = ["let me", "hmm", "wait", "oh", "i see", "interesting",
                      "notice", "observe", "examine", "looking at", "checking", "analyzing"]
        score += min(0.2, sum(1 for p in indicators if p in tl) * 0.05)
        # Artifact / deepfake terms
        artifacts = ["artifact", "boundary", "edge", "temporal", "motion",
                     "lighting", "shadow", "texture", "inconsisten", "blur",
                     "frame", "smooth", "unnatural", "synthetic", "warping", "glitch"]
        score += min(0.2, sum(1 for t in artifacts if t in tl) * 0.04)
        # Self-reflection
        if any(p in tl for p in ["but", "however", "although", "on the other hand", "let me reconsider"]):
            score += 0.2
        return min(1.0, score)

    def __call__(self, response: str) -> float:
        thinking = self._extract_thinking(response)
        if not thinking:
            return 0.0
        if self._loaded and self.model is not None:
            try:
                enc = self.tokenizer(thinking, return_tensors="pt",
                                     truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    out = self.model(**enc)
                    return torch.sigmoid(out.logits[:, -1]).item()
            except Exception:
                pass  # fall through to heuristic
        return self._heuristic(thinking)

    def batch_compute(self, responses: List[str]) -> torch.Tensor:
        return torch.tensor([self(r) for r in responses])


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

class CombinedReward:
    """
    Stage 1: r_total = r_format + r_overlong + r_accuracy        (range: -1 to  2)
    Stage 3: r_total = above + r_hybrid + r_thinking              (range: -1 to  4)
    """

    def __init__(self, config: RewardConfig, tokenizer=None, stage: int = 1,
                 device: str = "cuda"):
        self.config = config
        self.stage = stage

        self.format_reward   = FormatReward(config)
        self.overlong_reward = OverlongReward(config, tokenizer)
        self.accuracy_reward = AccuracyReward(config)
        self.hybrid_reward   = HybridReward(config)
        self.thinking_reward = ThinkingReward(config, device)

        if stage == 3:
            self.thinking_reward.load_model()

    def compute(self, response: str, label: Union[int, str],
                mode: str = "think") -> Dict[str, float]:
        r = {
            "r_format":   self.format_reward(response),
            "r_overlong": self.overlong_reward(response),
            "r_accuracy": self.accuracy_reward(response, label),
        }
        if self.stage == 1:
            r["r_total"] = r["r_format"] + r["r_overlong"] + r["r_accuracy"]
        elif self.stage == 3:
            r["r_hybrid"]   = self.hybrid_reward(response, mode)
            r["r_thinking"] = self.thinking_reward(response)
            r["r_total"]    = sum(r.values())
        return r


# ---------------------------------------------------------------------------
# DAPO utilities
# ---------------------------------------------------------------------------

def compute_advantages(rewards: List[float], epsilon: float = 1e-8) -> List[float]:
    """
    Group-wise advantage normalisation for a single group of G responses.
    advantages[i] = (r_i - mean(r)) / (std(r) + epsilon)
    """
    n = len(rewards)
    mean_r = sum(rewards) / n
    var_r  = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r  = var_r ** 0.5
    return [(r - mean_r) / (std_r + epsilon) for r in rewards]


def dynamic_sampling_filter(rewards: List[float]) -> bool:
    """
    DAPO Dynamic Sampling: require at least one correct (r > 0)
    AND at least one wrong (r <= 0) response in the group.
    Returns True if the group should be used, False if it should be skipped.
    """
    has_positive = any(r > 0 for r in rewards)
    has_nonpositive = any(r <= 0 for r in rewards)
    return has_positive and has_nonpositive

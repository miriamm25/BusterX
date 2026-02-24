#!/usr/bin/env python3
"""
train_dapo.py
=============
BusterX++ DAPO training on Deepfake-Eval-2024 (three-stage pipeline).

  Stage 1  (~70 % of steps, ~12-18 h on H200)
      Foundation RL.  Rewards: r_format + r_overlong + r_accuracy.
      Output → checkpoints/dapo/stage1/

  Stage 2  (~5 % of steps, ~1 h)
      Thinking-mode fusion via SFT (teaches /think vs /no_think).
      Uses label-conditioned templates (same style as train_sft_lora.py).
      Output → checkpoints/dapo/stage2/

  Stage 3  (~25 % of steps, ~6-8 h)
      Advanced RL.  Adds r_hybrid + r_thinking rewards.
      Output → checkpoints/dapo/stage3/  ← FINAL MODEL

Key DAPO innovations preserved from the BusterX++ paper
(Chu et al., 2025):
  1. Clip-Higher   – removes PPO upper clip → more reward signal
  2. Dynamic Sampling – skip groups where all G responses are correct
                        OR all wrong → model only trains on informative clips
  3. Token-Level Loss – skip responses with negative advantage entirely

Usage:
  # smoke test (5 steps, no GPU quota needed)
  python train_dapo.py --stage 1 --max_steps 5 --max_samples 10

  # Stage 1 only (single H200)
  python train_dapo.py --stage 1

  # Full pipeline
  python train_dapo.py --stage all

  # 2× H200 with DeepSpeed ZeRO-2
  deepspeed --num_gpus=2 train_dapo.py --stage 1 \\
      --deepspeed configs/ds_config_zero2.json

  # Start from Stage 3 using an existing Stage 2 checkpoint
  python train_dapo.py --stage 3 \\
      --ref_model checkpoints/dapo/stage2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model

from reward_functions import (
    CombinedReward,
    RewardConfig,
    compute_advantages,
    dynamic_sampling_filter,
)


# ---------------------------------------------------------------------------
# Defaults (server paths)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL   = "l8cv/BusterX_plusplus"
_DEFAULT_TRAIN   = "data/train.jsonl"
_DEFAULT_VAL     = "data/val.jsonl"
_DEFAULT_OUT     = "checkpoints/dapo"

PAD_TOKEN_ID     = 151643   # Qwen2.5 tokenizer pad id


# ---------------------------------------------------------------------------
# Prompts (identical to train_sft_lora.py / eval_on_test.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>"
    "<answer> answer here </answer>"
)

USER_PROMPT_THINK = (
    "Please analyze whether there are any inconsistencies or obvious signs "
    "of forgery in the video, and finally come to a conclusion: Is this "
    "video real or fake?\n\n"
    "Please think about this question as if you were a human pondering "
    "deeply. Engage in an internal dialogue using expressions such as "
    "'let me think', 'wait', 'Hmm', 'oh, I see', 'let\\'s break it down', "
    "etc, or other natural language thought expressions. It\\'s encouraged "
    "to include self-reflection or verification in the reasoning process.\n\n"
    "Then, just answer this MCQ with a single letter:\n"
    "Q: Is this video real or fake?\n"
    "Options:\n"
    "A) real\n"
    "B) fake"
)

USER_PROMPT_NOTHINK = (
    "Analyze this video clip and determine if it is real or fake.\n"
    "Answer directly with a single letter:\n"
    "A) real\n"
    "B) fake"
)

# Stage-2 SFT response templates (same pool as train_sft_lora.py)
_REAL_TEMPLATES = [
    "Examining the video carefully, I notice consistent lighting across frames, natural facial movements, and no visible boundary artifacts around the face region. The temporal coherence between frames looks natural. The motion patterns are consistent with real human movement.",
    "Let me analyze this systematically. The skin texture appears natural with realistic pores and subtle imperfections. Facial expressions transition smoothly. I see no signs of warping, blurring, or inconsistencies that would indicate manipulation. The video appears genuine.",
    "Looking at the visual cues: facial symmetry is normal, eye movements are natural, and there are no telltale compression artifacts in face regions. Background consistency is maintained throughout. This appears to be authentic footage.",
    "Hmm, let me think carefully. The lighting is consistent from frame to frame, shadows fall naturally, and there are no obvious temporal inconsistencies. The person's movements look organic and unforced. No deepfake signatures detected.",
    "Wait, let me check for common forgery indicators: edge artifacts around face — none visible; temporal flickering — not present; unnatural skin smoothing — absent; eye blinking pattern — natural. The video appears to be real.",
]

_FAKE_TEMPLATES = [
    "Examining the video carefully, I notice subtle inconsistencies: there appear to be boundary artifacts around the face region, and the skin texture looks unnaturally smooth in some frames. The facial movements seem slightly off, and I can detect signs of digital manipulation.",
    "Let me analyze this systematically. The lighting on the face appears inconsistent with the background lighting. I notice slight flickering and temporal artifacts between frames. The face boundaries show signs of blending artifacts typical of deepfake generation methods.",
    "Looking at the visual cues: there are subtle warping effects around the face edges, the eye movements appear slightly unnatural, and I can see compression artifacts in the face region that are inconsistent with the rest of the video. This appears to be manipulated.",
    "Hmm, let me think carefully. The skin texture is too smooth and lacks natural pores and micro-expressions. The facial boundary shows subtle blending artifacts, and there are occasional temporal inconsistencies between frames. These are characteristic deepfake signatures.",
    "Wait, let me check for forgery indicators: there are visible edge artifacts around the face; the temporal consistency is poor with flickering in certain regions; the skin tone blending appears artificial. This video shows signs of synthetic generation or face manipulation.",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DAPOConfig:
    # Paths
    model_path:  str = _DEFAULT_MODEL
    output_dir:  str = _DEFAULT_OUT
    train_data:  str = _DEFAULT_TRAIN
    val_data:    str = _DEFAULT_VAL
    ref_model:   Optional[str] = None   # override ref model path for stage 3

    # LoRA
    lora_r:      int = 16
    lora_alpha:  int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Video
    num_frames:  int   = 16
    target_fps:  float = 4.0
    max_pixels:  int   = 589_824   # ≈768². Original paper: 147456

    # DAPO innovations
    group_size:        int   = 4      # G: responses generated per clip
    clip_lower:        float = 0.2    # ε_low for PPO clipping
    clip_higher:       bool  = True   # Clip-Higher: remove upper clip
    dynamic_sampling:  bool  = True   # skip groups with all-same outcome
    token_level_loss:  bool  = True   # skip responses with A_i ≤ 0

    # Training
    batch_size:               int   = 1
    gradient_accumulation_steps: int = 8
    learning_rate:            float = 1e-5
    num_epochs:               int   = 3
    warmup_ratio:             float = 0.1
    max_grad_norm:            float = 1.0
    weight_decay:             float = 0.01

    # Stage ratios (fraction of total dataset-epoch steps)
    stage1_ratio: float = 0.70
    stage2_ratio: float = 0.05
    stage3_ratio: float = 0.25

    # Generation
    max_new_tokens: int   = 750    # Paper targets ≤600 tokens (L_max=600, L_cache=256)
    temperature:    float = 0.7
    top_p:          float = 0.9

    # Evaluation / checkpointing
    eval_steps:    int = 500
    save_steps:    int = 500
    logging_steps: int = 10

    # Hardware
    bf16:                    bool          = True
    gradient_checkpointing:  bool          = True
    deepspeed:               Optional[str] = None

    # Smoke-test overrides
    max_steps:   Optional[int] = None
    max_samples: Optional[int] = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"train_{ts}.log"

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # File (tail -f friendly)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger = logging.getLogger("dapo")
    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str, num_frames: int = 16,
                      target_fps: float = 4.0) -> Optional[List[Image.Image]]:
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = len(vr)
        fps   = vr.get_avg_fps()
        interval = max(1, int(fps / target_fps))
        indices  = [min(i * interval, total - 1) for i in range(num_frames)]
        frames   = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except Exception as exc:
        logging.getLogger("dapo").warning(f"Frame load failed: {video_path}  ({exc})")
        return None


class DAPODataset(Dataset):
    def __init__(self, data_path: str, max_samples: Optional[int] = None,
                 seed: int = 42):
        with open(data_path) as f:
            samples = [json.loads(l) for l in f if l.strip()]
        random.seed(seed)
        random.shuffle(samples)
        if max_samples:
            samples = samples[:max_samples]
        self.samples = samples
        logging.getLogger("dapo").info(
            f"Dataset: {len(self.samples)} samples from {data_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "video_path": s["video_path"],
            "label":      s["label"],
            "source":     s.get("source", ""),
        }


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_base_model(model_path: str, bf16: bool = True,
                    gradient_checkpointing: bool = True,
                    max_pixels: int = 589_824) -> Tuple:
    logger = logging.getLogger("dapo")
    logger.info(f"Loading base model: {model_path}")

    # GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        max_pixels=max_pixels,
    )
    return model, processor


def apply_lora(model, config: DAPOConfig):
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_reference_model(model_path: str) -> Qwen2_5_VLForConditionalGeneration:
    logger = logging.getLogger("dapo")
    logger.info(f"Loading reference model (frozen): {model_path}")
    ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# ---------------------------------------------------------------------------
# Core DAPO helpers
# ---------------------------------------------------------------------------

def _build_messages(frames: List[Image.Image], prompt_text: str) -> List[Dict]:
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


def prepare_input(frames: List[Image.Image], processor, mode: str,
                  device: torch.device) -> Dict:
    prompt = USER_PROMPT_THINK if mode == "think" else USER_PROMPT_NOTHINK
    msgs   = _build_messages(frames, prompt)
    text   = processor.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[text], images=frames,
                       return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def generate_responses(model, inputs: Dict, processor,
                       num_samples: int, config: DAPOConfig,
                       ) -> Tuple[List[str], List[torch.Tensor]]:
    """Return (list_of_text, list_of_response_id_tensors [1,T])."""
    prompt_len  = inputs["input_ids"].shape[1]
    texts, ids  = [], []
    for _ in range(num_samples):
        out = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=PAD_TOKEN_ID,
        )
        resp_ids = out[:, prompt_len:]   # [1, resp_len]
        resp_txt = processor.batch_decode(resp_ids,
                                          skip_special_tokens=True)[0]
        texts.append(resp_txt)
        ids.append(resp_ids)
    return texts, ids


def compute_token_log_probs(model, inputs: Dict, response_ids: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
    """
    Per-token log probs for the response under `model`.
    Returns tensor of shape [resp_len].
    """
    prompt_len = inputs["input_ids"].shape[1]
    resp_len   = response_ids.shape[1]

    full_ids  = torch.cat([inputs["input_ids"], response_ids], dim=1)
    full_mask = torch.cat(
        [inputs["attention_mask"],
         torch.ones(1, resp_len, dtype=inputs["attention_mask"].dtype, device=device)],
        dim=1,
    )
    fwd = {"input_ids": full_ids, "attention_mask": full_mask}
    for key in ("pixel_values", "image_grid_thw"):
        if key in inputs:
            fwd[key] = inputs[key]

    out    = model(**fwd)
    # logits[i] predicts token[i+1], so shift by 1
    logits = out.logits[:, :-1, :]                   # [1, full_len-1, vocab]
    tgts   = full_ids[:, 1:]                          # [1, full_len-1]

    log_p  = F.log_softmax(logits, dim=-1)
    tok_lp = log_p.gather(2, tgts.unsqueeze(-1)).squeeze(-1)  # [1, full_len-1]

    # Response tokens start at position prompt_len-1 in the shifted sequence
    return tok_lp[0, prompt_len - 1: prompt_len - 1 + resp_len]   # [resp_len]


def dapo_loss_fn(log_probs: torch.Tensor, ref_log_probs: torch.Tensor,
                 advantage: float, clip_lower: float,
                 clip_higher: bool) -> torch.Tensor:
    """DAPO surrogate loss for one response (per-token)."""
    ratio = torch.exp(log_probs - ref_log_probs.detach())   # [resp_len]

    if clip_higher:
        # Clip-Higher: only clip from below (allow unbounded positive updates)
        clipped = torch.clamp(ratio, min=1.0 - clip_lower)
    else:
        clipped = torch.clamp(ratio, 1.0 - clip_lower, 1.0 + clip_lower)

    surr1 = ratio   * advantage
    surr2 = clipped * advantage
    loss  = -torch.min(surr1, surr2).mean()
    return loss


# ---------------------------------------------------------------------------
# Stage 2: SFT mode-fusion helpers
# ---------------------------------------------------------------------------

def _generate_sft_response(label: int, mode: str) -> str:
    """Build a synthetic SFT target for stage 2 mode-fusion."""
    if mode == "no_think":
        letter = "A" if label == 0 else "B"
        return f"<answer>{letter})</answer>"
    # think mode
    tmpl   = random.choice(_REAL_TEMPLATES if label == 0 else _FAKE_TEMPLATES)
    letter = "A" if label == 0 else "B"
    return f"<think>{tmpl}</think><answer>{letter})</answer>"


def _sft_loss_for_sample(model, frames, label, mode, processor,
                         device) -> Optional[torch.Tensor]:
    """Compute cross-entropy SFT loss on assistant tokens only."""
    prompt = USER_PROMPT_THINK if mode == "think" else USER_PROMPT_NOTHINK
    response_text = _generate_sft_response(label, mode)

    # Build prompt-only inputs to get prompt length
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            [{"type": "image", "image": f} for f in frames]
            + [{"type": "text", "text": prompt}]},
    ]
    prompt_text = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_enc = processor(text=[prompt_text], images=frames,
                           return_tensors="pt", padding=True)
    prompt_len  = prompt_enc["input_ids"].shape[1]

    # Build full inputs (prompt + assistant response)
    full_msgs = prompt_msgs + [{"role": "assistant", "content": response_text}]
    full_text  = processor.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    full_enc = processor(text=[full_text], images=frames,
                         return_tensors="pt", padding=True)
    full_enc  = {k: v.to(device) for k, v in full_enc.items()}

    labels = full_enc["input_ids"].clone()
    labels[:, :prompt_len] = -100   # mask prompt tokens

    out  = model(**{k: v for k, v in full_enc.items()
                    if k != "labels"}, labels=labels)
    return out.loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DAPOTrainer:

    def __init__(self, config: DAPOConfig, model, processor,
                 reward_calc: CombinedReward):
        self.config        = config
        self.model         = model
        self.processor     = processor
        self.reward_calc   = reward_calc
        self.ref_model     = None
        self.device        = next(
            (p for p in model.parameters() if p.device.type != "cpu"),
            next(model.parameters()),
        ).device
        self.logger        = logging.getLogger("dapo")

    # ------------------------------------------------------------------
    def _make_optimizer_scheduler(self, num_steps: int):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        warmup = int(num_steps * self.config.warmup_ratio)
        sched  = get_linear_schedule_with_warmup(optim, warmup, num_steps)
        return optim, sched

    def _save(self, stage_name: str):
        path = Path(self.config.output_dir) / stage_name / "lora_adapter"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))
        self.logger.info(f"Checkpoint saved → {path}")

    # ------------------------------------------------------------------
    # Stage 1: Foundation RL
    # ------------------------------------------------------------------

    def train_stage1(self, dataloader: DataLoader, num_steps: int):
        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Foundation RL with DAPO")
        self.logger.info(f"  Steps: {num_steps}  |  Group size G={self.config.group_size}")
        self.logger.info("=" * 60)

        self.model.train()
        self.reward_calc.stage = 1

        if self.ref_model is None:
            self.ref_model = load_reference_model(self.config.model_path)

        optim, sched = self._make_optimizer_scheduler(num_steps)
        optim.zero_grad()

        global_step    = 0
        accum_loss     = 0.0
        accum_samples  = 0
        t_start        = time.time()

        bar = tqdm(total=num_steps, desc="Stage 1", dynamic_ncols=True)

        while global_step < num_steps:
            for batch in dataloader:
                if global_step >= num_steps:
                    break

                item       = batch[0]
                video_path = item["video_path"]
                label      = int(item["label"])

                frames = load_video_frames(video_path, self.config.num_frames,
                                           self.config.target_fps)
                if frames is None:
                    continue

                inputs = prepare_input(frames, self.processor, "think", self.device)

                # --- generate G responses ---
                resp_texts, resp_ids = generate_responses(
                    self.model, inputs, self.processor,
                    self.config.group_size, self.config,
                )

                # --- compute rewards ---
                rewards = [
                    self.reward_calc.compute(t, label, "think")["r_total"]
                    for t in resp_texts
                ]

                # --- dynamic sampling filter ---
                if self.config.dynamic_sampling:
                    if not dynamic_sampling_filter(rewards):
                        bar.update(1)
                        global_step += 1
                        continue

                # --- advantages ---
                advantages = compute_advantages(rewards)

                # --- DAPO loss ---
                step_loss   = torch.tensor(0.0, device=self.device)
                n_responses = 0

                for i, (resp_id, adv) in enumerate(zip(resp_ids, advantages)):
                    if self.config.token_level_loss and adv <= 0:
                        continue

                    # log probs under training model (with grad)
                    log_p = compute_token_log_probs(
                        self.model, inputs, resp_id, self.device
                    )
                    # log probs under reference model (no grad)
                    with torch.no_grad():
                        ref_lp = compute_token_log_probs(
                            self.ref_model, inputs, resp_id, self.device
                        )

                    loss = dapo_loss_fn(log_p, ref_lp, adv,
                                        self.config.clip_lower,
                                        self.config.clip_higher)
                    step_loss   = step_loss + loss
                    n_responses += 1

                if n_responses > 0:
                    step_loss = step_loss / n_responses
                    (step_loss / self.config.gradient_accumulation_steps).backward()
                    accum_loss    += step_loss.item()
                    accum_samples += 1

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optim.step()
                    sched.step()
                    optim.zero_grad()

                global_step += 1
                bar.update(1)

                if global_step % self.config.logging_steps == 0 and accum_samples > 0:
                    avg  = accum_loss / accum_samples
                    elapsed = time.time() - t_start
                    self.logger.info(
                        f"[Stage1] step={global_step}/{num_steps}  "
                        f"loss={avg:.4f}  elapsed={elapsed/60:.1f}min"
                    )
                    accum_loss    = 0.0
                    accum_samples = 0

                if global_step % self.config.save_steps == 0:
                    self._save(f"stage1/step{global_step}")

        bar.close()
        self._save("stage1")
        self.logger.info("Stage 1 complete.")

    # ------------------------------------------------------------------
    # Stage 2: Thinking-mode fusion (SFT)
    # ------------------------------------------------------------------

    def train_stage2(self, dataloader: DataLoader, num_steps: int):
        self.logger.info("=" * 60)
        self.logger.info("Stage 2: Thinking-mode fusion (SFT)")
        self.logger.info(f"  Steps: {num_steps}  |  alternating think / no_think")
        self.logger.info("=" * 60)

        self.model.train()
        optim, sched = self._make_optimizer_scheduler(num_steps)
        optim.zero_grad()

        global_step = 0
        accum_loss  = 0.0
        accum_n     = 0

        bar = tqdm(total=num_steps, desc="Stage 2", dynamic_ncols=True)

        while global_step < num_steps:
            for batch in dataloader:
                if global_step >= num_steps:
                    break

                item       = batch[0]
                video_path = item["video_path"]
                label      = int(item["label"])

                frames = load_video_frames(video_path, self.config.num_frames,
                                           self.config.target_fps)
                if frames is None:
                    continue

                mode = "think" if random.random() > 0.5 else "no_think"
                loss = _sft_loss_for_sample(
                    self.model, frames, label, mode,
                    self.processor, self.device,
                )
                if loss is None:
                    continue

                (loss / self.config.gradient_accumulation_steps).backward()
                accum_loss += loss.item()
                accum_n    += 1

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optim.step()
                    sched.step()
                    optim.zero_grad()

                global_step += 1
                bar.update(1)

                if global_step % self.config.logging_steps == 0 and accum_n > 0:
                    self.logger.info(
                        f"[Stage2] step={global_step}/{num_steps}  "
                        f"loss={accum_loss/accum_n:.4f}"
                    )
                    accum_loss = 0.0
                    accum_n    = 0

                if global_step % self.config.save_steps == 0:
                    self._save(f"stage2/step{global_step}")

        bar.close()
        self._save("stage2")
        self.logger.info("Stage 2 complete.")

    # ------------------------------------------------------------------
    # Stage 3: Advanced RL (all rewards + thinking reward)
    # ------------------------------------------------------------------

    def train_stage3(self, dataloader: DataLoader, num_steps: int):
        self.logger.info("=" * 60)
        self.logger.info("Stage 3: Advanced RL with Thinking Reward")
        self.logger.info(f"  Steps: {num_steps}  |  G={self.config.group_size}")
        self.logger.info("=" * 60)

        self.model.train()
        self.reward_calc.stage = 3
        self.reward_calc.thinking_reward.load_model()   # try SophiaVL, fall back to heuristic

        # Update reference model to Stage 2 output (or Stage 1 if Stage 2 skipped)
        stage2_adapter = Path(self.config.output_dir) / "stage2" / "lora_adapter"
        stage1_adapter = Path(self.config.output_dir) / "stage1" / "lora_adapter"
        if stage2_adapter.exists():
            ref_path = str(stage2_adapter)
        elif stage1_adapter.exists():
            ref_path = str(stage1_adapter)
        else:
            ref_path = self.config.ref_model or self.config.model_path

        self.logger.info(f"Stage 3 reference model: {ref_path}")
        if self.ref_model is not None:
            del self.ref_model
            torch.cuda.empty_cache()
        self.ref_model = load_reference_model(ref_path)

        optim, sched = self._make_optimizer_scheduler(num_steps)
        optim.zero_grad()

        global_step    = 0
        accum_loss     = 0.0
        accum_samples  = 0
        t_start        = time.time()

        bar = tqdm(total=num_steps, desc="Stage 3", dynamic_ncols=True)

        while global_step < num_steps:
            for batch in dataloader:
                if global_step >= num_steps:
                    break

                item       = batch[0]
                video_path = item["video_path"]
                label      = int(item["label"])

                frames = load_video_frames(video_path, self.config.num_frames,
                                           self.config.target_fps)
                if frames is None:
                    continue

                mode   = "think" if random.random() > 0.5 else "no_think"
                inputs = prepare_input(frames, self.processor, mode, self.device)

                resp_texts, resp_ids = generate_responses(
                    self.model, inputs, self.processor,
                    self.config.group_size, self.config,
                )

                rewards = [
                    self.reward_calc.compute(t, label, mode)["r_total"]
                    for t in resp_texts
                ]

                if self.config.dynamic_sampling:
                    if not dynamic_sampling_filter(rewards):
                        bar.update(1)
                        global_step += 1
                        continue

                advantages = compute_advantages(rewards)

                step_loss   = torch.tensor(0.0, device=self.device)
                n_responses = 0

                for resp_id, adv in zip(resp_ids, advantages):
                    if self.config.token_level_loss and adv <= 0:
                        continue

                    log_p = compute_token_log_probs(
                        self.model, inputs, resp_id, self.device
                    )
                    with torch.no_grad():
                        ref_lp = compute_token_log_probs(
                            self.ref_model, inputs, resp_id, self.device
                        )

                    loss = dapo_loss_fn(log_p, ref_lp, adv,
                                        self.config.clip_lower,
                                        self.config.clip_higher)
                    step_loss   = step_loss + loss
                    n_responses += 1

                if n_responses > 0:
                    step_loss = step_loss / n_responses
                    (step_loss / self.config.gradient_accumulation_steps).backward()
                    accum_loss    += step_loss.item()
                    accum_samples += 1

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optim.step()
                    sched.step()
                    optim.zero_grad()

                global_step += 1
                bar.update(1)

                if global_step % self.config.logging_steps == 0 and accum_samples > 0:
                    avg     = accum_loss / accum_samples
                    elapsed = time.time() - t_start
                    self.logger.info(
                        f"[Stage3] step={global_step}/{num_steps}  "
                        f"loss={avg:.4f}  elapsed={elapsed/60:.1f}min"
                    )
                    accum_loss    = 0.0
                    accum_samples = 0

                if global_step % self.config.save_steps == 0:
                    self._save(f"stage3/step{global_step}")

        bar.close()
        self._save("stage3")
        self.logger.info("Stage 3 complete.")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def train(self, train_dataset: DAPODataset, stages: List[int]):
        dl = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=lambda x: x,
        )

        total = len(dl) * self.config.num_epochs
        if self.config.max_steps:
            total = min(total, self.config.max_steps)

        s1 = int(total * self.config.stage1_ratio)
        s2 = int(total * self.config.stage2_ratio)
        s3 = total - s1 - s2

        self.logger.info(f"Total steps: {total}  (S1={s1}, S2={s2}, S3={s3})")

        if 1 in stages:
            self.train_stage1(dl, s1)
        if 2 in stages:
            self.train_stage2(dl, s2)
        if 3 in stages:
            self.train_stage3(dl, s3)

        self.logger.info("All requested stages complete.")
        self.logger.info(f"Checkpoints in: {self.config.output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BusterX++ DAPO training")
    parser.add_argument("--stage",        default="all",
                        choices=["1", "2", "3", "all"],
                        help="Stage(s) to run: 1, 2, 3, or all")
    parser.add_argument("--model_path",   default=_DEFAULT_MODEL)
    parser.add_argument("--output_dir",   default=_DEFAULT_OUT)
    parser.add_argument("--train_data",   default=_DEFAULT_TRAIN)
    parser.add_argument("--val_data",     default=_DEFAULT_VAL)
    parser.add_argument("--ref_model",    default=None,
                        help="Path to reference model (overrides auto-detection)")
    parser.add_argument("--num_epochs",   type=int,   default=3)
    parser.add_argument("--learning_rate",type=float, default=1e-5)
    parser.add_argument("--max_new_tokens", type=int, default=750)
    parser.add_argument("--max_pixels",   type=int,   default=589_824)
    parser.add_argument("--eval_steps",   type=int,   default=500)
    parser.add_argument("--save_steps",   type=int,   default=500)
    parser.add_argument("--max_steps",    type=int,   default=None,
                        help="Hard cap on total steps (smoke test)")
    parser.add_argument("--max_samples",  type=int,   default=None,
                        help="Limit dataset to first N samples")
    parser.add_argument("--deepspeed",    default=None)
    args = parser.parse_args()

    # Reproducibility
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Logging
    log_dir = str(Path(args.output_dir) / "logs")
    logger  = setup_logging(log_dir)
    logger.info(f"Stage(s): {args.stage}  |  model: {args.model_path}")

    # Config
    cfg             = DAPOConfig()
    cfg.model_path  = args.model_path
    cfg.output_dir  = args.output_dir
    cfg.train_data  = args.train_data
    cfg.val_data    = args.val_data
    cfg.ref_model   = args.ref_model
    cfg.num_epochs  = args.num_epochs
    cfg.learning_rate = args.learning_rate
    cfg.max_new_tokens = args.max_new_tokens
    cfg.max_pixels  = args.max_pixels
    cfg.eval_steps  = args.eval_steps
    cfg.save_steps  = args.save_steps
    cfg.max_steps   = args.max_steps

    # Model + LoRA
    model, processor = load_base_model(
        cfg.model_path, cfg.bf16, cfg.gradient_checkpointing, cfg.max_pixels
    )
    model = apply_lora(model, cfg)
    model.enable_input_require_grads()   # required for gradient checkpointing + LoRA

    # Reward calculator (stage=1 initially; Stage 3 changes it to 3 internally)
    reward_cfg  = RewardConfig()
    reward_calc = CombinedReward(
        reward_cfg, processor.tokenizer, stage=1, device="cuda"
    )

    # Trainer
    trainer = DAPOTrainer(cfg, model, processor, reward_calc)

    # Dataset
    dataset = DAPODataset(cfg.train_data, max_samples=args.max_samples)

    # Stages
    if args.stage == "all":
        stages = [1, 2, 3]
    else:
        stages = [int(args.stage)]

    trainer.train(dataset, stages)

    logger.info("Done.")


if __name__ == "__main__":
    main()

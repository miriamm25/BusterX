#!/usr/bin/env python3
"""
train_sft_lora.py
=================
LoRA Supervised Fine-Tuning of BusterX++ (Qwen2.5-VL-7B) on
Deepfake-Eval-2024 clips.

Key design choices:
  - Frames passed as 16 individual {"type":"image",...} dicts (not "video").
    This matches the chat_template.jinja which wraps each image in
    <|vision_start|><|image_pad|><|vision_end|>.
  - Assistant-only loss masking: system + user tokens are set to -100.
    Achieved by a double processor call (prompt-only → full) to get the
    exact prompt_len including image patch tokens.
  - No bitsandbytes quantization. bf16 throughout.
  - batch_size=1 per device; gradient_accumulation=8 (effective batch 8).
  - Seed=42 everywhere.
  - AccuracyCallback: runs generate() on 100 val clips every eval_steps and
    logs label accuracy to TensorBoard alongside loss.

Usage:
  # Single GPU (H200 – recommended)
  python train_sft_lora.py

  # Override any parameter
  python train_sft_lora.py --train_data data/train.jsonl \\
      --val_data data/val.jsonl --num_epochs 3 --eval_steps 200

  # Multi-GPU with DeepSpeed ZeRO-2
  deepspeed --num_gpus=2 train_sft_lora.py \\
      --deepspeed configs/ds_config_zero2.json

  # Smoke test (10 steps, 20 samples)
  python train_sft_lora.py --max_steps 10 --max_samples 20 \\
      --output_dir checkpoints/smoke
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# BusterX++ canonical prompts (must not be changed)
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

USER_PROMPT = (
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

# Label-conditioned reasoning templates (5 per class for variance).
# Style matches the BusterX++ paper and existing train_sft.py templates.
REAL_TEMPLATES = [
    "Let me carefully examine this video. Looking at the facial features, "
    "I notice consistent lighting and natural skin texture. The motion "
    "appears smooth and physically plausible. The edges around the face "
    "blend naturally with the background. Hmm, I don't see any obvious "
    "signs of manipulation. The temporal consistency across frames looks "
    "good – no flickering or unnatural transitions. The video appears to "
    "be authentic.",

    "Let me analyze this video frame by frame. The lighting on the face "
    "is consistent with the environment. I notice natural motion blur "
    "during movement. The facial boundaries integrate smoothly with the "
    "surrounding area. Wait, checking for artifacts... I don't detect any "
    "telltale signs of AI generation or face manipulation. The overall "
    "quality and consistency suggest this is a real video.",

    "Examining this video carefully. The subject's movements appear natural "
    "and physically coherent. Let me check the facial region – the texture "
    "looks organic, not synthetic. The lighting responds naturally to "
    "movement. I don't observe any boundary artifacts or temporal "
    "inconsistencies. This appears to be genuine footage.",

    "Oh, I see. Let me break this down step by step. First, temporal "
    "consistency: the face and background move in a coherent, plausible "
    "way across all sampled frames. Second, skin texture: I can see pore "
    "detail and natural imperfections – not the over-smooth appearance "
    "typical of synthetic generation. Third, edge regions: no halo or "
    "blending artifacts. Everything points to an authentic, unmodified "
    "video.",

    "Hmm, let me think through this. The inter-frame transitions are "
    "smooth and natural. The gaze direction and micro-expressions look "
    "human. I don't detect frequency anomalies in the background, and "
    "the compression artifacts are uniform rather than region-specific. "
    "Self-check: none of the hallmarks of deepfake generation are "
    "present. My conclusion is that this video is real.",
]

FAKE_TEMPLATES = [
    "Let me examine this video closely. Hmm, I notice some inconsistencies "
    "around the facial boundaries – there seems to be a subtle blending "
    "artifact. The lighting on the face doesn't quite match the "
    "environmental lighting. Wait, looking at the temporal consistency... "
    "there's slight flickering in the facial region between frames. The "
    "motion also seems slightly unnatural. These are signs of potential "
    "face manipulation or synthetic generation.",

    "Analyzing this video carefully. Oh, I see some issues – the face "
    "texture appears too smooth in places, lacking natural skin detail. "
    "Let me check the boundaries... yes, there are visible edge artifacts "
    "where the face meets the background. The temporal dynamics also show "
    "some inconsistency. These artifacts suggest this video has been "
    "manipulated.",

    "Let me think about this video. Looking at frame-to-frame consistency, "
    "I notice subtle temporal artifacts. The facial region shows signs of "
    "synthetic generation – the texture and lighting don't perfectly match "
    "the scene. Hmm, the motion dynamics seem slightly off too. The "
    "combination of these factors indicates this is likely a deepfake or "
    "AI-generated content.",

    "Breaking this down: First, I observe a slight halo around the "
    "face boundary that is characteristic of GAN-based or diffusion-based "
    "face synthesis. Second, the lighting on the face has a slightly "
    "different colour temperature than the ambient scene. Third, there are "
    "micro-jitter artefacts in the facial region between certain frames. "
    "Wait – the background is completely stable while the face has these "
    "inconsistencies. That is a strong deepfake indicator.",

    "Hmm, something is off here. Let me look at the skin texture – it "
    "has an unusually smooth, airbrushed quality that is inconsistent with "
    "the rest of the scene. Checking the temporal axis: I can see subtle "
    "flickering around the mouth and eye regions across frames, which is "
    "typical of imperfect temporal coherence in face-swap models. "
    "Additionally, the edge between face and hair shows blending seams. "
    "These converging indicators strongly suggest AI-generated or "
    "manipulated content.",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # Model
    model_path: str = "l8cv/BusterX_plusplus"
    output_dir: str = "./checkpoints/sft"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Data
    train_data: str = "./data/train.jsonl"
    val_data: str = "./data/val.jsonl"
    max_samples: Optional[int] = None   # set for quick smoke-test

    # Video
    num_frames: int = 16
    target_fps: float = 4.0
    max_pixels: int = 589_824           # ≈768² – tune up on H200 if desired

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_steps: int = -1                 # -1 = use num_epochs

    # Eval / checkpoint
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10
    accuracy_eval_samples: int = 100    # val clips for AccuracyCallback

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = None


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    target_fps: float = 4.0,
) -> Optional[List[Image.Image]]:
    """Sample num_frames from video using decord. Returns None on error."""
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        frame_interval = max(1, int(fps / target_fps))
        frame_indices = [
            min(i * frame_interval, total_frames - 1)
            for i in range(num_frames)
        ]

        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception as exc:
        print(f"  [WARN] failed to load {video_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """
    Loads JSONL manifest, samples frames, builds tokenised training examples.

    Loss masking strategy
    ---------------------
    1. Build prompt-only messages (no assistant turn) and call the processor
       with add_generation_prompt=True → gives prompt_len (includes image
       patch tokens inserted by the vision processor).
    2. Build full messages (including assistant turn) and call the processor
       again → gives full input_ids.
    3. Clone input_ids as labels; set labels[:, :prompt_len] = -100.
       → loss is computed only on the assistant response tokens.
    """

    def __init__(
        self,
        data_path: str,
        processor,
        num_frames: int = 16,
        target_fps: float = 4.0,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.processor = processor
        self.num_frames = num_frames
        self.target_fps = target_fps
        self._rng = random.Random(seed)   # local RNG for template selection

        with open(data_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        # Deterministic shuffle so epochs are consistent across restarts
        rng_shuffle = random.Random(seed)
        rng_shuffle.shuffle(samples)

        if max_samples:
            samples = samples[:max_samples]

        self.samples = samples
        print(f"  Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def _generate_response(self, label: int) -> str:
        if label == 0:
            reasoning = self._rng.choice(REAL_TEMPLATES)
            letter = "A"
        else:
            reasoning = self._rng.choice(FAKE_TEMPLATES)
            letter = "B"
        return f"<think>{reasoning}</think><answer>{letter})</answer>"

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]

        frames = load_video_frames(video_path, self.num_frames, self.target_fps)
        if frames is None:
            return None

        # --- Build content list (16 image dicts + text prompt) ---
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": USER_PROMPT})

        # --- Prompt-only messages (for computing prompt_len) ---
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=frames,
            return_tensors="pt", padding=False,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # --- Full messages (prompt + assistant response) ---
        full_messages = prompt_messages + [
            {"role": "assistant", "content": self._generate_response(label)}
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_inputs = self.processor(
            text=[full_text], images=frames,
            return_tensors="pt", padding=False,
        )

        # Squeeze batch dim
        input_ids      = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        pixel_values   = full_inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)

        # Assistant-only loss masking
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        if pixel_values is not None:
            item["pixel_values"] = pixel_values

        # Pass through any extra vision-related keys (image_grid_thw, etc.)
        for key in full_inputs:
            if key not in item:
                val = full_inputs[key]
                if val is not None:
                    item[key] = val.squeeze(0) if val.dim() > 1 else val

        return item


# ---------------------------------------------------------------------------
# Collate (batch_size=1 → padding never triggered, but implemented correctly)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    if len(batch) == 1:
        return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in batch[0].items()}

    # batch_size > 1: pad text tensors; stack vision tensors if same shape
    pad_token_id = 151643  # <|endoftext|> per tokenizer_config.json

    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        out["input_ids"].append(torch.cat([
            b["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=b["input_ids"].dtype),
        ]))
        out["attention_mask"].append(torch.cat([
            b["attention_mask"],
            torch.zeros(pad_len, dtype=b["attention_mask"].dtype),
        ]))
        out["labels"].append(torch.cat([
            b["labels"],
            torch.full((pad_len,), -100, dtype=b["labels"].dtype),
        ]))

    out["input_ids"]      = torch.stack(out["input_ids"])
    out["attention_mask"] = torch.stack(out["attention_mask"])
    out["labels"]         = torch.stack(out["labels"])

    # Vision tensors: stack only if all items have the same shape
    for key in batch[0]:
        if key in out:
            continue
        vals = [b[key] for b in batch if key in b]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            try:
                out[key] = torch.stack(vals)
            except RuntimeError:
                # Variable-shape pixel_values: concatenate along batch dim
                out[key] = torch.cat(vals, dim=0)
        else:
            out[key] = vals

    return out


# ---------------------------------------------------------------------------
# Accuracy callback (runs generate() on a small val subset each eval)
# ---------------------------------------------------------------------------

import re

def _extract_label(text: str) -> Optional[str]:
    m = re.search(r"<answer>\s*([AB])\s*\)?</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: last A) or B) in final 80 chars
    tail = text[-80:]
    m2 = re.search(r"\b([AB])\)", tail, re.IGNORECASE)
    return m2.group(1).upper() if m2 else None


class AccuracyCallback(TrainerCallback):
    """
    Runs model.generate() on a small val sample every eval_steps and logs
    label accuracy to TensorBoard.  Separate from loss-based eval so it
    doesn't interfere with Trainer's own eval loop.
    """

    def __init__(self, val_dataset: DeepfakeDataset, processor, config: SFTConfig):
        self.val_dataset = val_dataset
        self.processor = processor
        self.config = config

        n = min(config.accuracy_eval_samples, len(val_dataset))
        rng = random.Random(42)
        self._indices = rng.sample(range(len(val_dataset)), n)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for idx in tqdm(self._indices, desc="AccuracyEval", leave=False):
                sample = self.val_dataset.samples[idx]
                video_path = sample["video_path"]
                gt_label = sample["label"]

                frames = load_video_frames(
                    video_path,
                    self.config.num_frames,
                    self.config.target_fps,
                )
                if frames is None:
                    continue

                content = [{"type": "image", "image": f} for f in frames]
                content.append({"type": "text", "text": USER_PROMPT})
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": content},
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text], images=frames,
                    return_tensors="pt", padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                torch.cuda.empty_cache()
                output_ids = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                )
                response = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]

                pred_letter = _extract_label(response)
                if pred_letter is None:
                    continue

                pred_label = 0 if pred_letter == "A" else 1
                correct += int(pred_label == gt_label)
                total += 1

        if total > 0:
            acc = correct / total
            print(f"\n  [AccuracyCallback] step={state.global_step} "
                  f"label_accuracy={acc:.4f} ({correct}/{total})")
            # Log to TensorBoard if writer is available
            if state.is_world_process_zero and hasattr(state, "log_history"):
                state.log_history.append({
                    "step": state.global_step,
                    "eval/label_accuracy": acc,
                })

        model.train()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_dir) / f"train_{ts}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    logging.info(f"Log file: {log_path}")


def setup_model_and_lora(config: SFTConfig):
    print(f"Loading model: {config.model_path}")

    # GPU throughput optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        max_pixels=config.max_pixels,
    )

    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config: SFTConfig):
    print("=" * 60)
    print("BusterX++ LoRA SFT  –  Deepfake-Eval-2024")
    print("=" * 60)

    # Logging (file + console, tail -f friendly)
    setup_logging(config.output_dir)

    # Global seed
    set_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model, processor = setup_model_and_lora(config)

    print("\nLoading datasets…")
    train_dataset = DeepfakeDataset(
        config.train_data, processor,
        num_frames=config.num_frames, target_fps=config.target_fps,
        max_samples=config.max_samples,
    )

    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = DeepfakeDataset(
            config.val_data, processor,
            num_frames=config.num_frames, target_fps=config.target_fps,
            max_samples=config.max_samples,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=False,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        deepspeed=config.deepspeed,
        seed=42,
    )

    callbacks = []
    if val_dataset:
        callbacks.append(
            AccuracyCallback(val_dataset, processor, config)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    print("\nStarting training…")
    trainer.train()

    # Save LoRA adapter (small – ~50-100 MB)
    lora_path = os.path.join(config.output_dir, "lora_adapter")
    model.save_pretrained(lora_path)
    processor.save_pretrained(lora_path)
    print(f"\nLoRA adapter saved → {lora_path}")

    # Save full merged checkpoint (optional – large, ~15 GB)
    merged_path = os.path.join(config.output_dir, "merged")
    print(f"Merging LoRA into base weights → {merged_path}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    print(f"Merged model saved → {merged_path}")

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  LoRA adapter : {lora_path}")
    print(f"  Merged model : {merged_path}")
    print(f"  TensorBoard  : tensorboard --logdir {config.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BusterX++ LoRA SFT")

    parser.add_argument("--model_path",   default="l8cv/BusterX_plusplus")
    parser.add_argument("--output_dir",   default="./checkpoints/sft")
    parser.add_argument("--train_data",   default="./data/train.jsonl")
    parser.add_argument("--val_data",     default="./data/val.jsonl")
    parser.add_argument("--max_samples",  type=int,   default=None)
    parser.add_argument("--max_steps",    type=int,   default=-1)
    parser.add_argument("--num_epochs",   type=int,   default=3)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--max_pixels",   type=int,   default=589_824)
    parser.add_argument("--learning_rate",type=float, default=2e-5)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--grad_accum",   type=int,   default=8)
    parser.add_argument("--eval_steps",   type=int,   default=200)
    parser.add_argument("--save_steps",   type=int,   default=200)
    parser.add_argument("--deepspeed",    type=str,   default=None)

    args = parser.parse_args()

    config = SFTConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        train_data=args.train_data,
        val_data=args.val_data,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_pixels=args.max_pixels,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        deepspeed=args.deepspeed,
    )

    train(config)


if __name__ == "__main__":
    main()

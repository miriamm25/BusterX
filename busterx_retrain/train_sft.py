#!/usr/bin/env python3
"""
BusterX++ SFT Training Script
=============================
Supervised Fine-Tuning with LoRA for TikTok deepfake detection.

This implements Step 1 of the retraining plan:
- Train on mixed dataset with LoRA
- Validate pipeline before DAPO training
- Target: Improve from 43.8% baseline on TikTok videos

Usage:
    # Single GPU
    python train_sft.py --config configs/sft_config.yaml

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=4 train_sft.py --config configs/sft_config.yaml --deepspeed

    # Quick test run
    python train_sft.py --config configs/sft_config.yaml --max_steps 100 --eval_steps 50
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_path: str = "l8cv/BusterX_plusplus"
    output_dir: str = "./checkpoints/sft"

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Data
    train_data: str = "./training_data/train.jsonl"
    val_data: str = "./training_data/val.jsonl"
    max_samples: Optional[int] = None

    # Video processing
    num_frames: int = 16
    target_fps: float = 4.0
    max_pixels: int = 147456  # 384x384

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10

    # Hardware
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # DeepSpeed
    deepspeed: Optional[str] = None


# System prompt from BusterX++ paper
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

# User prompt template
USER_PROMPT = """Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""


def load_video_frames(video_path: str, num_frames: int = 16, target_fps: float = 4.0) -> List[Image.Image]:
    """Load frames from video using decord."""
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        frame_interval = max(1, int(fps / target_fps))
        frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection training."""

    def __init__(
        self,
        data_path: str,
        processor,
        num_frames: int = 16,
        target_fps: float = 4.0,
        max_samples: Optional[int] = None,
        include_reasoning: bool = False,
    ):
        self.processor = processor
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.include_reasoning = include_reasoning

        # Load data manifest
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def generate_response(self, label: int) -> str:
        """Generate expected response for training."""
        if label == 0:  # Real
            answer = "A"
            reasoning = self._generate_real_reasoning()
        else:  # Fake
            answer = "B"
            reasoning = self._generate_fake_reasoning()

        return f"<think>{reasoning}</think><answer>{answer})</answer>"

    def _generate_real_reasoning(self) -> str:
        """Generate reasoning for real videos."""
        templates = [
            "Let me carefully examine this video. Looking at the facial features, I notice consistent lighting and natural skin texture. The motion appears smooth and physically plausible. The edges around the face blend naturally with the background. Hmm, I don't see any obvious signs of manipulation. The temporal consistency across frames looks good - no flickering or unnatural transitions. The video appears to be authentic.",
            "Let me analyze this video frame by frame. The lighting on the face is consistent with the environment. I notice natural motion blur during movement. The facial boundaries integrate smoothly with the surrounding area. Wait, checking for artifacts... I don't detect any telltale signs of AI generation or face manipulation. The overall quality and consistency suggest this is a real video.",
            "Examining this video carefully. The subject's movements appear natural and physically coherent. Let me check the facial region - the texture looks organic, not synthetic. The lighting responds naturally to movement. I don't observe any boundary artifacts or temporal inconsistencies. This appears to be genuine footage.",
        ]
        return random.choice(templates)

    def _generate_fake_reasoning(self) -> str:
        """Generate reasoning for fake videos."""
        templates = [
            "Let me examine this video closely. Hmm, I notice some inconsistencies around the facial boundaries - there seems to be a subtle blending artifact. The lighting on the face doesn't quite match the environmental lighting. Wait, looking at the temporal consistency... there's slight flickering in the facial region between frames. The motion also seems slightly unnatural. These are signs of potential face manipulation or synthetic generation.",
            "Analyzing this video carefully. Oh, I see some issues - the face texture appears too smooth in places, lacking natural skin detail. Let me check the boundaries... yes, there are visible edge artifacts where the face meets the background. The temporal dynamics also show some inconsistency. These artifacts suggest this video has been manipulated.",
            "Let me think about this video. Looking at frame-to-frame consistency, I notice subtle temporal artifacts. The facial region shows signs of synthetic generation - the texture and lighting don't perfectly match the scene. Hmm, the motion dynamics seem slightly off too. The combination of these factors indicates this is likely a deepfake or AI-generated content.",
        ]
        return random.choice(templates)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]

        # Load video frames
        frames = load_video_frames(video_path, self.num_frames, self.target_fps)

        if frames is None:
            # Return a dummy sample on error (will be filtered)
            return None

        # Build conversation
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": USER_PROMPT})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": self.generate_response(label)}
        ]

        # Process with tokenizer
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(
            text=[text],
            images=frames,
            return_tensors="pt",
            padding=True
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


def collate_fn(batch):
    """Custom collate function that filters None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Find max lengths
    max_len = max(b["input_ids"].shape[0] for b in batch)

    # Pad sequences
    padded_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    # Handle pixel values if present
    has_pixel_values = "pixel_values" in batch[0]
    if has_pixel_values:
        padded_batch["pixel_values"] = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids
        padded_ids = torch.cat([
            b["input_ids"],
            torch.full((pad_len,), 0, dtype=b["input_ids"].dtype)
        ])
        padded_batch["input_ids"].append(padded_ids)

        # Pad attention_mask
        padded_mask = torch.cat([
            b["attention_mask"],
            torch.zeros(pad_len, dtype=b["attention_mask"].dtype)
        ])
        padded_batch["attention_mask"].append(padded_mask)

        # Pad labels (use -100 for padding)
        padded_labels = torch.cat([
            b["labels"],
            torch.full((pad_len,), -100, dtype=b["labels"].dtype)
        ])
        padded_batch["labels"].append(padded_labels)

        if has_pixel_values:
            padded_batch["pixel_values"].append(b["pixel_values"])

    # Stack tensors
    padded_batch["input_ids"] = torch.stack(padded_batch["input_ids"])
    padded_batch["attention_mask"] = torch.stack(padded_batch["attention_mask"])
    padded_batch["labels"] = torch.stack(padded_batch["labels"])

    if has_pixel_values:
        padded_batch["pixel_values"] = torch.stack(padded_batch["pixel_values"])

    return padded_batch


class MetricsCallback(TrainerCallback):
    """Callback for logging custom metrics."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\nEvaluation metrics at step {state.global_step}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")


def setup_model_and_lora(config: SFTConfig):
    """Load model and apply LoRA."""
    print(f"Loading model from {config.model_path}...")

    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        max_pixels=config.max_pixels
    )

    return model, processor


def train(config: SFTConfig):
    """Main training function."""
    print("=" * 60)
    print("BusterX++ SFT Training")
    print("=" * 60)

    # Setup model
    model, processor = setup_model_and_lora(config)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DeepfakeDataset(
        config.train_data,
        processor,
        num_frames=config.num_frames,
        target_fps=config.target_fps,
        max_samples=config.max_samples,
    )

    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = DeepfakeDataset(
            config.val_data,
            processor,
            num_frames=config.num_frames,
            target_fps=config.target_fps,
            max_samples=config.max_samples // 5 if config.max_samples else None,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        deepspeed=config.deepspeed,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[MetricsCallback()],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final"))

    # Save LoRA adapter separately
    model.save_pretrained(os.path.join(config.output_dir, "lora_adapter"))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BusterX++ SFT Training")

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Model
    parser.add_argument("--model_path", type=str, default="l8cv/BusterX_plusplus")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--train_data", type=str, default="./training_data/train.jsonl")
    parser.add_argument("--val_data", type=str, default="./training_data/val.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)

    # Evaluation
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)

    # Hardware
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    # Load config from YAML if provided
    config = SFTConfig()
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Override with CLI args
    for k, v in vars(args).items():
        if v is not None and hasattr(config, k):
            setattr(config, k, v)

    train(config)


if __name__ == "__main__":
    main()

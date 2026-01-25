#!/usr/bin/env python3
"""
BusterX++ Evaluation Script
===========================
Evaluates trained models on test sets and TikTok videos.

Metrics:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Per-category breakdown
- Error analysis

Usage:
    # Evaluate on test set
    python evaluate.py --model ./checkpoints/sft/final --data ./training_data/test.jsonl

    # Evaluate on TikTok videos
    python evaluate.py --model ./checkpoints/sft/final --tiktok ../deepfakes/

    # Compare baseline vs trained
    python evaluate.py --model l8cv/BusterX_plusplus --data ./training_data/test.jsonl --name baseline
    python evaluate.py --model ./checkpoints/sft/final --data ./training_data/test.jsonl --name trained
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model_path: str = "l8cv/BusterX_plusplus"
    lora_path: Optional[str] = None
    num_frames: int = 16
    target_fps: float = 4.0
    max_new_tokens: int = 1500
    batch_size: int = 1


# Prompts
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

USER_PROMPT = """Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""


def load_video_frames(video_path: str, num_frames: int = 16, target_fps: float = 4.0) -> Optional[List[Image.Image]]:
    """Load frames from video."""
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


def parse_response(response: str) -> Dict:
    """Parse model response to extract verdict and reasoning."""
    response_lower = response.lower()

    # Extract thinking
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    thinking = think_match.group(1).strip() if think_match else ""

    # Extract answer
    answer_match = re.search(r'<answer>\s*([AB])\s*\)?</answer>', response, re.IGNORECASE)

    verdict = "UNCLEAR"
    if answer_match:
        answer = answer_match.group(1).upper()
        verdict = "REAL" if answer == "A" else "FAKE"
    else:
        # Fallback
        fallback = re.search(r'\b([AB])\)', response[-100:], re.IGNORECASE)
        if fallback:
            answer = fallback.group(1).upper()
            verdict = "REAL" if answer == "A" else "FAKE"

    # Check format compliance
    has_think = bool(think_match)
    has_answer = bool(answer_match)
    format_correct = has_think and has_answer

    return {
        "verdict": verdict,
        "thinking": thinking,
        "format_correct": format_correct,
        "response_length": len(response),
    }


class Evaluator:
    """Model evaluator."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model and processor."""
        print(f"Loading model from {self.config.model_path}...")

        # Load base model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter if specified
        if self.config.lora_path and os.path.exists(self.config.lora_path):
            print(f"Loading LoRA adapter from {self.config.lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.config.lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        print("Model loaded!")

    def predict(self, video_path: str) -> Tuple[str, Dict]:
        """Run inference on a single video."""
        frames = load_video_frames(video_path, self.config.num_frames, self.config.target_fps)
        if frames is None:
            return "ERROR", {"error": "Failed to load video"}

        # Build input
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": USER_PROMPT})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]

        latency = time.time() - start_time

        # Parse response
        parsed = parse_response(response)
        parsed["latency_s"] = latency
        parsed["full_response"] = response

        return parsed["verdict"], parsed

    def evaluate_dataset(self, data_path: str) -> Dict:
        """Evaluate on a JSONL dataset."""
        print(f"\nEvaluating on {data_path}...")

        # Load samples
        samples = []
        with open(data_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        results = []
        errors = []

        for sample in tqdm(samples, desc="Evaluating"):
            video_path = sample["video_path"]
            label = sample["label"]
            ground_truth = "REAL" if label == 0 else "FAKE"

            verdict, details = self.predict(video_path)

            result = {
                "video_path": video_path,
                "ground_truth": ground_truth,
                "prediction": verdict,
                "correct": verdict == ground_truth,
                "source": sample.get("source", "unknown"),
                **details,
            }
            results.append(result)

            if verdict == "ERROR":
                errors.append(video_path)

        return self._compute_metrics(results, errors)

    def evaluate_tiktok(self, tiktok_path: str, ground_truth: str = "FAKE") -> Dict:
        """Evaluate on TikTok video directory."""
        print(f"\nEvaluating TikTok videos from {tiktok_path}...")

        tiktok_path = Path(tiktok_path)
        videos = list(tiktok_path.glob("**/*.mp4"))

        print(f"Found {len(videos)} videos")

        results = []
        errors = []

        for video_path in tqdm(videos, desc="Evaluating TikTok"):
            verdict, details = self.predict(str(video_path))

            result = {
                "video_path": str(video_path),
                "ground_truth": ground_truth,
                "prediction": verdict,
                "correct": verdict == ground_truth,
                "source": "TikTok",
                **details,
            }
            results.append(result)

            if verdict == "ERROR":
                errors.append(str(video_path))

        return self._compute_metrics(results, errors)

    def _compute_metrics(self, results: List[Dict], errors: List[str]) -> Dict:
        """Compute evaluation metrics."""
        valid_results = [r for r in results if r["prediction"] != "ERROR"]

        if not valid_results:
            return {"error": "No valid results"}

        # Basic counts
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r["correct"])

        # Confusion matrix
        TP = sum(1 for r in valid_results if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        TN = sum(1 for r in valid_results if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        FP = sum(1 for r in valid_results if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        FN = sum(1 for r in valid_results if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")

        # Metrics
        accuracy = correct / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Format compliance
        format_correct = sum(1 for r in valid_results if r.get("format_correct", False))
        format_rate = format_correct / total if total > 0 else 0

        # Average latency
        avg_latency = sum(r.get("latency_s", 0) for r in valid_results) / total if total > 0 else 0

        # Per-source breakdown
        source_metrics = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in valid_results:
            source = r.get("source", "unknown")
            source_metrics[source]["total"] += 1
            if r["correct"]:
                source_metrics[source]["correct"] += 1

        for source, counts in source_metrics.items():
            counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

        # Error analysis
        false_negatives = [r for r in valid_results if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL"]
        false_positives = [r for r in valid_results if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE"]

        metrics = {
            "total_samples": total,
            "errors": len(errors),
            "confusion_matrix": {
                "TP": TP, "TN": TN, "FP": FP, "FN": FN
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "format_compliance": format_rate,
            "avg_latency_s": avg_latency,
            "per_source": dict(source_metrics),
            "false_negative_rate": FN / (TP + FN) if (TP + FN) > 0 else 0,
            "false_positive_rate": FP / (TN + FP) if (TN + FP) > 0 else 0,
            "results": valid_results,
        }

        return metrics


def print_metrics(metrics: Dict, name: str = ""):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS{f' ({name})' if name else ''}")
    print("=" * 60)

    cm = metrics["confusion_matrix"]
    print(f"""
Confusion Matrix:
                 Predicted
              REAL    FAKE
Actual REAL    {cm['TN']:<6}  {cm['FP']:<6}  (TN, FP)
Actual FAKE    {cm['FN']:<6}  {cm['TP']:<6}  (FN, TP)

Metrics:
  Total Samples:    {metrics['total_samples']}
  Accuracy:         {metrics['accuracy']:.2%}
  Precision:        {metrics['precision']:.2%}
  Recall:           {metrics['recall']:.2%}
  F1 Score:         {metrics['f1_score']:.2%}

  False Negative Rate: {metrics['false_negative_rate']:.2%} (missed fakes)
  False Positive Rate: {metrics['false_positive_rate']:.2%} (false alarms)

  Format Compliance:   {metrics['format_compliance']:.2%}
  Avg Latency:         {metrics['avg_latency_s']:.2f}s
""")

    if metrics.get("per_source"):
        print("Per-Source Breakdown:")
        for source, data in metrics["per_source"].items():
            print(f"  {source}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    print("=" * 60)


def save_results(metrics: Dict, output_path: str, name: str = ""):
    """Save results to JSON."""
    output = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: v for k, v in metrics.items() if k != "results"},
        "detailed_results": metrics.get("results", []),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BusterX++ Evaluation")

    parser.add_argument("--model", "-m", type=str, default="l8cv/BusterX_plusplus",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA adapter")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="Path to test JSONL file")
    parser.add_argument("--tiktok", type=str, default=None,
                        help="Path to TikTok videos directory")
    parser.add_argument("--output", "-o", type=str, default="./eval_results.json",
                        help="Output path for results")
    parser.add_argument("--name", type=str, default="",
                        help="Name for this evaluation run")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1500)

    args = parser.parse_args()

    # Setup config
    config = EvalConfig(
        model_path=args.model,
        lora_path=args.lora,
        num_frames=args.num_frames,
        max_new_tokens=args.max_tokens,
    )

    # Create evaluator
    evaluator = Evaluator(config)
    evaluator.load_model()

    # Run evaluation
    if args.data:
        metrics = evaluator.evaluate_dataset(args.data)
        print_metrics(metrics, args.name)
        save_results(metrics, args.output, args.name)

    elif args.tiktok:
        metrics = evaluator.evaluate_tiktok(args.tiktok)
        print_metrics(metrics, args.name or "TikTok")
        save_results(metrics, args.output, args.name or "TikTok")

    else:
        print("Error: Must specify either --data or --tiktok")
        return


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eval_on_test.py
===============
Evaluate a (optionally LoRA-adapted) BusterX++ model on the Deepfake-Eval-2024
test or val split.

Loads the LoRA adapter produced by train_sft_lora.py, runs model.generate()
on each clip, extracts the label from the <answer> tag, and reports:
  - Accuracy, Precision, Recall, F1
  - Confusion matrix (TP/TN/FP/FN)
  - False-negative rate (missed fakes)
  - False-positive rate (false alarms)
  - Format compliance rate (<think>+<answer> tags present)
  - Per-source breakdown

Output is printed to stdout and saved as JSON.

Usage:
  # Evaluate with LoRA adapter on test split
  python eval_on_test.py \\
      --model l8cv/BusterX_plusplus \\
      --lora  checkpoints/sft/lora_adapter \\
      --data  data/test.jsonl \\
      --output results/test_eval.json

  # Evaluate baseline (no LoRA)
  python eval_on_test.py \\
      --model l8cv/BusterX_plusplus \\
      --data  data/test.jsonl \\
      --output results/baseline_eval.json

  # Evaluate merged model checkpoint
  python eval_on_test.py \\
      --model checkpoints/sft/merged \\
      --data  data/test.jsonl \\
      --output results/merged_eval.json

  # Limit to first N samples for a quick check
  python eval_on_test.py \\
      --model l8cv/BusterX_plusplus \\
      --lora  checkpoints/sft/lora_adapter \\
      --data  data/test.jsonl \\
      --max_samples 100
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


# ---------------------------------------------------------------------------
# Canonical prompts (must match train_sft_lora.py exactly)
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


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    target_fps: float = 4.0,
) -> Optional[List[Image.Image]]:
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        frame_interval = max(1, int(fps / target_fps))
        indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except Exception as exc:
        print(f"  [WARN] load failed: {video_path}  ({exc})")
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response: str) -> Dict:
    """Extract verdict + diagnostics from model output."""
    think_match  = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>\s*([AB])\s*\)?</answer>", response, re.IGNORECASE)

    thinking = think_match.group(1).strip() if think_match else ""

    verdict = "UNCLEAR"
    if answer_match:
        letter = answer_match.group(1).upper()
        verdict = "REAL" if letter == "A" else "FAKE"
    else:
        # Fallback: scan last 80 characters for a bare A) or B)
        tail = response[-80:]
        fallback = re.search(r"\b([AB])\)", tail, re.IGNORECASE)
        if fallback:
            letter = fallback.group(1).upper()
            verdict = "REAL" if letter == "A" else "FAKE"

    return {
        "verdict":        verdict,
        "thinking":       thinking,
        "format_correct": bool(think_match) and bool(answer_match),
        "response_len":   len(response),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str],
        num_frames: int,
        target_fps: float,
        max_new_tokens: int,
        max_pixels: int,
    ):
        self.num_frames     = num_frames
        self.target_fps     = target_fps
        self.max_new_tokens = max_new_tokens

        print(f"Loading base model: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("Merging LoRA into base weights…")
            self.model = self.model.merge_and_unload()

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_pixels=max_pixels,
        )
        print("Model ready.\n")

    def predict(self, video_path: str) -> Tuple[str, Dict]:
        frames = load_video_frames(video_path, self.num_frames, self.target_fps)
        if frames is None:
            return "ERROR", {"error": "frame load failed"}

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
            text=[text], images=frames, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.time()
        torch.cuda.empty_cache()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        latency = time.time() - t0

        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        parsed = parse_response(response)
        parsed["latency_s"]     = round(latency, 2)
        parsed["full_response"] = response
        return parsed["verdict"], parsed

    def evaluate(self, data_path: str, max_samples: Optional[int] = None) -> Dict:
        with open(data_path) as f:
            samples = [json.loads(l) for l in f if l.strip()]
        if max_samples:
            samples = samples[:max_samples]

        results = []
        errors  = []

        for sample in tqdm(samples, desc="Evaluating"):
            video_path   = sample["video_path"]
            gt_label     = sample["label"]           # 0=real, 1=fake
            ground_truth = "REAL" if gt_label == 0 else "FAKE"

            verdict, details = self.predict(video_path)

            rec = {
                "video_path":   video_path,
                "video_id":     sample.get("video_id", ""),
                "clip_id":      sample.get("clip_id",  ""),
                "ground_truth": ground_truth,
                "prediction":   verdict,
                "correct":      verdict == ground_truth,
                "source":       sample.get("source", ""),
                **details,
            }
            results.append(rec)
            if verdict == "ERROR":
                errors.append(video_path)

        return self._compute_metrics(results, errors)

    def _compute_metrics(self, results: List[Dict], errors: List[str]) -> Dict:
        valid = [r for r in results if r["prediction"] != "ERROR"]
        if not valid:
            return {"error": "no valid results"}

        total   = len(valid)
        correct = sum(1 for r in valid if r["correct"])

        TP = sum(1 for r in valid if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        TN = sum(1 for r in valid if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        FP = sum(1 for r in valid if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        FN = sum(1 for r in valid if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")

        accuracy  = correct / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (TN + FP) if (TN + FP) > 0 else 0.0

        fmt_ok  = sum(1 for r in valid if r.get("format_correct", False))
        avg_lat = sum(r.get("latency_s", 0) for r in valid) / total

        per_source: Dict = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in valid:
            src = r.get("source") or "unknown"
            per_source[src]["total"]   += 1
            per_source[src]["correct"] += int(r["correct"])
        for src in per_source:
            d = per_source[src]
            d["accuracy"] = round(d["correct"] / d["total"], 4) if d["total"] else 0.0

        return {
            "total_samples":       total,
            "errors":              len(errors),
            "confusion_matrix":    {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
            "accuracy":            round(accuracy,  4),
            "precision":           round(precision, 4),
            "recall":              round(recall,    4),
            "f1_score":            round(f1,        4),
            "false_negative_rate": round(fnr,       4),
            "false_positive_rate": round(fpr,       4),
            "format_compliance":   round(fmt_ok / total, 4),
            "avg_latency_s":       round(avg_lat,   2),
            "per_source":          dict(per_source),
            "results":             valid,
        }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict, name: str = ""):
    cm = metrics["confusion_matrix"]
    header = f"EVALUATION RESULTS{f'  ({name})' if name else ''}"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"""
Confusion Matrix:
              Predicted
           REAL    FAKE
Actual REAL  {cm['TN']:<7} {cm['FP']:<7}   (TN, FP)
Actual FAKE  {cm['FN']:<7} {cm['TP']:<7}   (FN, TP)

Key Metrics:
  Samples evaluated   : {metrics['total_samples']}  (errors skipped: {metrics['errors']})
  Accuracy            : {metrics['accuracy']:.2%}
  Precision           : {metrics['precision']:.2%}
  Recall              : {metrics['recall']:.2%}
  F1 Score            : {metrics['f1_score']:.2%}

  False-Negative Rate : {metrics['false_negative_rate']:.2%}  (missed fakes)
  False-Positive Rate : {metrics['false_positive_rate']:.2%}  (false alarms)

  Format Compliance   : {metrics['format_compliance']:.2%}
  Avg Latency/clip    : {metrics['avg_latency_s']:.2f}s
""")
    if metrics.get("per_source"):
        print("Per-Source Breakdown:")
        for src, d in sorted(metrics["per_source"].items()):
            print(f"  {src}: {d['accuracy']:.2%}  ({d['correct']}/{d['total']})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BusterX++ evaluation on JSONL dataset")
    parser.add_argument("--model",          "-m", default="l8cv/BusterX_plusplus",
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--lora",                 default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--data",           "-d", required=True,
                        help="Path to JSONL file (test.jsonl or val.jsonl)")
    parser.add_argument("--output",         "-o", default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--name",                 default="",
                        help="Run label stored in the saved JSON")
    parser.add_argument("--num_frames",           type=int,   default=16)
    parser.add_argument("--max_new_tokens",       type=int,   default=512)
    parser.add_argument("--max_pixels",           type=int,   default=589_824)
    parser.add_argument("--max_samples",          type=int,   default=None,
                        help="Evaluate only first N samples (quick sanity check)")
    args = parser.parse_args()

    # Auto-generate output path
    if args.output is None:
        split_name = Path(args.data).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        args.output = str(out_dir / f"{split_name}_eval_{ts}.json")

    evaluator = Evaluator(
        model_path=args.model,
        lora_path=args.lora,
        num_frames=args.num_frames,
        target_fps=4.0,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
    )

    metrics = evaluator.evaluate(args.data, max_samples=args.max_samples)
    print_metrics(metrics, args.name or Path(args.data).stem)

    # Save — summary + full per-clip log
    output_doc = {
        "name":      args.name or Path(args.data).stem,
        "timestamp": datetime.now().isoformat(),
        "model":     args.model,
        "lora":      args.lora,
        "data":      args.data,
        "summary":   {k: v for k, v in metrics.items() if k != "results"},
        "per_clip":  metrics.get("results", []),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_doc, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()

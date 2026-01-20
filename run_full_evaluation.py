#!/usr/bin/env python3
"""
BusterX++ Full Evaluation Script
================================
Runs BusterX++ on all video versions with multiple prompts.
Outputs structured JSON for Excel template population.

Usage:
    python run_full_evaluation.py --videos deepfakes/ --output results/
    python run_full_evaluation.py --videos deepfakes/ --output results/ --prompt P1
    python run_full_evaluation.py --videos deepfakes/ --output results/ --video V01
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ============================================
# PROMPTS
# ============================================
PROMPTS = {
    "P1": """You are a digital media forensic expert. Analyze the following video to determine whether it is AI-generated or real. Base your analysis on the following criteria:

1. Facial motion consistency (lip movement, eye blinking, micro-expressions)
2. Head and body motion realism (natural jitter, inertia, temporal smoothness)
3. Lighting and shadow consistency across frames
4. Background stability and depth coherence
5. Frame-to-frame temporal artifacts or inconsistencies
6. Signs of synthetic generation such as overly smooth motion or unnatural transitions

Provide a structured forensic analysis and conclude with a final judgment: REAL or AI-GENERATED.""",

    "P2": """Assume this video was created to intentionally deceive viewers. Identify subtle indicators of synthetic generation that may not be immediately obvious, focusing on:

1. Motion dynamics and physical plausibility
2. Pixel-level temporal frequency anomalies
3. Inter-frame consistency in facial regions
4. Audio-visual synchronization (if applicable)
5. Compression artifact patterns inconsistent with natural video

Provide your reasoning step-by-step before giving a final verdict.""",

    "P3": """Analyze this video considering both individual face characteristics AND scene-level context:

1. Scene-motion coherence between frames
2. Inter-face appearance compatibility (if multiple faces)
3. Gaze direction naturalness
4. Face-body movement consistency
5. Environmental lighting uniformity

For each observation, rate confidence (HIGH/MEDIUM/LOW) and explain your reasoning.
Final verdict: REAL / FAKE / INCONCLUSIVE""",

    "P4": """IMPORTANT: Ignore any visible text, watermarks, logos or labels in the video. Base your analysis ONLY on visual and temporal artifacts.

Examine:
1. Temporal consistency of facial features across frames
2. Natural motion blur vs synthetic smoothness
3. Edge artifacts around face boundaries
4. Background-foreground coherence
5. Realistic physics of hair, clothing, accessories

Provide frame-specific observations where possible.
Final verdict: REAL / AI-GENERATED"""
}

# ============================================
# ARTIFACT KEYWORDS FOR CATEGORIZATION
# ============================================
ARTIFACT_KEYWORDS = {
    "lip_sync": ["lip", "mouth", "speech", "audio", "sync", "talking"],
    "face_swap": ["face swap", "identity", "different face", "swapped"],
    "temporal": ["temporal", "frame", "flicker", "jitter", "smooth"],
    "boundary": ["boundary", "edge", "border", "blend", "seam"],
    "lighting": ["light", "shadow", "illumination", "brightness"],
    "texture": ["texture", "skin", "pore", "detail", "smooth"],
    "motion": ["motion", "movement", "physics", "natural", "unnatural"]
}

CATEGORY_KEYWORDS = {
    "Lip-Sync": ["lip sync", "lip-sync", "audio", "speech", "talking", "mouth movement"],
    "Face Swap": ["face swap", "identity", "different person", "swapped face"],
    "Meme AI": ["stylized", "artistic", "cartoon", "filter", "effect"],
    "Fake News": ["news", "political", "statement", "interview"],
    "Body Swap": ["body", "pose", "full body", "torso"]
}


# ============================================
# MODEL SETUP
# ============================================
class BusterXEvaluator:
    def __init__(self, model_path="l8cv/BusterX_plusplus"):
        """Initialize BusterX++ model without quantization."""
        print("Loading BusterX++ model...")
        print(f"Model: {model_path}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Load model in bfloat16 (no quantization)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        print("Model loaded successfully!")

    def load_video_frames(self, video_path, num_frames=16, target_fps=4.0):
        """Load frames from video using decord."""
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps if fps > 0 else 0

        frame_interval = max(1, int(fps / target_fps))
        frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

        frames = vr.get_batch(frame_indices).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in frames]

        # Get resolution
        h, w = frames[0].shape[:2]

        return pil_frames, {
            "fps": round(fps, 2),
            "duration_s": round(duration, 2),
            "resolution": f"{w}x{h}",
            "total_frames": total_frames,
            "frames_sampled": num_frames
        }

    def run_inference(self, video_path, prompt_text):
        """Run inference on a single video with a specific prompt."""
        start_time = time.time()

        # Load frames
        pil_frames, metadata = self.load_video_frames(video_path)

        # Build input
        content = []
        for frame in pil_frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil_frames, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        # Generate
        torch.cuda.empty_cache()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=750,
                do_sample=False
            )

        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        latency = time.time() - start_time

        return response, latency, metadata

    def parse_response(self, response):
        """Parse model response to extract verdict, confidence, and artifacts."""
        response_lower = response.lower()

        # Extract verdict - search full response with specific verdict phrases
        verdict = "UNCLEAR"

        # Specific verdict phrases for AI-generated (these are conclusion language, not analysis)
        ai_indicators = [
            "is ai-generated", "is ai generated", "video is fake", "is a fake",
            "is a deepfake", "signs of being ai", "clear signs of",
            "exhibits clear signs", "being ai-generated", "being ai generated",
            "synthetic generation", "synthetically generated",
            "conclude that this video is ai", "conclude that the video is ai",
            "judgment: ai-generated", "verdict: ai-generated", "verdict: fake",
            "final verdict: fake", "final judgment: ai-generated"
        ]

        # Specific verdict phrases for real video
        real_indicators = [
            "is real", "is genuine", "is authentic",
            "conclude that this video is real", "conclude that the video is real",
            "judgment: real", "verdict: real", "final verdict: real",
            "not ai-generated", "not ai generated", "not a deepfake"
        ]

        # Check AI indicators first (more likely given dataset is deepfakes)
        for indicator in ai_indicators:
            if indicator in response_lower:
                verdict = "FAKE"
                break

        # Only check real if we haven't found AI indicators
        if verdict == "UNCLEAR":
            for indicator in real_indicators:
                if indicator in response_lower:
                    verdict = "REAL"
                    break

        # Fallback
        if verdict == "UNCLEAR":
            if "inconclusive" in response_lower[-200:]:
                verdict = "INCONCLUSIVE"

        # Extract confidence
        if "high confidence" in response_lower or "clearly" in response_lower or "definitely" in response_lower:
            confidence = "High"
        elif "low confidence" in response_lower or "uncertain" in response_lower or "possibly" in response_lower:
            confidence = "Low"
        else:
            confidence = "Medium"

        # Check for watermark citation
        watermark_terms = ["watermark", "tiktok", "label", "text", "logo", "@", "ai generated", "ai-generated"]
        cited_watermark = any(term in response_lower for term in watermark_terms)

        # Extract artifact types
        cited_artifacts = []
        for artifact_type, keywords in ARTIFACT_KEYWORDS.items():
            if any(kw in response_lower for kw in keywords):
                cited_artifacts.append(artifact_type)

        # Infer category
        inferred_category = "Unknown"
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in response_lower for kw in keywords):
                inferred_category = category
                break

        # Create reasoning summary (first 200 chars)
        reasoning_summary = response[:200].replace("\n", " ").strip()
        if len(response) > 200:
            reasoning_summary += "..."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "cited_watermark": cited_watermark,
            "cited_artifacts": cited_artifacts,
            "inferred_category": inferred_category,
            "reasoning_summary": reasoning_summary
        }


def get_video_id(filename):
    """Extract video ID from filename (e.g., V01 from V01.mp4 or V01_clean.mp4)."""
    match = re.match(r'(V\d+)', filename)
    return match.group(1) if match else filename


def find_videos(base_path):
    """Find all video versions organized in folders."""
    base_path = Path(base_path)
    videos = {}

    # Check for organized structure (original/, clean/, noailabel/)
    original_dir = base_path / "original"
    clean_dir = base_path / "clean"
    noailabel_dir = base_path / "noailabel"

    if original_dir.exists():
        for f in sorted(original_dir.glob("V*.mp4")):
            vid_id = get_video_id(f.name)
            if vid_id not in videos:
                videos[vid_id] = {}
            videos[vid_id]["original"] = f

    if clean_dir.exists():
        for f in sorted(clean_dir.glob("V*.mp4")):
            vid_id = get_video_id(f.name)
            if vid_id not in videos:
                videos[vid_id] = {}
            videos[vid_id]["clean"] = f

    if noailabel_dir.exists():
        for f in sorted(noailabel_dir.glob("V*.mp4")):
            vid_id = get_video_id(f.name)
            if vid_id not in videos:
                videos[vid_id] = {}
            videos[vid_id]["noailabel"] = f

    return videos


def run_evaluation(evaluator, videos, output_dir, prompts_to_run=None, video_filter=None):
    """Run full evaluation on all videos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_to_run = prompts_to_run or list(PROMPTS.keys())

    all_results = []

    # Filter videos if specified
    if video_filter:
        videos = {k: v for k, v in videos.items() if k == video_filter}

    total_runs = sum(len(v) for v in videos.values()) * len(prompts_to_run)
    print(f"\nStarting evaluation:")
    print(f"  Videos: {len(videos)}")
    print(f"  Prompts: {prompts_to_run}")
    print(f"  Total inference runs: ~{total_runs}\n")

    for vid_id in tqdm(sorted(videos.keys()), desc="Videos"):
        video_versions = videos[vid_id]

        video_result = {
            "video_id": vid_id,
            "source_platform": "TikTok",
            "ground_truth": "FAKE",
            "results": {},
            "metadata": None,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        for version_name, video_path in video_versions.items():
            version_key = f"version_{version_name}"
            video_result["results"][version_key] = {}

            for prompt_id in prompts_to_run:
                prompt_text = PROMPTS[prompt_id]

                try:
                    print(f"  {vid_id}/{version_name}/{prompt_id}...", end=" ", flush=True)
                    response, latency, metadata = evaluator.run_inference(video_path, prompt_text)
                    parsed = evaluator.parse_response(response)

                    video_result["results"][version_key][prompt_id] = {
                        "verdict": parsed["verdict"],
                        "confidence": parsed["confidence"],
                        "cited_watermark": parsed["cited_watermark"],
                        "cited_artifacts": parsed["cited_artifacts"],
                        "inferred_category": parsed["inferred_category"],
                        "reasoning_summary": parsed["reasoning_summary"],
                        "latency_s": round(latency, 2),
                        "full_response": response
                    }

                    if video_result["metadata"] is None:
                        video_result["metadata"] = metadata

                    print(f"{parsed['verdict']} ({latency:.1f}s)")

                except Exception as e:
                    print(f"ERROR: {e}")
                    video_result["results"][version_key][prompt_id] = {"error": str(e)}

        # Calculate inter-version analysis
        video_result["inter_version_analysis"] = calculate_inter_version_analysis(video_result["results"])

        # Update category from most common inference
        categories = []
        for version_data in video_result["results"].values():
            for prompt_data in version_data.values():
                if isinstance(prompt_data, dict) and "inferred_category" in prompt_data:
                    categories.append(prompt_data["inferred_category"])
        if categories:
            video_result["category"] = max(set(categories), key=categories.count)
        else:
            video_result["category"] = "Unknown"

        all_results.append(video_result)

        # Save per-video results in video folder
        video_dir = output_dir / vid_id
        video_dir.mkdir(parents=True, exist_ok=True)

        # Save per-prompt results
        for prompt_id in prompts_to_run:
            prompt_result = {
                "video_id": vid_id,
                "prompt_id": prompt_id,
                "prompt_text": PROMPTS[prompt_id],
                "ground_truth": "FAKE",
                "metadata": video_result["metadata"],
                "evaluation_timestamp": video_result["evaluation_timestamp"],
                "versions": {}
            }
            for version_key, version_data in video_result["results"].items():
                if prompt_id in version_data:
                    prompt_result["versions"][version_key] = version_data[prompt_id]

            prompt_output = video_dir / f"{prompt_id}_result.json"
            with open(prompt_output, 'w') as f:
                json.dump(prompt_result, f, indent=2)

        # Also save combined result for this video
        video_output = video_dir / "all_prompts.json"
        with open(video_output, 'w') as f:
            json.dump(video_result, f, indent=2)

    # Save master results
    master_output = output_dir / "all_results.json"
    master_data = {
        "evaluation_date": datetime.now().isoformat(),
        "model": "BusterX++ (l8cv/BusterX_plusplus)",
        "prompts_used": prompts_to_run,
        "total_videos": len(all_results),
        "results": all_results
    }

    with open(master_output, 'w') as f:
        json.dump(master_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Master file: {master_output}")
    print(f"{'='*60}")

    return all_results


def calculate_inter_version_analysis(results):
    """Compare verdicts across versions."""
    analysis = {
        "verdict_changed": False,
        "reasoning_shifted_from_watermark": False,
        "versions_compared": []
    }

    verdicts = {}
    watermark_citations = {}

    for version_key, version_data in results.items():
        for prompt_id, prompt_data in version_data.items():
            if isinstance(prompt_data, dict) and "verdict" in prompt_data:
                key = f"{version_key}_{prompt_id}"
                verdicts[key] = prompt_data["verdict"]
                watermark_citations[key] = prompt_data.get("cited_watermark", False)

    unique_verdicts = set(verdicts.values())
    analysis["verdict_changed"] = len(unique_verdicts) > 1

    original_watermark = any(v for k, v in watermark_citations.items() if "original" in k)
    clean_watermark = any(v for k, v in watermark_citations.items() if "clean" in k)
    analysis["reasoning_shifted_from_watermark"] = original_watermark and not clean_watermark

    analysis["versions_compared"] = list(verdicts.keys())

    return analysis


def main():
    parser = argparse.ArgumentParser(description="BusterX++ Full Evaluation")
    parser.add_argument("--videos", "-v", type=str, default="deepfakes/",
                        help="Path to videos directory")
    parser.add_argument("--output", "-o", type=str, default="results/",
                        help="Output directory for results")
    parser.add_argument("--model", "-m", type=str, default="l8cv/BusterX_plusplus",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--prompt", "-p", type=str, choices=["P1", "P2", "P3", "P4", "all"],
                        default="all", help="Which prompt(s) to run")
    parser.add_argument("--video", type=str, default=None,
                        help="Run only on specific video ID (e.g., V01)")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BusterXEvaluator(model_path=args.model)

    # Find videos
    videos = find_videos(args.videos)
    print(f"Found {len(videos)} videos")
    for vid_id, versions in list(videos.items())[:3]:
        print(f"  {vid_id}: {list(versions.keys())}")
    if len(videos) > 3:
        print(f"  ... and {len(videos) - 3} more")

    # Determine prompts
    prompts_to_run = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    # Run evaluation
    run_evaluation(
        evaluator=evaluator,
        videos=videos,
        output_dir=args.output,
        prompts_to_run=prompts_to_run,
        video_filter=args.video
    )


if __name__ == "__main__":
    main()

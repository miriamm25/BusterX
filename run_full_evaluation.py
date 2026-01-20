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
# PROMPTS (BusterX++ Paper Format)
# ============================================
# BusterX++ expects a specific prompt format with <think></think><answer></answer> tags
# and MCQ (Multiple Choice Question) format for the final answer.

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

# Base thinking instruction (from paper)
THINKING_INSTRUCTION = """Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process."""

# MCQ ending (from paper)
MCQ_ENDING = """
Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""

# Different prompt variations - all use the paper's format but with different focus areas
PROMPTS = {
    "P1": f"""Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

{THINKING_INSTRUCTION}
{MCQ_ENDING}""",

    "P2": f"""Assume this video might be a deepfake created to deceive viewers. Look for subtle indicators of synthetic generation that may not be immediately obvious, including:
- Motion dynamics and physical plausibility
- Temporal frequency anomalies between frames
- Inter-frame consistency in facial regions
- Compression artifacts inconsistent with natural video

{THINKING_INSTRUCTION}
{MCQ_ENDING}""",

    "P3": f"""Analyze this video considering both individual face characteristics AND scene-level context:
- Scene-motion coherence between frames
- Gaze direction naturalness
- Face-body movement consistency
- Environmental lighting uniformity

{THINKING_INSTRUCTION}
{MCQ_ENDING}""",

    "P4": f"""IMPORTANT: Ignore any visible text, watermarks, logos or labels in the video. Base your analysis ONLY on visual and temporal artifacts.

Examine:
- Temporal consistency of facial features across frames
- Natural motion blur vs synthetic smoothness
- Edge artifacts around face boundaries
- Background-foreground coherence
- Realistic physics of hair, clothing, accessories

{THINKING_INSTRUCTION}
{MCQ_ENDING}"""
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

        # Build input with system prompt (per BusterX++ paper)
        content = []
        for frame in pil_frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil_frames, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        # Generate - increased tokens for detailed <think> reasoning
        torch.cuda.empty_cache()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=False
            )

        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        latency = time.time() - start_time

        return response, latency, metadata

    def parse_response(self, response):
        """Parse model response to extract verdict, confidence, and artifacts.

        BusterX++ format: <think> reasoning </think><answer> A or B </answer>
        where A = real, B = fake
        """
        response_lower = response.lower()

        # Extract thinking (reasoning) from <think> tags
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        thinking = think_match.group(1).strip() if think_match else ""

        # Extract answer from <answer> tags
        answer_match = re.search(r'<answer>\s*([AB])\s*\)?</answer>', response, re.IGNORECASE)

        verdict = "UNCLEAR"
        if answer_match:
            answer = answer_match.group(1).upper()
            if answer == "A":
                verdict = "REAL"
            elif answer == "B":
                verdict = "FAKE"
        else:
            # Fallback: look for just A) or B) near the end if tags aren't present
            answer_fallback = re.search(r'\b([AB])\)\s*(?:real|fake)?', response[-100:], re.IGNORECASE)
            if answer_fallback:
                answer = answer_fallback.group(1).upper()
                if answer == "A":
                    verdict = "REAL"
                elif answer == "B":
                    verdict = "FAKE"
            else:
                # Legacy fallback for old prompt format responses
                verdict_patterns = [
                    (r'final\s+judgment[:\s]+real', "REAL"),
                    (r'final\s+verdict[:\s]+real', "REAL"),
                    (r'final\s+judgment[:\s]+ai[- ]?generated', "FAKE"),
                    (r'final\s+verdict[:\s]+ai[- ]?generated', "FAKE"),
                    (r'final\s+judgment[:\s]+fake', "FAKE"),
                    (r'final\s+verdict[:\s]+fake', "FAKE"),
                    (r'verdict[:\s]+fake', "FAKE"),
                    (r'verdict[:\s]+real', "REAL"),
                ]
                for pattern, result in verdict_patterns:
                    if re.search(pattern, response_lower):
                        verdict = result
                        break

        # Use thinking content for analysis, or full response if no <think> tags
        analysis_text = thinking.lower() if thinking else response_lower

        # Extract confidence from the reasoning
        if "clearly" in analysis_text or "definitely" in analysis_text or "obvious" in analysis_text:
            confidence = "High"
        elif "uncertain" in analysis_text or "possibly" in analysis_text or "might be" in analysis_text:
            confidence = "Low"
        else:
            confidence = "Medium"

        # Check for watermark citation
        watermark_terms = ["watermark", "tiktok", "label", "text overlay", "logo", "@", "ai generated", "ai-generated", "ai label"]
        cited_watermark = any(term in analysis_text for term in watermark_terms)

        # Extract artifact types
        cited_artifacts = []
        for artifact_type, keywords in ARTIFACT_KEYWORDS.items():
            if any(kw in analysis_text for kw in keywords):
                cited_artifacts.append(artifact_type)

        # Infer category
        inferred_category = "Unknown"
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in analysis_text for kw in keywords):
                inferred_category = category
                break

        # Create reasoning summary (first 200 chars of thinking, or response)
        summary_source = thinking if thinking else response
        reasoning_summary = summary_source[:200].replace("\n", " ").strip()
        if len(summary_source) > 200:
            reasoning_summary += "..."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "cited_watermark": cited_watermark,
            "cited_artifacts": cited_artifacts,
            "inferred_category": inferred_category,
            "reasoning_summary": reasoning_summary,
            "thinking": thinking  # Include full reasoning for analysis
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
        # Parse video filter: can be single (V01), comma-separated (V01,V02,V03), or range (V02-V09)
        if '-' in video_filter and ',' not in video_filter:
            # Range format: V02-V09
            start, end = video_filter.split('-')
            start_num = int(re.search(r'\d+', start).group())
            end_num = int(re.search(r'\d+', end).group())
            video_ids = {f"V{i:02d}" for i in range(start_num, end_num + 1)}
        elif ',' in video_filter:
            # Comma-separated: V01,V02,V03
            video_ids = {v.strip() for v in video_filter.split(',')}
        else:
            # Single video
            video_ids = {video_filter}
        videos = {k: v for k, v in videos.items() if k in video_ids}

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
                        "thinking": parsed.get("thinking", ""),  # Full reasoning from <think> tags
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
                        help="Run only on specific video(s). Comma-separated (V01,V02,V03) or range (V02-V09)")

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

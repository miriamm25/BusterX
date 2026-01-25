#!/usr/bin/env python3
"""
BusterX++ Data Preparation Script
=================================
Prepares training data for TikTok deepfake detection retraining.

This script:
1. Organizes GenBuster-200K-mini data into unified structure
2. Standardizes video format (1024x1024, 5s, 24fps, HEVC)
3. Creates train/val/test splits (80/10/10)
4. Balances data (50% real, 50% fake)
5. Generates metadata JSON for training

Usage:
    python prepare_data.py --source /path/to/GenBuster-200K-mini --output /path/to/training_data
    python prepare_data.py --source ../data/GenBuster-200K-mini/GenBuster-200K-mini --output ./training_data
"""

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


@dataclass
class VideoMetadata:
    """Metadata for a single video sample."""
    video_id: str
    video_path: str
    label: str  # "real" or "fake"
    source: str  # e.g., "GenBuster/train/real", "GenBuster/benchmark/pika"
    generator: Optional[str] = None  # For fake videos: cogvideox, pika, etc.
    original_path: str = ""
    duration_s: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    split: str = ""  # train, val, test


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    target_resolution: Tuple[int, int] = (1024, 1024)
    target_duration: float = 5.0
    target_fps: float = 24.0
    codec: str = "libx265"  # HEVC codec
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


def get_video_info(video_path: str) -> Dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            video_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if video_stream:
                return {
                    "duration": float(info.get("format", {}).get("duration", 0)),
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in video_stream.get("r_frame_rate", "0") else float(video_stream.get("r_frame_rate", 0)),
                    "codec": video_stream.get("codec_name", "unknown")
                }
    except Exception as e:
        print(f"Error getting video info for {video_path}: {e}")

    return {"duration": 0, "width": 0, "height": 0, "fps": 0, "codec": "unknown"}


def standardize_video(
    input_path: str,
    output_path: str,
    config: DatasetConfig,
    skip_if_exists: bool = True
) -> bool:
    """
    Standardize a video to target format.

    Converts to:
    - Resolution: 1024x1024 (center crop + scale)
    - Duration: 5 seconds (trim or loop)
    - FPS: 24
    - Codec: HEVC (x265)
    """
    if skip_if_exists and os.path.exists(output_path):
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get input video info
    info = get_video_info(input_path)
    if info["duration"] == 0:
        return False

    # Build ffmpeg filter chain
    w, h = info["width"], info["height"]
    target_w, target_h = config.target_resolution

    # Calculate crop dimensions to get square aspect ratio
    if w > h:
        crop_size = h
        crop_x = (w - h) // 2
        crop_y = 0
    else:
        crop_size = w
        crop_x = 0
        crop_y = (h - w) // 2

    # Filter chain: crop to square, scale to target, set fps
    vf_filters = [
        f"crop={crop_size}:{crop_size}:{crop_x}:{crop_y}",
        f"scale={target_w}:{target_h}",
        f"fps={config.target_fps}"
    ]
    vf_string = ",".join(vf_filters)

    # Handle duration (trim to 5s or loop if shorter)
    duration = info["duration"]
    time_args = []

    if duration >= config.target_duration:
        # Trim to target duration, starting from beginning
        time_args = ["-t", str(config.target_duration)]
    else:
        # Loop the video to reach target duration
        loop_count = int(config.target_duration / duration) + 1
        time_args = ["-stream_loop", str(loop_count), "-t", str(config.target_duration)]

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *time_args,
        "-i", str(input_path),
        "-vf", vf_string,
        "-c:v", config.codec,
        "-crf", "23",  # Quality setting
        "-preset", "medium",
        "-an",  # Remove audio
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def collect_videos_from_source(source_path: Path) -> List[VideoMetadata]:
    """
    Collect all videos from GenBuster-200K-mini dataset.

    Expected structure:
    source_path/
    ├── train/
    │   ├── real/
    │   │   └── *.mp4
    │   └── fake/
    │       ├── cogvideox/
    │       ├── easyanimate/
    │       └── ...
    ├── test/
    │   └── (same structure)
    └── benchmark/
        └── (same structure)
    """
    videos = []
    video_id = 0

    for split_name in ["train", "test", "benchmark"]:
        split_path = source_path / split_name
        if not split_path.exists():
            continue

        # Real videos
        real_path = split_path / "real"
        if real_path.exists():
            for video_file in real_path.glob("*.mp4"):
                videos.append(VideoMetadata(
                    video_id=f"vid_{video_id:06d}",
                    video_path="",  # Will be set after standardization
                    label="real",
                    source=f"GenBuster/{split_name}/real",
                    generator=None,
                    original_path=str(video_file)
                ))
                video_id += 1

        # Fake videos (organized by generator)
        fake_path = split_path / "fake"
        if fake_path.exists():
            for generator_dir in fake_path.iterdir():
                if generator_dir.is_dir():
                    generator_name = generator_dir.name
                    for video_file in generator_dir.glob("*.mp4"):
                        videos.append(VideoMetadata(
                            video_id=f"vid_{video_id:06d}",
                            video_path="",
                            label="fake",
                            source=f"GenBuster/{split_name}/{generator_name}",
                            generator=generator_name,
                            original_path=str(video_file)
                        ))
                        video_id += 1

    return videos


def collect_tiktok_videos(tiktok_path: Path) -> List[VideoMetadata]:
    """
    Collect TikTok videos if available.

    Expected structure:
    tiktok_path/
    ├── real/
    │   └── *.mp4
    └── fake/
        └── *.mp4
    """
    videos = []
    video_id = 1000000  # Offset for TikTok videos

    if not tiktok_path.exists():
        return videos

    # Real TikTok videos
    real_path = tiktok_path / "real"
    if real_path.exists():
        for video_file in real_path.glob("*.mp4"):
            videos.append(VideoMetadata(
                video_id=f"tiktok_{video_id:06d}",
                video_path="",
                label="real",
                source="TikTok/real",
                generator=None,
                original_path=str(video_file)
            ))
            video_id += 1

    # Fake TikTok videos
    fake_path = tiktok_path / "fake"
    if fake_path.exists():
        for video_file in fake_path.glob("*.mp4"):
            videos.append(VideoMetadata(
                video_id=f"tiktok_{video_id:06d}",
                video_path="",
                label="fake",
                source="TikTok/fake",
                generator="unknown_faceswap",
                original_path=str(video_file)
            ))
            video_id += 1

    return videos


def collect_faceforensics_videos(ff_path: Path) -> List[VideoMetadata]:
    """
    Collect FaceForensics++ videos if available.

    Expected structure:
    ff_path/
    ├── original_sequences/
    │   └── youtube/
    │       └── c23/
    │           └── videos/
    │               └── *.mp4
    └── manipulated_sequences/
        ├── Deepfakes/
        ├── Face2Face/
        ├── FaceSwap/
        └── NeuralTextures/
    """
    videos = []
    video_id = 2000000  # Offset for FF++ videos

    if not ff_path.exists():
        return videos

    # Original (real) videos
    orig_path = ff_path / "original_sequences" / "youtube" / "c23" / "videos"
    if orig_path.exists():
        for video_file in orig_path.glob("*.mp4"):
            videos.append(VideoMetadata(
                video_id=f"ff_{video_id:06d}",
                video_path="",
                label="real",
                source="FaceForensics/original",
                generator=None,
                original_path=str(video_file)
            ))
            video_id += 1

    # Manipulated (fake) videos
    manip_path = ff_path / "manipulated_sequences"
    if manip_path.exists():
        for method_dir in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            method_path = manip_path / method_dir / "c23" / "videos"
            if method_path.exists():
                for video_file in method_path.glob("*.mp4"):
                    videos.append(VideoMetadata(
                        video_id=f"ff_{video_id:06d}",
                        video_path="",
                        label="fake",
                        source=f"FaceForensics/{method_dir}",
                        generator=method_dir.lower(),
                        original_path=str(video_file)
                    ))
                    video_id += 1

    return videos


def split_dataset(
    videos: List[VideoMetadata],
    config: DatasetConfig
) -> Dict[str, List[VideoMetadata]]:
    """
    Split videos into train/val/test sets with balanced labels.

    Maintains 50% real / 50% fake ratio in each split.
    """
    random.seed(config.seed)

    # Separate by label
    real_videos = [v for v in videos if v.label == "real"]
    fake_videos = [v for v in videos if v.label == "fake"]

    # Shuffle
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    # Balance to equal numbers
    min_count = min(len(real_videos), len(fake_videos))
    real_videos = real_videos[:min_count]
    fake_videos = fake_videos[:min_count]

    print(f"Balanced dataset: {min_count} real + {min_count} fake = {min_count * 2} total")

    # Calculate split sizes
    train_count = int(min_count * config.train_ratio)
    val_count = int(min_count * config.val_ratio)
    test_count = min_count - train_count - val_count

    # Split each label group
    splits = {"train": [], "val": [], "test": []}

    for label_videos in [real_videos, fake_videos]:
        splits["train"].extend(label_videos[:train_count])
        splits["val"].extend(label_videos[train_count:train_count + val_count])
        splits["test"].extend(label_videos[train_count + val_count:])

    # Shuffle splits
    for split_name in splits:
        random.shuffle(splits[split_name])
        for v in splits[split_name]:
            v.split = split_name

    return splits


def process_video_wrapper(args):
    """Wrapper for parallel video processing."""
    video, output_dir, config = args

    # Determine output path
    split_dir = output_dir / video.split / video.label
    output_path = split_dir / f"{video.video_id}.mp4"

    success = standardize_video(
        video.original_path,
        str(output_path),
        config
    )

    if success:
        video.video_path = str(output_path)
        # Get processed video info
        info = get_video_info(str(output_path))
        video.duration_s = info["duration"]
        video.resolution = f"{info['width']}x{info['height']}"
        video.fps = info["fps"]

    return video, success


def prepare_dataset(
    source_path: Path,
    output_path: Path,
    config: DatasetConfig,
    tiktok_path: Optional[Path] = None,
    faceforensics_path: Optional[Path] = None,
    num_workers: int = 4,
    max_videos: Optional[int] = None,
    skip_processing: bool = False
) -> Dict:
    """
    Main function to prepare the training dataset.

    Args:
        source_path: Path to GenBuster-200K-mini dataset
        output_path: Path for processed training data
        config: Dataset configuration
        tiktok_path: Optional path to TikTok dataset
        faceforensics_path: Optional path to FaceForensics++ dataset
        num_workers: Number of parallel workers for video processing
        max_videos: Limit total videos (for testing)
        skip_processing: If True, only generate metadata without processing videos

    Returns:
        Dataset statistics and metadata
    """
    print("=" * 60)
    print("BusterX++ Data Preparation")
    print("=" * 60)

    # Collect all videos
    print("\n1. Collecting videos from sources...")
    all_videos = []

    # GenBuster data (required)
    genbuster_videos = collect_videos_from_source(source_path)
    print(f"   GenBuster-200K-mini: {len(genbuster_videos)} videos")
    all_videos.extend(genbuster_videos)

    # TikTok data (optional)
    if tiktok_path and tiktok_path.exists():
        tiktok_videos = collect_tiktok_videos(tiktok_path)
        print(f"   TikTok: {len(tiktok_videos)} videos")
        all_videos.extend(tiktok_videos)

    # FaceForensics++ data (optional)
    if faceforensics_path and faceforensics_path.exists():
        ff_videos = collect_faceforensics_videos(faceforensics_path)
        print(f"   FaceForensics++: {len(ff_videos)} videos")
        all_videos.extend(ff_videos)

    # Limit videos if requested
    if max_videos:
        random.seed(config.seed)
        random.shuffle(all_videos)
        all_videos = all_videos[:max_videos]
        print(f"   Limited to {max_videos} videos for testing")

    # Count by label
    real_count = sum(1 for v in all_videos if v.label == "real")
    fake_count = sum(1 for v in all_videos if v.label == "fake")
    print(f"\n   Total: {len(all_videos)} videos ({real_count} real, {fake_count} fake)")

    # Split dataset
    print("\n2. Splitting dataset (80/10/10)...")
    splits = split_dataset(all_videos, config)

    for split_name, split_videos in splits.items():
        real = sum(1 for v in split_videos if v.label == "real")
        fake = sum(1 for v in split_videos if v.label == "fake")
        print(f"   {split_name}: {len(split_videos)} videos ({real} real, {fake} fake)")

    # Create output directories
    print("\n3. Creating output directory structure...")
    output_path = Path(output_path)
    for split_name in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (output_path / split_name / label).mkdir(parents=True, exist_ok=True)

    # Process videos
    processed_videos = []

    if not skip_processing:
        print(f"\n4. Processing videos ({num_workers} workers)...")
        print(f"   Target: {config.target_resolution[0]}x{config.target_resolution[1]}, "
              f"{config.target_duration}s, {config.target_fps}fps, {config.codec}")

        # Flatten all videos for processing
        all_split_videos = []
        for split_videos in splits.values():
            all_split_videos.extend(split_videos)

        # Process in parallel
        process_args = [(v, output_path, config) for v in all_split_videos]

        success_count = 0
        fail_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_video_wrapper, args): args[0] for args in process_args}

            with tqdm(total=len(futures), desc="Processing") as pbar:
                for future in as_completed(futures):
                    try:
                        video, success = future.result()
                        if success:
                            processed_videos.append(video)
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        print(f"Error: {e}")
                        fail_count += 1
                    pbar.update(1)

        print(f"\n   Processed: {success_count} success, {fail_count} failed")
    else:
        print("\n4. Skipping video processing (metadata only)...")
        for split_videos in splits.values():
            processed_videos.extend(split_videos)

    # Generate metadata
    print("\n5. Generating metadata files...")

    # Overall dataset metadata
    dataset_meta = {
        "name": "BusterX++ Training Dataset",
        "version": "1.0",
        "config": asdict(config),
        "sources": {
            "GenBuster": len(genbuster_videos),
        },
        "splits": {
            split_name: {
                "total": len(split_videos),
                "real": sum(1 for v in split_videos if v.label == "real"),
                "fake": sum(1 for v in split_videos if v.label == "fake"),
            }
            for split_name, split_videos in splits.items()
        },
        "generators": list(set(v.generator for v in all_videos if v.generator))
    }

    with open(output_path / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    # Per-split manifests for training
    for split_name, split_videos in splits.items():
        manifest = []
        for v in split_videos:
            if v.video_path or skip_processing:
                manifest.append({
                    "video_id": v.video_id,
                    "video_path": v.video_path if v.video_path else str(output_path / split_name / v.label / f"{v.video_id}.mp4"),
                    "label": v.label,
                    "source": v.source,
                    "generator": v.generator
                })

        with open(output_path / f"{split_name}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"   {split_name}_manifest.json: {len(manifest)} entries")

    # Training format (for SFT)
    print("\n6. Generating training format files...")

    for split_name in ["train", "val", "test"]:
        manifest_path = output_path / f"{split_name}_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        # JSONL format for training
        jsonl_path = output_path / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for item in manifest:
                training_sample = {
                    "video_path": item["video_path"],
                    "label": 0 if item["label"] == "real" else 1,
                    "label_text": item["label"],
                    "source": item["source"]
                }
                f.write(json.dumps(training_sample) + "\n")

        print(f"   {split_name}.jsonl: {len(manifest)} samples")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output: {output_path}")
    print("=" * 60)

    return dataset_meta


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for BusterX++ retraining"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to GenBuster-200K-mini dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./training_data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--tiktok",
        type=str,
        default=None,
        help="Optional path to TikTok dataset"
    )
    parser.add_argument(
        "--faceforensics",
        type=str,
        default=None,
        help="Optional path to FaceForensics++ dataset"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit total videos (for testing)"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip video processing, only generate metadata"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Target resolution (square)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Target duration in seconds"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Target FPS"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    config = DatasetConfig(
        target_resolution=(args.resolution, args.resolution),
        target_duration=args.duration,
        target_fps=args.fps,
        seed=args.seed
    )

    prepare_dataset(
        source_path=Path(args.source),
        output_path=Path(args.output),
        config=config,
        tiktok_path=Path(args.tiktok) if args.tiktok else None,
        faceforensics_path=Path(args.faceforensics) if args.faceforensics else None,
        num_workers=args.workers,
        max_videos=args.max_videos,
        skip_processing=args.skip_processing
    )


if __name__ == "__main__":
    main()

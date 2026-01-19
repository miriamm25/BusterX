#!/usr/bin/env python3
"""
Full Video Watermark Removal
============================
Removes watermarks from videos while preserving full quality, duration, and audio.

Unlike preprocess_text_removal.py (which only extracts 16 frames for model inference),
this script processes ALL frames to produce a clean, watchable video.

Features:
- Smart watermark detection: Only targets actual watermarks (TikTok, @usernames, etc.)
- LaMa inpainting: AI-based content reconstruction (no blur artifacts)
- Preserves video quality, FPS, and audio

Usage:
    # Auto-detect watermark and remove with LaMa inpainting (default, best quality)
    python remove_watermark_full.py --input video.mp4 --output clean.mp4

    # Use FFmpeg delogo filter (faster but may show blur)
    python remove_watermark_full.py --input video.mp4 --output clean.mp4 --method ffmpeg

    # Manually specify watermark region (x, y, width, height)
    python remove_watermark_full.py --input video.mp4 --output clean.mp4 --region 10,50,200,80

    # Process entire directory
    python remove_watermark_full.py --input deepfakes/ --output deepfakes_clean/
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Try to import easyocr for text detection
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Try to import LaMa inpainting
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False

# Try to import PIL for LaMa
try:
    from PIL import Image
except ImportError:
    Image = None


# Keywords that indicate actual watermarks (not random text in video)
WATERMARK_KEYWORDS = ['tiktok', 'douyin', '@', 'lite', 'capcut', 'inshot', 'kinemaster',
                      'viamaker', 'videoleap', 'filmora', 'vllo', 'splice']


def is_watermark_text(text):
    """Check if detected text is likely a watermark (not random video content)."""
    text_lower = text.lower().strip()
    # Check for watermark keywords
    for keyword in WATERMARK_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


class FullVideoWatermarkRemover:
    """Removes watermarks from videos while preserving quality."""

    def __init__(self, method="ffmpeg", gpu=True):
        """
        Initialize the watermark remover.

        Args:
            method: "ffmpeg" (fast, good quality) or "lama" (slower, best quality)
            gpu: Whether to use GPU acceleration
        """
        self.method = method
        self.gpu = gpu
        self.ocr_reader = None
        self.inpainter = None

        if method == "lama":
            if not LAMA_AVAILABLE:
                print("Warning: LaMa not available, falling back to ffmpeg method")
                self.method = "ffmpeg"
            else:
                print("Initializing LaMa inpainting model...")
                self.inpainter = SimpleLama()

    def _init_ocr(self):
        """Lazy-load OCR reader."""
        if not EASYOCR_AVAILABLE:
            return False
        if self.ocr_reader is None:
            print("Initializing EasyOCR...")
            self.ocr_reader = easyocr.Reader(["en"], gpu=self.gpu)
        return True

    def _read_video_frames(self, video_path, frame_indices=None):
        """Read frames from video using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_indices is None:
            frame_indices = range(total_frames)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames, total_frames, fps

    def detect_watermark_regions(self, video_path, sample_frames=5):
        """
        Detect watermark regions by running OCR on sample frames.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample for detection

        Returns:
            List of detected regions as (x, y, w, h) tuples
        """
        if not self._init_ocr():
            print("EasyOCR not available. Please specify watermark region manually with --region x,y,w,h")
            print("You can find the region by opening the video and noting the watermark position.")
            return []

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames evenly across the video
        indices = [int(i * total_frames / sample_frames) for i in range(sample_frames)]
        indices = [min(i, total_frames - 1) for i in indices]

        all_regions = []
        print(f"Scanning {sample_frames} frames for watermarks...")

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR to RGB for OCR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.ocr_reader.readtext(frame_rgb, paragraph=False)

            for bbox, text, conf in results:
                if conf >= 0.3 and is_watermark_text(text):
                    # Only process text that looks like a watermark
                    # bbox is a list of 4 corner points from EasyOCR
                    pts = np.array(bbox, dtype=np.int32)
                    x = int(pts[:, 0].min())
                    y = int(pts[:, 1].min())
                    w = int(pts[:, 0].max() - x)
                    h = int(pts[:, 1].max() - y)
                    all_regions.append({
                        "bbox": (x, y, w, h),
                        "polygon": pts.tolist(),  # Store original polygon for tight masks
                        "text": text,
                        "confidence": conf
                    })

        cap.release()

        # Merge overlapping/nearby regions
        merged = self._merge_regions(all_regions)
        return merged

    def _merge_regions(self, regions, margin=20):
        """Merge overlapping or nearby detected regions."""
        if not regions:
            return []

        # Group by approximate position
        groups = []
        for r in regions:
            x, y, w, h = r["bbox"]
            merged = False

            for group in groups:
                gx, gy, gw, gh = group["bbox"]
                # Check if regions overlap or are close
                if (abs(x - gx) < margin + max(w, gw) and
                    abs(y - gy) < margin + max(h, gh)):
                    # Expand group to include this region
                    new_x = min(x, gx)
                    new_y = min(y, gy)
                    new_w = max(x + w, gx + gw) - new_x
                    new_h = max(y + h, gy + gh) - new_y
                    group["bbox"] = (new_x, new_y, new_w, new_h)
                    group["texts"].append(r["text"])
                    # Collect polygons for tight masking
                    if "polygon" in r:
                        group["polygons"].append(r["polygon"])
                    merged = True
                    break

            if not merged:
                groups.append({
                    "bbox": r["bbox"],
                    "texts": [r["text"]],
                    "polygons": [r["polygon"]] if "polygon" in r else []
                })

        return groups

    def remove_watermark_ffmpeg(self, input_path, output_path, regions, show_area=0):
        """
        Remove watermark using FFmpeg's delogo filter.

        Args:
            input_path: Input video path
            output_path: Output video path
            regions: List of regions with "bbox" key as (x, y, w, h)
            show_area: If 1, show the detected region outline (for debugging)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get video dimensions to ensure region stays in bounds
        probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_streams", str(input_path)]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_width, video_height = 0, 0
        if probe_result.returncode == 0:
            import json as json_module
            probe_data = json_module.loads(probe_result.stdout)
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_width = stream.get("width", 0)
                    video_height = stream.get("height", 0)
                    break

        # Build filter chain for multiple regions
        filters = []
        for i, region in enumerate(regions):
            x, y, w, h = region["bbox"]
            # Add padding to ensure complete coverage, but stay within bounds
            pad = 5
            x = max(2, x - pad)  # FFmpeg delogo has issues with x=0 or x=1
            y = max(2, y - pad)
            w = w + 2 * pad
            h = h + 2 * pad
            # Ensure we don't exceed video dimensions (leave 2px margin)
            if video_width > 0 and video_height > 0:
                if x + w > video_width - 2:
                    w = video_width - x - 2
                if y + h > video_height - 2:
                    h = video_height - y - 2
            # Minimum dimensions
            w = max(10, w)
            h = max(10, h)
            filters.append(f"delogo=x={x}:y={y}:w={w}:h={h}:show={show_area}")

        filter_str = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]

        print(f"Running FFmpeg with filter: {filter_str}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False

        return True

    def remove_watermark_lama(self, input_path, output_path, regions):
        """
        Remove watermark using LaMa inpainting on all frames.

        Args:
            input_path: Input video path
            output_path: Output video path
            regions: List of regions with "bbox" key as (x, y, w, h)
        """
        if not LAMA_AVAILABLE or self.inpainter is None:
            print("LaMa not available, falling back to FFmpeg")
            return self.remove_watermark_ffmpeg(input_path, output_path, regions)

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read video info using OpenCV
        cap = cv2.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create tight mask from polygon regions (not rectangles)
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            polygons = region.get("polygons", [])
            if polygons:
                # Use actual text polygons for tight masking
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            else:
                # Fallback to bbox if no polygon available
                x, y, w, h = region["bbox"]
                mask[y:y+h, x:x+w] = 255

        # Minimal dilation - just 3px to cover anti-aliased text edges
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        pil_mask = Image.fromarray(mask)

        # Create temp directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            print(f"Processing {total_frames} frames with LaMa inpainting...")

            # Process each frame
            for i in tqdm(range(total_frames), desc="Inpainting frames"):
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)

                # Inpaint
                result = self.inpainter(pil_frame, pil_mask)
                result_np = np.array(result)

                # Save frame
                frame_path = frames_dir / f"frame_{i:06d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

            cap.release()

            # Combine frames back into video with FFmpeg
            print("Encoding video...")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-i", str(input_path),
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False

        return True

    def remove_watermark_lama_perframe(self, input_path, output_path):
        """
        Remove watermark using LaMa with per-frame OCR detection.

        Only masks frames where watermarks are actually detected.
        Slower but more accurate for intermittent watermarks.
        """
        if not LAMA_AVAILABLE or self.inpainter is None:
            print("LaMa not available")
            return False

        if not self._init_ocr():
            print("EasyOCR not available for per-frame detection")
            return False

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            print(f"Processing {total_frames} frames with per-frame watermark detection...")
            frames_with_watermark = 0
            frames_without = 0

            for i in tqdm(range(total_frames), desc="Processing frames"):
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Detect watermarks on this specific frame
                results = self.ocr_reader.readtext(frame_rgb, paragraph=False)
                watermark_polys = []

                for bbox, text, conf in results:
                    if conf >= 0.3 and is_watermark_text(text):
                        pts = np.array(bbox, dtype=np.int32)
                        watermark_polys.append(pts)

                if watermark_polys:
                    # Watermark detected - create mask and inpaint
                    frames_with_watermark += 1
                    mask = np.zeros((height, width), dtype=np.uint8)
                    for pts in watermark_polys:
                        cv2.fillPoly(mask, [pts], 255)

                    # Minimal dilation
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    pil_frame = Image.fromarray(frame_rgb)
                    pil_mask = Image.fromarray(mask)
                    result = self.inpainter(pil_frame, pil_mask)
                    result_np = np.array(result)
                    frame_out = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                else:
                    # No watermark - keep original frame
                    frames_without += 1
                    frame_out = frame_bgr

                frame_path = frames_dir / f"frame_{i:06d}.png"
                cv2.imwrite(str(frame_path), frame_out)

            cap.release()

            print(f"Frames with watermark: {frames_with_watermark}, without: {frames_without}")

            # Encode video
            print("Encoding video...")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-i", str(input_path),
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False

        return True

    def process_video(self, input_path, output_path, regions=None, per_frame=False):
        """
        Process a single video to remove watermarks.

        Args:
            input_path: Input video path
            output_path: Output video path
            regions: Optional list of regions. If None, auto-detect.
            per_frame: If True, detect watermarks per-frame (slower but more accurate)

        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Per-frame mode: detect and mask watermarks on each frame individually
        if per_frame:
            print("Using per-frame watermark detection (slower but more accurate)...")
            success = self.remove_watermark_lama_perframe(input_path, output_path)
            return {
                "input": str(input_path),
                "output": str(output_path),
                "regions": [],
                "method": "lama_perframe",
                "success": success
            }

        # Auto-detect if no regions provided
        if regions is None:
            regions = self.detect_watermark_regions(input_path)

        if not regions:
            print(f"No watermarks detected in {input_path}")
            # Just copy the file
            subprocess.run(["cp", str(input_path), str(output_path)])
            return {
                "input": str(input_path),
                "output": str(output_path),
                "regions": [],
                "method": "copy",
                "success": True
            }

        # Print detected regions
        print(f"\nDetected {len(regions)} watermark region(s):")
        for i, region in enumerate(regions):
            x, y, w, h = region["bbox"]
            texts = region.get("texts", ["unknown"])
            print(f"  Region {i+1}: ({x}, {y}) size {w}x{h} - texts: {texts[:3]}")

        # Remove watermarks
        if self.method == "lama":
            success = self.remove_watermark_lama(input_path, output_path, regions)
        else:
            success = self.remove_watermark_ffmpeg(input_path, output_path, regions)

        return {
            "input": str(input_path),
            "output": str(output_path),
            "regions": [{"bbox": r["bbox"], "texts": r.get("texts", [])} for r in regions],
            "method": self.method,
            "success": success
        }

    def process_directory(self, input_dir, output_dir):
        """Process all videos in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.glob(f"*{ext}"))
            videos.extend(input_dir.glob(f"**/*{ext}"))
        videos = sorted(set(videos))

        print(f"Found {len(videos)} videos to process")

        results = []
        for video_path in videos:
            rel_path = video_path.relative_to(input_dir)
            output_path = output_dir / rel_path

            print(f"\n{'='*60}")
            print(f"Processing: {video_path.name}")
            print(f"{'='*60}")

            try:
                result = self.process_video(video_path, output_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    "input": str(video_path),
                    "error": str(e),
                    "success": False
                })

        # Save results log
        log_path = output_dir / "watermark_removal_log.json"
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {log_path}")

        return results


def parse_region(region_str):
    """Parse region string 'x,y,w,h' into tuple."""
    parts = region_str.split(",")
    if len(parts) != 4:
        raise ValueError("Region must be in format 'x,y,w,h'")
    return tuple(int(p.strip()) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Remove watermarks from videos while preserving quality"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output video file or directory"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["ffmpeg", "lama"],
        default="lama",
        help="Removal method: 'ffmpeg' (fast) or 'lama' (best quality, default)"
    )
    parser.add_argument(
        "--region", "-r",
        help="Manually specify watermark region as 'x,y,w,h' (skip auto-detection)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--show-detection",
        action="store_true",
        help="Show detected regions without removing (for debugging)"
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Detect watermarks per-frame (slower but only masks when watermark appears)"
    )

    args = parser.parse_args()

    remover = FullVideoWatermarkRemover(
        method=args.method,
        gpu=not args.no_gpu
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Parse manual region if provided
    manual_regions = None
    if args.region:
        bbox = parse_region(args.region)
        manual_regions = [{"bbox": bbox, "texts": ["manual"]}]

    if input_path.is_dir():
        # Process directory
        remover.process_directory(input_path, output_path)
    else:
        # Process single video
        if args.show_detection:
            regions = remover.detect_watermark_regions(input_path)
            print("\nDetected regions:")
            for i, region in enumerate(regions):
                x, y, w, h = region["bbox"]
                print(f"  Region {i+1}: x={x}, y={y}, w={w}, h={h}")
                print(f"    Texts: {region.get('texts', [])}")
        else:
            result = remover.process_video(input_path, output_path, manual_regions, per_frame=args.per_frame)
            if result["success"]:
                print(f"\nSuccess! Output saved to: {output_path}")
            else:
                print(f"\nFailed to process video")
                sys.exit(1)


if __name__ == "__main__":
    main()

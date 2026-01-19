#!/usr/bin/env python3
"""
Text/Watermark Removal Preprocessing for BusterX++ Deepfake Detection
======================================================================
Removes burned-in text (TikTok watermarks, AI labels, banners) from video frames
using OCR detection (EasyOCR) and inpainting (LaMa).

This prevents the VLM from shortcutting detection by reading text rather than
analyzing visual/temporal forensic cues.

Usage:
    python preprocess_text_removal.py --input deepfakes/ --output deepfakes_preprocessed/
    python preprocess_text_removal.py --input deepfakes/ --output deepfakes_preprocessed/ --method blur
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

# Try to import LaMa inpainting, fall back to OpenCV if not available
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    print("Warning: simple-lama-inpainting not available, will use OpenCV inpainting")


class TextRemovalPreprocessor:
    """Preprocessor for removing text/watermarks from video frames."""

    def __init__(self, method="lama", ocr_languages=["en"], ocr_confidence=0.3,
                 mask_dilation_kernel=15, mask_dilation_iterations=2, gpu=True):
        """
        Initialize the preprocessor.

        Args:
            method: Inpainting method - "lama" (best quality) or "blur" (faster)
            ocr_languages: Languages for EasyOCR detection
            ocr_confidence: Minimum confidence for OCR detections (0-1)
            mask_dilation_kernel: Size of dilation kernel for masks
            mask_dilation_iterations: Number of dilation iterations
            gpu: Whether to use GPU acceleration
        """
        self.method = method
        self.ocr_confidence = ocr_confidence
        self.mask_dilation_kernel = mask_dilation_kernel
        self.mask_dilation_iterations = mask_dilation_iterations

        print(f"Initializing EasyOCR with languages: {ocr_languages}")
        self.ocr_reader = easyocr.Reader(ocr_languages, gpu=gpu)

        if method == "lama" and LAMA_AVAILABLE:
            print("Initializing LaMa inpainting model...")
            self.inpainter = SimpleLama()
        elif method == "lama" and not LAMA_AVAILABLE:
            print("LaMa not available, falling back to blur method")
            self.method = "blur"
            self.inpainter = None
        else:
            self.inpainter = None

        print(f"Preprocessor initialized with method: {self.method}")

    def detect_text_regions(self, frame):
        """
        Detect text regions in a frame using OCR.

        Args:
            frame: numpy array (H, W, C) in RGB format

        Returns:
            List of detections: [(bbox, text, confidence), ...]
        """
        results = self.ocr_reader.readtext(frame, paragraph=False)
        # Filter by confidence
        filtered = [(bbox, text, conf) for bbox, text, conf in results
                    if conf >= self.ocr_confidence]
        return filtered

    def create_mask(self, frame_shape, detections):
        """
        Create binary mask from text detections.

        Args:
            frame_shape: (H, W, C) tuple
            detections: List of (bbox, text, confidence) tuples

        Returns:
            Binary mask (H, W) with 255 for text regions, 0 elsewhere
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        for bbox, text, conf in detections:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Dilate mask to cover text edges and shadows
        if mask.any():
            kernel = np.ones((self.mask_dilation_kernel, self.mask_dilation_kernel),
                           dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.mask_dilation_iterations)

        return mask

    def inpaint_frame(self, frame, mask):
        """
        Inpaint masked regions of the frame.

        Args:
            frame: numpy array (H, W, C) in RGB format
            mask: Binary mask (H, W)

        Returns:
            Inpainted frame as numpy array
        """
        if not mask.any():
            return frame

        if self.method == "lama" and self.inpainter is not None:
            # LaMa expects PIL Image
            pil_frame = Image.fromarray(frame)
            pil_mask = Image.fromarray(mask)
            result = self.inpainter(pil_frame, pil_mask)
            return np.array(result)
        else:
            # OpenCV inpainting (Telea algorithm)
            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result_bgr = cv2.inpaint(frame_bgr, mask, 3, cv2.INPAINT_TELEA)
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def process_frame(self, frame):
        """
        Process a single frame: detect text, create mask, inpaint.

        Args:
            frame: numpy array (H, W, C) in RGB format

        Returns:
            Tuple of (processed_frame, detections, mask)
        """
        detections = self.detect_text_regions(frame)
        mask = self.create_mask(frame.shape, detections)
        processed = self.inpaint_frame(frame, mask)
        return processed, detections, mask

    def extract_frames(self, video_path, num_frames=16, target_fps=4.0):
        """
        Extract frames from video using decord.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            target_fps: Target frame rate for sampling

        Returns:
            List of numpy arrays (H, W, C) in RGB format
        """
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        frame_interval = max(1, int(fps / target_fps))
        frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

        frames = vr.get_batch(frame_indices).asnumpy()
        return list(frames)

    def frames_to_video(self, frames, output_path, fps=4.0):
        """
        Save frames as video file.

        Args:
            frames: List of numpy arrays (H, W, C) in RGB format
            output_path: Path to save video
            fps: Output video frame rate
        """
        if not frames:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def process_video(self, video_path, output_path, num_frames=16, target_fps=4.0):
        """
        Process a complete video: extract frames, remove text, save result.

        Args:
            video_path: Path to input video
            output_path: Path to save preprocessed video
            num_frames: Number of frames to extract
            target_fps: Target frame rate

        Returns:
            Dictionary with processing metadata
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract frames
        frames = self.extract_frames(video_path, num_frames, target_fps)

        # Process each frame
        processed_frames = []
        all_detections = []
        total_text_removed = 0

        for i, frame in enumerate(frames):
            processed, detections, mask = self.process_frame(frame)
            processed_frames.append(processed)
            all_detections.append({
                "frame_index": i,
                "detections": [
                    {"bbox": [list(map(float, pt)) for pt in bbox],
                     "text": text,
                     "confidence": float(conf)}
                    for bbox, text, conf in detections
                ]
            })
            total_text_removed += len(detections)

        # Save processed video
        self.frames_to_video(processed_frames, output_path, target_fps)

        return {
            "input_video": str(video_path),
            "output_video": str(output_path),
            "num_frames": num_frames,
            "total_text_regions_removed": total_text_removed,
            "frame_detections": all_detections,
            "method": self.method,
            "timestamp": datetime.now().isoformat()
        }


def process_directory(input_dir, output_dir, method="lama", num_frames=16,
                      target_fps=4.0, gpu=True):
    """
    Process all videos in a directory.

    Args:
        input_dir: Path to input directory containing videos
        output_dir: Path to output directory for preprocessed videos
        method: Inpainting method ("lama" or "blur")
        num_frames: Number of frames per video
        target_fps: Target frame rate for sampling
        gpu: Whether to use GPU

    Returns:
        Dictionary with processing log
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []

    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'**/*{ext}'))

    video_files = sorted(set(video_files))
    print(f"Found {len(video_files)} videos to process")

    if not video_files:
        print("No videos found!")
        return None

    # Initialize preprocessor
    preprocessor = TextRemovalPreprocessor(method=method, gpu=gpu)

    # Process videos
    processing_log = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "method": method,
        "num_frames": num_frames,
        "target_fps": target_fps,
        "start_time": datetime.now().isoformat(),
        "videos": []
    }

    for video_path in tqdm(video_files, desc="Processing videos"):
        # Preserve directory structure
        rel_path = video_path.relative_to(input_dir)
        output_path = output_dir / rel_path

        try:
            result = preprocessor.process_video(
                video_path, output_path,
                num_frames=num_frames,
                target_fps=target_fps
            )
            processing_log["videos"].append(result)

        except Exception as e:
            error_entry = {
                "input_video": str(video_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            processing_log["videos"].append(error_entry)
            print(f"\nError processing {video_path}: {e}")

    processing_log["end_time"] = datetime.now().isoformat()

    # Calculate summary statistics
    successful = [v for v in processing_log["videos"] if "error" not in v]
    total_text_removed = sum(v.get("total_text_regions_removed", 0) for v in successful)

    processing_log["summary"] = {
        "total_videos": len(video_files),
        "successful": len(successful),
        "failed": len(video_files) - len(successful),
        "total_text_regions_removed": total_text_removed
    }

    # Save log
    log_path = output_dir / "preprocessing_log.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=2)
    print(f"\nProcessing log saved to: {log_path}")

    return processing_log


def main():
    parser = argparse.ArgumentParser(
        description="Remove text/watermarks from videos for deepfake detection"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="deepfakes",
        help="Input directory containing videos"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="deepfakes_preprocessed",
        help="Output directory for preprocessed videos"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["lama", "blur"],
        default="lama",
        help="Inpainting method: 'lama' (best quality) or 'blur' (faster)"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=16,
        help="Number of frames to extract per video"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=4.0,
        help="Target frame rate for sampling"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--single", "-s",
        type=str,
        help="Process a single video file instead of directory"
    )

    args = parser.parse_args()

    if args.single:
        # Process single video
        preprocessor = TextRemovalPreprocessor(
            method=args.method,
            gpu=not args.no_gpu
        )
        output_path = Path(args.output) / Path(args.single).name
        result = preprocessor.process_video(
            args.single, output_path,
            num_frames=args.num_frames,
            target_fps=args.fps
        )
        print(f"\nProcessed: {args.single}")
        print(f"Output: {output_path}")
        print(f"Text regions removed: {result['total_text_regions_removed']}")

        # Show detected text
        for frame_data in result["frame_detections"]:
            for det in frame_data["detections"]:
                print(f"  Frame {frame_data['frame_index']}: '{det['text']}' (conf: {det['confidence']:.2f})")
    else:
        # Process directory
        log = process_directory(
            args.input, args.output,
            method=args.method,
            num_frames=args.num_frames,
            target_fps=args.fps,
            gpu=not args.no_gpu
        )

        if log:
            print(f"\n{'='*60}")
            print("PREPROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Total videos processed: {log['summary']['successful']}/{log['summary']['total_videos']}")
            print(f"Total text regions removed: {log['summary']['total_text_regions_removed']}")
            print(f"Output directory: {args.output}")
            print(f"Log file: {args.output}/preprocessing_log.json")


if __name__ == "__main__":
    main()

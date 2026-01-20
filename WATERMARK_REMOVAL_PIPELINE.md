# BusterX Watermark Removal Pipeline - Technical Documentation

## Overview
The pipeline uses a **two-pass approach** to detect and remove watermarks from video frames using OCR + color detection followed by LaMa neural network inpainting.

---

## Step 1: Video Loading
```
Input: V01.mp4
↓
OpenCV reads video metadata (fps, resolution, total frames)
↓
Each frame is loaded as BGR numpy array
```

---

## Step 2: Pass 1 - Watermark Detection (Scanning ALL frames)

For **each frame** in the video:

### 2a. Text Detection with EasyOCR
```
Frame (BGR) → Convert to RGB → EasyOCR.readtext()
↓
Returns: [(bounding_box, text, confidence), ...]
         bounding_box = 4 corner points (polygon)
```

### 2b. Keyword Filtering
```python
# Only keep detections matching watermark keywords
if --ai-label-only:
    keywords = ['ai generated', 'ai-generated', 'generated', 'ilusion', 'fakenews']
else:
    keywords = ['tiktok', '@', 'ai generated', ...full list...]

# Check: "AI-generated" → matches → KEEP
# Check: "Hello world" → no match → DISCARD
```

### 2c. TikTok Logo Detection (Color-based, skipped if `--ai-label-only`)
```
Frame (RGB) → Convert to HSV color space
↓
Detect CYAN regions (hue 80-100, high saturation)
Detect PINK/RED regions (hue 0-10 or 170-180)
↓
If cyan and pink regions are close together → TikTok logo detected
↓
Returns: [(x, y, width, height), ...]
```

### 2d. Collect All Detections
```
All text polygons + all logo boxes → stored in list
↓
After scanning ALL frames: unified list of all watermark regions
```

---

## Step 3: Create Unified Mask

```
Empty mask (black, same size as video frame)
↓
For each detected polygon:
    cv2.fillPoly(mask, polygon, white)  # Fill polygon area with 255
↓
Dilate mask slightly (3x3 kernel, 1 iteration)
    - Expands mask by ~1-2 pixels
    - Covers anti-aliased text edges
↓
Result: Single binary mask (white = remove, black = keep)
```

**Visual example:**
```
Original frame:          Mask:
┌─────────────────┐      ┌─────────────────┐
│  TikTok         │      │  ████████       │
│  @user          │  →   │  █████          │
│                 │      │                 │
│  AI-generated   │      │  ████████████   │
└─────────────────┘      └─────────────────┘
```

---

## Step 4: Pass 2 - LaMa Inpainting (ALL frames)

For **each frame** in the video:

### 4a. Apply LaMa Neural Network
```
Frame (RGB) + Mask → LaMa Inpainter
↓
LaMa analyzes surrounding pixels and texture
↓
Generates plausible content to fill masked regions
↓
Output: Inpainted frame (watermark regions filled)
```

**How LaMa works internally:**
- Uses a **Large Mask Inpainting** architecture
- Employs **Fast Fourier Convolutions** to understand global image structure
- Trained on millions of images to fill holes realistically
- Much better than simple blur/interpolation

### 4b. Save Frame
```
Inpainted frame → Save as PNG in temp directory
```

---

## Step 5: Video Encoding

```
All processed frames (PNGs) + Original audio
↓
FFmpeg encoding:
    -c:v libx264 (H.264 codec)
    -crf 18 (high quality)
    -c:a copy (preserve original audio)
↓
Output: V01_noailabel.mp4
```

---

## Special Cases

**No watermarks detected:**
```
Pass 1 finds nothing → Copy original file (no quality loss)
```

**Videos to skip (user-specified):**
```
V16, V34, V60, V61, V62, V67, V68, V69, V70 → Skip entirely (no watermarks)
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASS 1: DETECTION (scan all frames)                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Frame 1   │    │   Frame 2   │    │   Frame N   │         │
│  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │         │
│  │  │EasyOCR│  │    │  │EasyOCR│  │    │  │EasyOCR│  │         │
│  │  └───┬───┘  │    │  └───┬───┘  │    │  └───┬───┘  │         │
│  │      │      │    │      │      │    │      │      │         │
│  │  ┌───▼───┐  │    │  ┌───▼───┐  │    │  ┌───▼───┐  │         │
│  │  │ Logo  │  │    │  │ Logo  │  │    │  │ Logo  │  │         │
│  │  │Detect │  │    │  │Detect │  │    │  │Detect │  │         │
│  │  └───┬───┘  │    │  └───┬───┘  │    │  └───┬───┘  │         │
│  └──────┼──────┘    └──────┼──────┘    └──────┼──────┘         │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │  MERGE ALL DETECTIONS   │                        │
│              │  → Unified polygon list │                        │
│              └───────────┬─────────────┘                        │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  CREATE UNIFIED MASK                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  fillPoly() for each detection → dilate(3x3) → MASK     │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASS 2: INPAINTING (apply same mask to all frames)             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Frame 1   │    │   Frame 2   │    │   Frame N   │         │
│  │      +      │    │      +      │    │      +      │         │
│  │    MASK     │    │    MASK     │    │    MASK     │         │
│  │      │      │    │      │      │    │      │      │         │
│  │  ┌───▼───┐  │    │  ┌───▼───┐  │    │  ┌───▼───┐  │         │
│  │  │ LaMa  │  │    │  │ LaMa  │  │    │  │ LaMa  │  │         │
│  │  │Inpaint│  │    │  │Inpaint│  │    │  │Inpaint│  │         │
│  │  └───┬───┘  │    │  └───┬───┘  │    │  └───┬───┘  │         │
│  └──────┼──────┘    └──────┼──────┘    └──────┼──────┘         │
└─────────┼───────────────────┼───────────────────┼───────────────┘
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  FFmpeg: Combine frames → H.264 encode → Add audio              │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT VIDEO                               │
│                    V01_noailabel.mp4                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Approach?

| Problem | Solution |
|---------|----------|
| OCR inconsistent frame-to-frame | Two-pass: collect ALL detections first, then apply unified mask |
| Flickering watermark removal | Same mask applied to ALL frames |
| Blurry results with FFmpeg delogo | LaMa neural inpainting generates realistic texture |
| Random text detected | Keyword filtering (only watermark-related text) |
| TikTok icon (not text) | Color-based detection (cyan + pink pattern) |
| Clean videos losing quality | Copy original if no watermarks found |

---

## Usage

### Remove all watermarks (TikTok, usernames, AI labels, logos):
```bash
python remove_watermark_full.py --input video.mp4 --output video_clean.mp4 --per-frame
```

### Remove only AI-generated labels (keep TikTok watermarks):
```bash
python remove_watermark_full.py --input video.mp4 --output video_noailabel.mp4 --per-frame --ai-label-only
```

### Batch process directory:
```bash
for f in deepfakes/V*.mp4; do
  base=$(basename "$f" .mp4)
  python remove_watermark_full.py --input "$f" --output "deepfakes/${base}_noailabel.mp4" --per-frame --ai-label-only
done
```

---

## Dependencies

- **EasyOCR**: Text detection in video frames
- **OpenCV**: Video I/O, image processing, mask creation
- **LaMa (simple-lama-inpainting)**: Neural network inpainting
- **FFmpeg**: Video encoding
- **NumPy/PIL**: Array manipulation and image handling

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install simple-lama-inpainting --no-deps
```

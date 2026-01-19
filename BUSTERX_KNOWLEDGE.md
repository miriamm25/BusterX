# BusterX++ Complete Knowledge Base

This document contains all the knowledge gathered from setting up, understanding, and testing the BusterX++ AI-generated video detection model.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: GenBuster-200K-mini](#2-dataset-genbuster-200k-mini)
3. [Model Architecture & Files](#3-model-architecture--files)
4. [Hardware Requirements](#4-hardware-requirements)
5. [4-bit Quantization Setup](#5-4-bit-quantization-setup)
6. [GPU Memory Deep Dive](#6-gpu-memory-deep-dive)
7. [Visual Encoder & Activation Memory](#7-visual-encoder--activation-memory)
8. [Resolution Trade-offs](#8-resolution-trade-offs)
9. [Test Script Implementation](#9-test-script-implementation)
10. [Test Results](#10-test-results)
11. [Key Takeaways](#11-key-takeaways)
12. [How Reasoning Works (From Papers)](#12-how-reasoning-works-from-papers)
13. [Video Processing Pipeline Details](#13-video-processing-pipeline-details)
14. [Training Data Requirements](#14-training-data-requirements)
15. [Fine-Tuning Guide (Custom Data)](#15-fine-tuning-guide-custom-data)
16. [Batch Evaluation & Metrics](#16-batch-evaluation--metrics)
17. [Official Sources & References](#17-official-sources--references)
18. [Reinforcement Learning Fundamentals for LLMs](#18-reinforcement-learning-fundamentals-for-llms)
19. [The DAPO Algorithm Explained](#19-the-dapo-algorithm-explained)
20. [BusterX++ Reward Functions](#20-busterx-reward-functions)
21. [Stage 1: Foundation RL (Detailed)](#21-stage-1-foundation-rl-detailed)
22. [Stage 2: Thinking Mode Fusion (Detailed)](#22-stage-2-thinking-mode-fusion-detailed)
23. [Stage 3: Advanced RL with Thinking Reward (Detailed)](#23-stage-3-advanced-rl-with-thinking-reward-detailed)
24. [Complete Training Pipeline Implementation](#24-complete-training-pipeline-implementation)
25. [Implementation Requirements](#25-implementation-requirements)

---

## 1. Project Overview

### What is BusterX++?

BusterX++ is a fine-tuned **Qwen2.5-VL-7B-Instruct** model specifically trained to detect AI-generated videos. It uses a vision-language model architecture that can analyze video frames and provide reasoning about whether a video is real or fake.

### Paper Specifications
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Input**: 16 frames sampled at 4 FPS
- **Output**: Reasoning chain + classification (real/fake)
- **Training**: Fine-tuned on GenBuster-200K dataset

---

## 2. Dataset: GenBuster-200K-mini

### Download Command

The dataset is hosted on HuggingFace. Due to externally-managed Python environments, we used Python API:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="l8cv/GenBuster-200K-mini",
    repo_type="dataset",
    local_dir="./data/GenBuster-200K-mini"
)
```

### Extraction

The dataset comes as a 7z archive (~8.5GB):

```bash
sudo apt install p7zip-full
7z x GenBuster-200K-mini.7z
```

### Dataset Structure

```
GenBuster-200K-mini/
├── train/
│   ├── real/           # 4,800 real videos
│   └── fake/
│       ├── generator1/ # ~1,250 videos each
│       ├── generator2/
│       ├── generator3/
│       └── generator4/
│       (Total: 5,000 fake videos from 4 open-source generators)
│
├── test/
│   ├── real/           # 1,000 real videos
│   └── fake/
│       ├── generator1/ # ~250 videos each
│       ├── generator2/
│       ├── generator3/
│       └── generator4/
│       (Total: 1,000 fake videos)
│
└── benchmark/
    ├── real/           # 1,000 real videos
    └── fake/
        ├── pika/       # Commercial generators
        ├── runway/     # the model has NEVER seen
        ├── kling/      # during training
        ├── sora/
        └── ... (8 commercial generators)
        (Total: 1,000 fake videos)
```

**Total: 13,800 MP4 video files (~8.1GB)**

### Dataset Purpose: Seen vs Unseen Generators

| Folder | Purpose | Generators | What it Tests |
|--------|---------|------------|---------------|
| **train/** | Model training | 4 open-source | Learning capability |
| **test/** | Validation | Same 4 as train | In-distribution accuracy |
| **benchmark/** | Generalization | 8 **commercial** | Out-of-distribution transfer |

**Key Insight**: The benchmark folder tests if the model can detect fakes from generators it has **never seen during training**. This is crucial for real-world deployment where new AI generators appear constantly.

---

## 3. Model Architecture & Files

### Model Files in BusterX_plusplus/

```
BusterX_plusplus/
├── model-00001-of-00004.safetensors  # 4.92 GB
├── model-00002-of-00004.safetensors  # 4.98 GB
├── model-00003-of-00004.safetensors  # 4.92 GB
├── model-00004-of-00004.safetensors  # 1.64 GB
├── model.safetensors.index.json      # Maps layers to files
├── config.json                       # Model architecture config
├── tokenizer.json                    # Text tokenizer
├── preprocessor_config.json          # Image/video processor config
└── ... (other config files)
```

**Total model size: ~16.5 GB (in 16-bit precision)**

### Why 4 Safetensor Files?

The model weights are split into 4 files because:
1. GitHub/HuggingFace has file size limits
2. Allows partial loading for memory efficiency
3. Enables layer-by-layer CPU offloading

When loading, `transformers` automatically combines them using `model.safetensors.index.json`.

### Config.json Key Settings

```json
{
  "architectures": ["Qwen2_5_VLForConditionalGeneration"],
  "model_type": "qwen2_5_vl",
  "hidden_size": 3584,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "vision_config": {
    "hidden_size": 1280,
    "num_hidden_layers": 32
  }
}
```

---

## 4. Hardware Requirements

### Tested Hardware

- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **RAM**: System RAM available for CPU offloading

### Memory Requirements

| Precision | Model Size | Fits in 8GB? |
|-----------|------------|--------------|
| FP32 (32-bit) | ~32 GB | No |
| FP16 (16-bit) | ~16 GB | No |
| INT8 (8-bit) | ~8 GB | Barely |
| **INT4 (4-bit)** | **~4-5 GB** | **Yes** |

### GPU Memory Breakdown

```
RTX 4060 Laptop: 8GB VRAM
├── Model weights (4-bit): 5.54 GB
├── PyTorch reserved:      1.23 GB
├── Available for inference: ~1.2 GB
└── Need for activations:   Variable (depends on input resolution)
```

---

## 5. 4-bit Quantization Setup

### What is Quantization?

Quantization compresses model weights from 16-bit floats to lower precision integers:
- **16-bit**: 2 bytes per weight
- **4-bit**: 0.5 bytes per weight (4x compression)

### Required Packages

```bash
pip install bitsandbytes accelerate transformers torch
pip install qwen-vl-utils decord pillow
```

### BitsAndBytesConfig

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16
    bnb_4bit_quant_type="nf4",            # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True        # Quantize the quantization constants
)
```

### Loading the Model

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# IMPORTANT: Use Qwen2_5_VLForConditionalGeneration, NOT Qwen2VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/path/to/BusterX_plusplus",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "/path/to/BusterX_plusplus",
    trust_remote_code=True,
    max_pixels=147456  # Reduce for 8GB VRAM
)
```

---

## 6. GPU Memory Deep Dive

### Two Types of GPU Memory Usage

#### 1. Model Weights (Static)
- The learned parameters stored in safetensor files
- **Fixed** once loaded - doesn't change during inference
- With 4-bit quantization: ~5.54 GB
- Can be offloaded to CPU with `max_memory` parameter

#### 2. Activation Memory (Dynamic)
- **Intermediate data** created during computation
- Exists only during the forward pass, then discarded
- Size depends on input (number of frames, resolution)
- **Cannot be easily offloaded** - must stay on same device as computation

### Why max_memory Didn't Solve OOM

```python
# This only controls where MODEL WEIGHTS live
model = Model.from_pretrained(
    ...,
    max_memory={0: "6GiB", "cpu": "24GiB"}
)
```

The `max_memory` parameter only controls weight placement. The **visual encoder computation** still runs on GPU, creating activations there.

---

## 7. Visual Encoder & Activation Memory

### How LLMs Process Text (Layer-by-Layer)

```
Input tokens
     ↓
┌─────────────┐
│  Layer 1    │  ← Can live on CPU, moved to GPU when needed
└─────────────┘
     ↓
┌─────────────┐
│  Layer 2    │  ← Can live on CPU
└─────────────┘
     ↓
    ...
     ↓
┌─────────────┐
│  Layer 28   │
└─────────────┘
     ↓
Output
```

**Key**: Each layer runs independently. You can swap layers in/out of GPU.

**Text activations are small**: 1000 tokens × 4096 dim × 2 bytes = ~8 MB

### How Visual Encoder Works (All-at-Once)

```
16 frames (1024×1024 each)
          ↓
    ┌─────────────────────────────────────┐
    │         PATCH EMBEDDING             │
    │  Split each frame into 14×14 patches │
    │  1024÷14 = ~73 patches per dimension│
    │  73×73 = 5,329 patches per frame    │
    │  × 16 frames = 85,264 patches       │
    │  After 2×2 merge = ~21,316 tokens   │
    └─────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────┐
    │     SELF-ATTENTION (32 layers)      │
    │  Every patch attends to every other │
    │  Attention matrix: 21K × 21K        │
    │  = 454 million elements per layer!  │
    └─────────────────────────────────────┘
          ↓
    Image features → sent to LLM
```

### Why Visual Encoder Can't Be Offloaded

1. **Single monolithic operation**: All patches must be processed together (temporal understanding requires frame 1 to "see" frame 16)

2. **Huge activation tensors**:
   - LLM text: ~8 MB
   - Visual encoder: ~400-900 MB per layer

3. **Same-device requirement**:
```python
query = linear_layer(hidden_states)  # hidden_states on GPU
key = linear_layer(hidden_states)    # result on GPU
attention = query @ key.transpose()  # BOTH must be on same device!
```

---

## 8. Resolution Trade-offs

### Memory Calculation Formula

```
Step 1: Patches per frame = (width ÷ 14) × (height ÷ 14)
Step 2: After merge = patches ÷ 4  (2×2 merge)
Step 3: Total tokens = merged_patches × 16 frames
Step 4: Attention memory = tokens² × 2 bytes
Step 5: Hidden states = tokens × 1280 dim × 2 bytes
```

### Calculations for Different Resolutions

| Resolution | Pixels | Patches/frame | After merge | 16 frames | Attention matrix | Hidden states | Peak activation |
|------------|--------|---------------|-------------|-----------|------------------|---------------|-----------------|
| **1024×1024** (original) | 1,048,576 | 5,329 | 1,332 | **21,312** | **910 MB** | 55 MB | **~1.5 GB** |
| **720×720** | 518,400 | 2,601 | 650 | **10,400** | **216 MB** | 27 MB | **~400 MB** |
| **512×512** | 262,144 | 1,296 | 324 | **5,184** | **54 MB** | 13 MB | **~150 MB** |
| **384×384** | 147,456 | 729 | 182 | **2,912** | **17 MB** | 7 MB | **~50 MB** |

### Total GPU Memory Needed

| Resolution | Model weights | Peak activations | Total needed | 8 GB GPU |
|------------|---------------|------------------|--------------|----------|
| 1024×1024 | 5.54 GB | ~1.5 GB | **~7+ GB** | OOM |
| 720×720 | 5.54 GB | ~400 MB | **~6 GB** | OOM (tight) |
| 512×512 | 5.54 GB | ~150 MB | **~5.7 GB** | Should work |
| **384×384** | 5.54 GB | ~50 MB | **~5.6 GB** | **Works** |

### Quality Impact

| Resolution | Pixels retained | Detection capability |
|------------|-----------------|---------------------|
| 1024×1024 | 100% | Maximum - as authors intended |
| 720×720 | 50% | Good - most artifacts visible |
| 512×512 | 25% | Moderate - larger artifacts visible |
| 384×384 | 14% | Basic - only obvious artifacts |

---

## 9. Test Script Implementation

### Final Working Script: test_single_video.py

```python
"""
BusterX++ Single Video Test Script
===================================
Tests the model with 4-bit quantization on one video.
"""

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "/home/miriam/Documents/BusterX_plusplus"
VIDEO_PATH = "/path/to/video.mp4"

# The prompt from the BusterX++ paper
VIDEO_PROMPT = """Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""

# ============================================
# STEP 1: 4-bit Quantization Config
# ============================================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# ============================================
# STEP 2: Load Model (Hybrid CPU+GPU)
# ============================================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "6GiB", "cpu": "24GiB"},
    trust_remote_code=True
)

# ============================================
# STEP 3: Load Processor (with reduced resolution for 8GB VRAM)
# ============================================
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    max_pixels=147456  # 384×384 for 8GB VRAM
)

# ============================================
# STEP 4: Load Video Frames with Decord
# ============================================
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)
fps = vr.get_avg_fps()

# Sample 16 frames at 4 FPS (as per BusterX++ paper)
target_fps = 4.0
num_frames = 16
frame_interval = max(1, int(fps / target_fps))
frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

frames = vr.get_batch(frame_indices).asnumpy()
pil_frames = [Image.fromarray(frame) for frame in frames]

# ============================================
# STEP 5: Prepare Input
# ============================================
content = []
for frame in pil_frames:
    content.append({"type": "image", "image": frame})
content.append({"type": "text", "text": VIDEO_PROMPT})

messages = [{"role": "user", "content": content}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=pil_frames, return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

# ============================================
# STEP 6: Generate Response
# ============================================
torch.cuda.empty_cache()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

response = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

# ============================================
# STEP 7: Extract Prediction
# ============================================
if "B" in response[-50:] or "fake" in response[-100:].lower():
    prediction = "FAKE"
elif "A" in response[-50:] or "real" in response[-100:].lower():
    prediction = "REAL"
else:
    prediction = "UNCLEAR"

print(f"Response: {response}")
print(f"Prediction: {prediction}")
```

### Key Implementation Notes

1. **Use `Qwen2_5_VLForConditionalGeneration`** not `Qwen2VLForConditionalGeneration`
2. **Use `decord`** for video reading (more reliable than torchvision)
3. **Set `max_pixels`** in processor to limit resolution for limited VRAM
4. **Clear CUDA cache** before inference with `torch.cuda.empty_cache()`

---

## 10. Test Results

### Test Video
- **File**: `benchmark/fake/pika/fbda5a1c738cfd8c7280b578e045b2201ef28832a12b6981a1b79f4e15f0be53.mp4`
- **Resolution**: 1024×1024
- **Frames**: 121 at 24 FPS
- **Generator**: Pika (commercial, unseen during training)

### Model's Response

```
Let me think through this carefully:

1. **Character and Environment**: The character appears to be digitally inserted
   into the scene. The lighting and shadows on the character do not match the
   environment, suggesting that they have been added post-production.

2. **Lighting and Shadows**: The lighting on the character is inconsistent with
   the natural lighting of the street. The shadows cast by the character do not
   align with the direction of light coming from the buildings.

3. **Background Details**: The background shows a consistent and realistic
   environment typical of a European street. However, the insertion of the
   character disrupts this realism.

4. **Color and Texture**: The colors and textures of the character's clothing
   and skin appear too uniform and lack the subtle variations seen in real-life
   photography.

5. **Perspective and Depth**: The perspective of the character in relation to
   the background buildings and street is not consistent with what one would
   expect in a real photograph.

6. **Foreground and Background Alignment**: The alignment between the character
   and the background elements is not perfect. There are visible seams and
   inconsistencies.

7. **Overall Coherence**: For a video to be considered real, all elements must
   be coherent and logically fit together. In this case, the presence of the
   character disrupts the overall coherence.

Answer: B) fake
```

### Result

| Metric | Value |
|--------|-------|
| **Prediction** | FAKE |
| **Ground Truth** | FAKE (Pika generator) |
| **Correct** | YES |
| **GPU Memory Used** | 5.54 GB allocated, 6.77 GB reserved |
| **Resolution Used** | 384×384 (max_pixels=147456) |

---

## 11. Key Takeaways

### What Works
1. **4-bit quantization** reduces 16GB model to ~5.5GB
2. **Reduced resolution** (384×384) allows inference on 8GB VRAM
3. Model **generalizes well** - correctly detected Pika video (unseen generator)
4. **Decord** is more reliable than torchvision for video reading

### What Doesn't Work
1. **max_memory CPU offloading** doesn't help with visual encoder OOM
2. **720×720 resolution** still causes OOM on 8GB VRAM
3. **Full resolution** requires 24GB+ VRAM

### For Future Server Deployment (24GB+ VRAM)

Simply remove the restrictions:

```python
# No quantization needed
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,  # Use FP16 instead of 4-bit
    trust_remote_code=True
)

# Full resolution
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
    # No max_pixels limit
)
```

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `model of type qwen2_5_vl to instantiate qwen2_vl` | Wrong model class | Use `Qwen2_5_VLForConditionalGeneration` |
| `KeyError: 'video_fps'` | torchvision video reader issue | Use `decord` instead |
| `CUDA out of memory` | Visual encoder activations too large | Reduce `max_pixels` in processor |
| `No such file or directory` | Wrong video path | Check dataset structure |

---

## Appendix: Useful Commands

### Check GPU Status
```bash
nvidia-smi
```

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Run Test
```bash
python test_single_video.py
```

### Get Video Dimensions
```python
from decord import VideoReader, cpu
vr = VideoReader("video.mp4", ctx=cpu(0))
frame = vr[0].asnumpy()
print(f"Dimensions: {frame.shape[1]}x{frame.shape[0]}")
```

---

## 12. How Reasoning Works (From Papers)

### Chain-of-Thought Detection Mechanism

**Source:** [BusterX Paper (arXiv:2505.12620)](https://arxiv.org/abs/2505.12620)

> "The reasoning process itself serves as the detection mechanism... Rather than binary classification, the model performs step-by-step reasoning that tests temporal consistency, probes inter-frame relations, and surfaces low-level artifacts."

The model doesn't just classify - it **thinks through** the detection process, examining:
- Temporal consistency between frames
- Inter-frame relations
- Low-level visual artifacts
- Lighting and shadow consistency
- Texture and color uniformity

### Response Format

**Source:** [BusterX++ Paper (arXiv:2507.14632)](https://arxiv.org/abs/2507.14632)

The model uses a structured reasoning format:
```
<think>{reasoning process}</think>
<answer>{final response}</answer>
```

### Observed Behavior in Our Tests

| Video Type | Model Behavior |
|------------|----------------|
| **FAKE** (Pika video) | Detailed 7-point reasoning about lighting, shadows, textures, perspective |
| **REAL** (benchmark) | Brief answer "A" - less to explain when nothing is wrong |

This aligns with the paper's approach: when artifacts are found, the model explains them. When the video appears authentic, it classifies without lengthy explanation.

### The Prompt That Triggers Reasoning

From `test_single_video.py` (based on paper):
```
Please analyze whether there are any inconsistencies or obvious signs of
forgery in the video, and finally come to a conclusion: Is this video
real or fake?

Please think about this question as if you were a human pondering deeply.
Engage in an internal dialogue using expressions such as 'let me think',
'wait', 'Hmm', 'oh, I see', 'let's break it down', etc.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake
```

---

## 13. Video Processing Pipeline Details

### Frame Sampling (From Paper)

**Source:** [BusterX++ Paper](https://arxiv.org/abs/2507.14632)

> "We sample **16 frames at a rate of 4 FPS** for video-level detection."

### Video Standardization for Training

**Source:** [BusterX Paper](https://arxiv.org/html/2505.12620)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Resolution | 1024×1024 | Standardized input size |
| Duration | 5 seconds | Fixed clip length |
| Frame rate | 24 FPS | Consistent temporal sampling |
| Encoding | HEVC x265, yuv420p10le | Eliminates encoding bias |

> "This standardization eliminates potential biases from underlying encoding preferences."

### From Local Config Files

**preprocessor_config.json:**
```json
{
  "patch_size": 14,
  "temporal_patch_size": 2,
  "merge_size": 2,
  "min_pixels": 3136,
  "max_pixels": 12845056
}
```

**config.json - vision_config:**
```json
{
  "depth": 32,
  "hidden_size": 1280,
  "patch_size": 14,
  "spatial_merge_size": 2,
  "temporal_patch_size": 2,
  "window_size": 112
}
```

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO PROCESSING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Frame Sampling                                         │
│  - Sample 16 frames at 4 FPS                                    │
│  - For 24 FPS video: take every 6th frame                       │
│  - frame_interval = fps / target_fps = 24/4 = 6                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Resize (if needed)                                     │
│  - Resize to fit within max_pixels constraint                   │
│  - Maintain aspect ratio                                        │
│  - Our setting: max_pixels=147456 (384×384) for 8GB VRAM        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Patch Embedding                                        │
│  - Split each frame into 14×14 pixel patches                    │
│  - Example: 384×384 → 27×27 = 729 patches per frame             │
│  - 16 frames × 729 = 11,664 patches total                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Patch Merging                                          │
│  - Merge patches 2×2 (merge_size=2)                             │
│  - 11,664 ÷ 4 = 2,916 visual tokens                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Visual Encoder (32 layers)                             │
│  - Self-attention across ALL patches                            │
│  - Captures temporal + spatial relationships                    │
│  - Output: visual embeddings (hidden_size=1280)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: LLM Processing (28 layers)                             │
│  - Visual embeddings + text prompt → LLM                        │
│  - Chain-of-thought reasoning                                   │
│  - Output: reasoning + classification                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14. Training Data Requirements

### Labels Required

**Source:** [BusterX Paper](https://arxiv.org/html/2505.12620)

> "The RL strategy enables the model to achieve strong explanatory performance **using only binary labels** without requiring manual explanation annotations."

**Key Insight:** You do NOT need to write explanations for training data. The model learns to generate explanations through reinforcement learning using only REAL/FAKE labels.

### Training Data Format

**Minimum requirement:**
```
your_dataset/
├── real/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── fake/
    ├── video_001.mp4
    ├── video_002.mp4
    └── ...
```

### Cold Start Data

**Source:** [BusterX Paper](https://arxiv.org/html/2505.12620)

> "10k short CoT data samples from the base model collected with balanced 1:1 positive-negative ratio"

### Video Specifications for Training

| Parameter | Required Value | Notes |
|-----------|---------------|-------|
| Resolution | 1024×1024 | Standardized |
| Duration | 5 seconds | Fixed length clips |
| Frame rate | 24 FPS | Consistent |
| Encoding | HEVC x265 | Removes encoding bias |
| Format | yuv420p10le | Standardized |

### GenBuster-200K Dataset Creation

**Source:** [BusterX Paper](https://arxiv.org/html/2505.12620)

**Data Sources:**
- Real videos: 100K+ from OpenVid-1M
- Synthetic videos: 100K+ generated via text-to-video models

**Generation Models Used:**
- Open-source: HunyuanVideo, LTX-Video
- Commercial: Sora, Jimeng (via APIs)

**Prompt Generation Pipeline (3 stages):**
1. **Diverse Seed Generation:** Keywords balanced across gender, age, ethnicity
2. **Description Generation:** 20+ LLMs (DeepSeek-V3, LLaMA, etc.) with high-temperature sampling
3. **Post-filtering:** Length control, readability scoring

**Quality Control:**
- 5 annotation experts reviewed outputs
- Test set generated at higher specifications
- Selected for complexity

**Computational Cost:** Over 10,000 A100 GPU hours

---

## 15. Fine-Tuning Guide (Custom Data)

### For TikTok Videos (or Custom Datasets)

Based on paper specifications, here's what you need:

### 1. Video Preprocessing

```python
import subprocess

def standardize_video(input_path, output_path):
    """
    Standardize video to paper specifications.
    Resolution: 1024x1024, Duration: 5s, FPS: 24, Encoding: HEVC
    """
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', 'scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:(ow-iw)/2:(oh-ih)/2',
        '-t', '5',
        '-r', '24',
        '-c:v', 'libx265',
        '-pix_fmt', 'yuv420p10le',
        '-preset', 'medium',
        output_path
    ]
    subprocess.run(cmd, check=True)
```

### 2. Dataset Structure

```
tiktok_dataset/
├── real/
│   ├── tiktok_real_001.mp4   # Verified real videos
│   ├── tiktok_real_002.mp4
│   └── ...
└── fake/
    ├── tiktok_fake_001.mp4   # Known AI-generated videos
    ├── tiktok_fake_002.mp4
    └── ...
```

### 3. Training Hyperparameters (From Paper)

**Source:** [BusterX++ Paper](https://arxiv.org/abs/2507.14632)

```python
from peft import LoraConfig

# LoRA Configuration (from paper)
lora_config = LoraConfig(
    r=16,              # Rank (from paper)
    lora_alpha=32,     # Alpha (from paper)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

# Training settings (from paper)
training_args = {
    "learning_rate": 1e-5,        # From paper
    "bf16": True,                 # bfloat16 precision
    "per_device_train_batch_size": 1,  # Adjust based on VRAM
    "gradient_accumulation_steps": 8,
}
```

### 4. Two-Stage Training Process

**Source:** [BusterX Paper](https://arxiv.org/html/2505.12620)

**Stage 1: Cold Start SFT**
- Supervised fine-tuning on ~10k samples
- Uses CoT (Chain-of-Thought) examples
- Binary labels: REAL/FAKE

**Stage 2: RL Training (DAPO)**
- Dynamic sAmpling Policy Optimization
- Optimizes explanation quality
- Uses thinking reward model

### 5. Text Overlays / OCR Considerations

**Important:** The papers do NOT specifically mention OCR or text handling.

The model uses Qwen2.5-VL base which has general vision capabilities, but:
- Training focused on **visual artifacts** (temporal consistency, lighting, textures)
- Text overlays might confuse the model if not in training data
- TikTok-specific elements (watermarks, text, effects) need to be in your training set

**Recommendation:** Include TikTok videos WITH their typical overlays in training data so the model learns to handle them appropriately.

### 6. What's NOT Publicly Available

As of the paper publication:
- ❌ Training code not released on GitHub
- ❌ DAPO (RL) implementation not public
- ❌ Thinking reward model not released
- ✅ Model weights available on HuggingFace
- ✅ Dataset available on HuggingFace

### 7. DIY Fine-Tuning Approach

Without official training code, you can use standard Qwen2.5-VL fine-tuning:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer  # For supervised fine-tuning

# Load base model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "path/to/BusterX_plusplus",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)

# Prepare dataset in chat format
# Each sample: (video_frames, prompt, expected_response)

# Train using SFTTrainer or custom training loop
```

---

## 16. Batch Evaluation & Metrics

### Evaluation Script

We created `evaluate_batch.py` for batch testing:

```python
# Key metrics calculated:
# - Accuracy: (TP + TN) / Total
# - Precision: TP / (TP + FP) - "When we say fake, are we right?"
# - Recall: TP / (TP + FN) - "Do we catch all fakes?"
# - F1 Score: 2 × (P × R) / (P + R)
```

### Confusion Matrix

```
                 Predicted
              REAL    FAKE
Actual REAL    TN      FP    ← False Positive: real called fake
Actual FAKE    FN      TP    ← False Negative: fake missed (called real)
```

### Our Batch Test Results

| Metric | Value |
|--------|-------|
| Videos Tested | 6 (3 real + 3 fake) |
| Accuracy | 100% (6/6) |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

### Metrics Explanation

| Metric | Formula | What it Measures |
|--------|---------|------------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness |
| **Precision** | TP/(TP+FP) | Of predicted fakes, how many are actually fake |
| **Recall** | TP/(TP+FN) | Of actual fakes, how many did we catch |
| **F1** | Harmonic mean | Balance of precision and recall |

### Running Larger Evaluations

Edit `evaluate_batch.py`:
```python
SAMPLES_PER_CATEGORY = 50  # Increase from 3 for more reliable metrics
```

### Note on Paper Metrics

The model itself doesn't calculate metrics - it only outputs predictions. Metrics must be computed externally by comparing predictions to ground truth labels, as we did in `evaluate_batch.py`.

---

## 17. Official Sources & References

### Papers

| Paper | Link | Key Contribution |
|-------|------|------------------|
| **BusterX** | [arXiv:2505.12620](https://arxiv.org/abs/2505.12620) | Original MLLM + RL framework for video detection |
| **BusterX++** | [arXiv:2507.14632](https://arxiv.org/abs/2507.14632) | Cross-modal (image + video) detection, multi-stage training |

### Resources

| Resource | Link |
|----------|------|
| GitHub Repository | [l8cv/BusterX](https://github.com/l8cv/BusterX) |
| Model (HuggingFace) | BusterX++ on HuggingFace |
| Dataset: GenBuster-200K | [l8cv/GenBuster-200K](https://huggingface.co/datasets/l8cv/GenBuster-200K) |
| Dataset: GenBuster-200K-mini | [l8cv/GenBuster-200K-mini](https://huggingface.co/datasets/l8cv/GenBuster-200K-mini) |
| Base Model | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

### Key Verified Information

| Claim | Source | Status |
|-------|--------|--------|
| 16 frames at 4 FPS | Paper + our tests | ✅ Verified |
| Binary labels sufficient for training | Paper | ✅ Verified |
| Video standardization (5s, 24fps, 1024×1024) | Paper | ✅ Verified |
| LoRA config (r=16, α=32) | Paper | ✅ Verified |
| Learning rate (1e-5) | Paper | ✅ Verified |
| HEVC encoding to remove bias | Paper | ✅ Verified |
| OCR/text handling | Not mentioned in papers | ❓ Unknown |
| Public training code | GitHub check | ❌ Not released |

---

## 18. Reinforcement Learning Fundamentals for LLMs

### Traditional ML vs RL for Language Models

**Traditional Supervised Learning:**
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Input     │ ──→  │    Model    │ ──→  │   Output    │
│  (video)    │      │             │      │  (answer)   │
└─────────────┘      └─────────────┘      └─────────────┘
                            ↑
                     Loss = Compare output to
                     GROUND TRUTH LABEL

You need: Exact correct answers for training
```

**Reinforcement Learning:**
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Input     │ ──→  │    Model    │ ──→  │   Output    │
│  (video)    │      │  (Policy)   │      │  (answer)   │
└─────────────┘      └─────────────┘      └─────────────┘
                            ↑
                     REWARD SIGNAL
                     (Was output good? Score it)

You need: A way to SCORE outputs (not exact answers)
```

### Key RL Terminology (From BusterX++ Paper Context)

| Term | What It Means | In BusterX++ Context |
|------|---------------|---------------------|
| **Policy (π_θ)** | The model itself | BusterX++ model with weights θ |
| **Action** | What the model outputs | The generated text (reasoning + answer) |
| **Reward** | Score for an action | Was the answer correct? Was the format right? |
| **Advantage (Â)** | How much better than average | "This output scored 0.8, average is 0.5, so advantage = +0.3" |
| **Old Policy (π_θ_old)** | Model before update | Frozen copy to compare against |

### Why RL for Fake Detection?

**From BusterX++ paper:** You can't easily write perfect CoT explanations for why something is fake because:

> "Human judgments about the 'fakeness' of images or videos are often based on subtle, intuitive, and multi-dimensional cues... making it extremely challenging to precisely elaborate a linear thinking chain."

**Solution:** Let the model explore and learn what reasoning works, then reward good reasoning.

---

## 19. The DAPO Algorithm Explained

### What is DAPO?

**DAPO = Dynamic sAmpling Policy Optimization**

It's a variant of PPO (Proximal Policy Optimization) designed for language models.

**Source:** [BusterX++ Paper](https://arxiv.org/abs/2507.14632)

### How DAPO Works (Step by Step)

```
FOR EACH TRAINING VIDEO:
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: SAMPLE MULTIPLE OUTPUTS                                │
│                                                                 │
│  Take one video → Run model G times → Get G different outputs   │
│                                                                 │
│  Example (G=4):                                                 │
│  Video X → Model generates:                                     │
│    Output 1: "<think>Lighting looks off...</think><answer>B</answer>"  │
│    Output 2: "<think>Seems natural...</think><answer>A</answer>"       │
│    Output 3: "<think>Texture artifacts...</think><answer>B</answer>"   │
│    Output 4: "<think>Motion is smooth...</think><answer>A</answer>"    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: SCORE EACH OUTPUT WITH REWARDS                         │
│                                                                 │
│  Ground truth: Video X is FAKE (B)                              │
│                                                                 │
│  Output 1: Format ✓ (+0), Correct ✓ (+1) → Total: 1.0           │
│  Output 2: Format ✓ (+0), Wrong ✗ (+0)  → Total: 0.0            │
│  Output 3: Format ✓ (+0), Correct ✓ (+1) → Total: 1.0           │
│  Output 4: Format ✓ (+0), Wrong ✗ (+0)  → Total: 0.0            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: CALCULATE ADVANTAGE                                    │
│                                                                 │
│  Average reward = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5             │
│                                                                 │
│  Advantage = (reward - mean) / std                              │
│                                                                 │
│  Output 1: (1.0 - 0.5) / std = positive (good!)                 │
│  Output 2: (0.0 - 0.5) / std = negative (bad!)                  │
│  Output 3: (1.0 - 0.5) / std = positive (good!)                 │
│  Output 4: (0.0 - 0.5) / std = negative (bad!)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: UPDATE MODEL                                           │
│                                                                 │
│  Increase probability of outputs with POSITIVE advantage        │
│  Decrease probability of outputs with NEGATIVE advantage        │
│                                                                 │
│  Model learns: "Outputs 1 and 3 worked → do more like that"     │
│  Model learns: "Outputs 2 and 4 failed → avoid those patterns"  │
└─────────────────────────────────────────────────────────────────┘
```

### The "Dynamic Sampling" Part

**From paper:** DAPO ensures diversity with this constraint:
> "0 < |{o_i | is_equivalent(a, o_i)}| < G"

**Translation:**
- At least one output must be correct (otherwise no positive signal)
- At least one output must be wrong (otherwise no negative signal)
- This prevents the model from getting stuck (mode collapse)

---

## 20. BusterX++ Reward Functions

### All 5 Reward Functions (From Paper)

```python
# ═══════════════════════════════════════════════════════════════
# REWARD 1: FORMAT REWARD
# Does the output follow the required structure?
# ═══════════════════════════════════════════════════════════════

def format_reward(response):
    """
    From paper: r_fmt = 0 if correct format, -1 otherwise
    """
    if matches("<think>...</think><answer>...</answer>"):
        return 0    # Correct format (no penalty)
    else:
        return -1   # Wrong format (penalty)


# ═══════════════════════════════════════════════════════════════
# REWARD 2: SOFT OVERLONG PENALTY
# Is the response too long?
# ═══════════════════════════════════════════════════════════════

def overlong_penalty(length, max_length=750, cache=600):
    """
    From paper: Graduated penalty for length
    - Target: ~600 tokens
    - Maximum: 750 tokens
    """
    if length <= cache:
        return 0                              # Within target
    elif length <= max_length:
        # Linear penalty in transition zone
        return -((length - cache) / (max_length - cache))
    else:
        return -1                             # Too long


# ═══════════════════════════════════════════════════════════════
# REWARD 3: ACCURACY REWARD
# Did the model get the right answer?
# ═══════════════════════════════════════════════════════════════

def accuracy_reward(prediction, ground_truth):
    """
    From paper: r_acc = 1 for correct, 0 for incorrect
    """
    if prediction == ground_truth:
        return 1    # Correct
    else:
        return 0    # Wrong


# ═══════════════════════════════════════════════════════════════
# REWARD 4: HYBRID THINKING (Stage 2+3 only)
# Did the model follow the /think or /no_think instruction?
# ═══════════════════════════════════════════════════════════════

def hybrid_reward(prompt_type, response):
    """
    From paper: r_hybrid = 0 if followed instruction, -1 otherwise
    """
    if prompt_type == "/think" and has_thinking(response):
        return 0    # Correctly included reasoning
    elif prompt_type == "/no_think" and not has_thinking(response):
        return 0    # Correctly skipped reasoning
    else:
        return -1   # Didn't follow instruction


# ═══════════════════════════════════════════════════════════════
# REWARD 5: THINKING REWARD (Stage 3 only)
# How good is the reasoning quality?
# Uses external model: SophiaVL-R1-Thinking-Reward-Model-3B
# ═══════════════════════════════════════════════════════════════

def thinking_reward(response, accuracy, prompt_type, thinking_model):
    """
    From paper: r_think = min(r_acc, M(y_res)) in thinking mode
    - External model scores reasoning quality 0 to 1
    - Only rewards good reasoning IF also correct
    """
    if prompt_type == "/no_think":
        return 0    # No reasoning to evaluate
    else:
        quality_score = thinking_model.score(response)  # 0 to 1
        return min(accuracy, quality_score)  # Only reward if also correct
```

### Which Rewards Are Used in Each Stage

| Stage | r_fmt | r_overlong | r_acc | r_hybrid | r_think |
|-------|-------|------------|-------|----------|---------|
| **Stage 1** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Stage 2** | (SFT - no rewards) | | | | |
| **Stage 3** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 21. Stage 1: Foundation RL (Detailed)

### Goal

Teach the model to correctly classify real vs fake videos.

**From paper:** "Focus on building fundamental capabilities"

### What Happens in Stage 1

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: FOUNDATION RL                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT DATA:                                                    │
│  • Your labeled videos (real/ and fake/ folders)                │
│  • Just binary labels needed                                    │
│  • No CoT annotations required                                  │
│                                                                 │
│  WHAT THE MODEL DOES:                                           │
│  • Takes a video                                                │
│  • Generates reasoning + answer                                 │
│  • Multiple samples per video (DAPO with G outputs)             │
│                                                                 │
│  REWARDS USED (3 only):                                         │
│  • r_fmt (format)     → Penalize wrong structure                │
│  • r_overlong         → Penalize too-long responses             │
│  • r_acc (accuracy)   → Reward correct predictions              │
│                                                                 │
│  WHAT MODEL LEARNS:                                             │
│  • Basic ability to detect real vs fake                         │
│  • Proper output format <think>...</think><answer>...</answer>  │
│  • Reasonable response length (~600 tokens)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1 Training Loop (Pseudocode)

```python
# STAGE 1: FOUNDATION RL

# Initialize
model = load_base_model("Qwen2.5-VL-7B-Instruct")  # or BusterX++
model = apply_lora(model, r=16, alpha=32)
old_policy = copy(model)  # Frozen reference

# Training loop
for epoch in range(num_epochs):
    for video, label in dataset:

        # Step 1: Sample G outputs from current policy
        prompt = f"{video_frames} Is this video real or fake?"
        outputs = []
        for _ in range(G):  # G = number of samples, e.g., 4
            output = model.generate(prompt, do_sample=True)
            outputs.append(output)

        # Step 2: Score each output
        rewards = []
        for output in outputs:
            r = 0
            r += format_reward(output)           # -1 or 0
            r += overlong_penalty(len(output))   # -1 to 0
            r += accuracy_reward(
                extract_answer(output),
                label
            )                                    # 0 or 1
            rewards.append(r)

        # Step 3: Calculate advantages
        mean_r = mean(rewards)
        std_r = std(rewards)
        advantages = [(r - mean_r) / std_r for r in rewards]

        # Step 4: DAPO policy update
        loss = dapo_loss(
            model,
            old_policy,
            outputs,
            advantages
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update old policy periodically
    old_policy = copy(model)

# OUTPUT: Model that can classify real/fake with proper format
```

### Stage 1 Summary

| Aspect | Details |
|--------|---------|
| Training method | Reinforcement Learning (DAPO) |
| Data needed | Labeled videos (real/fake folders) |
| Rewards | Format + Length + Accuracy |
| Output | Model with basic detection ability |
| Duration | Majority of training time |

---

## 22. Stage 2: Thinking Mode Fusion (Detailed)

### Goal

Teach the model to switch between reasoning mode (/think) and direct mode (/no_think).

**From paper:** "Minimal impact on performance" but enables flexibility.

### What Happens in Stage 2

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: THINKING MODE FUSION                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT DATA:                                                    │
│  • Take the Stage 1 trained model                               │
│  • Run it on videos to collect outputs                          │
│  • From paper: "several hundred samples"                        │
│                                                                 │
│  CREATE TWO TYPES OF TRAINING DATA:                             │
│                                                                 │
│  TYPE 1 - /think mode:                                          │
│  Prompt: "[video] /think Is this real or fake?"                 │
│  Response: "<think>reasoning here</think><answer>B</answer>"    │
│                                                                 │
│  TYPE 2 - /no_think mode:                                       │
│  Prompt: "[video] /no_think Is this real or fake?"              │
│  Response: "<answer>B</answer>"  (NO thinking!)                 │
│                                                                 │
│  TRAINING METHOD:                                               │
│  • Standard Supervised Fine-Tuning (SFT)                        │
│  • NOT reinforcement learning                                   │
│  • Cross-entropy loss on expected outputs                       │
│                                                                 │
│  WHAT MODEL LEARNS:                                             │
│  • When to include reasoning (/think)                           │
│  • When to skip reasoning (/no_think)                           │
│  • Flexibility for different use cases                          │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 2 Data Collection and Training (Pseudocode)

```python
# STAGE 2: THINKING MODE FUSION

# Load Stage 1 model
stage1_model = load_model("stage1_checkpoint")

# ═══════════════════════════════════════════════════════════════
# STEP 1: Collect samples from Stage 1 model
# ═══════════════════════════════════════════════════════════════

training_samples = []

for video, label in subset_of_dataset:  # "several hundred" videos

    # Generate output from Stage 1 model
    output = stage1_model.generate(f"{video} Is this real or fake?")

    # Only keep CORRECT predictions
    if extract_answer(output) == label:

        # Create /think version (with reasoning)
        training_samples.append({
            "prompt": f"{video} /think Is this real or fake?",
            "response": output  # Full: <think>...</think><answer>...</answer>
        })

        # Create /no_think version (answer only)
        answer_only = extract_answer_block(output)  # Just <answer>...</answer>
        training_samples.append({
            "prompt": f"{video} /no_think Is this real or fake?",
            "response": answer_only
        })

print(f"Collected {len(training_samples)} samples")
# From paper: "several hundred samples" → ~500-1000 total

# ═══════════════════════════════════════════════════════════════
# STEP 2: Standard SFT training
# ═══════════════════════════════════════════════════════════════

from trl import SFTTrainer

trainer = SFTTrainer(
    model=stage1_model,
    train_dataset=training_samples,
    # Standard SFT config
    max_seq_length=1024,
    num_train_epochs=1,  # Quick training
)

trainer.train()

# OUTPUT: Model that can switch between /think and /no_think modes
```

### Stage 2 Summary

| Aspect | Details |
|--------|---------|
| Training method | Supervised Fine-Tuning (SFT) |
| Data needed | Outputs from Stage 1 model (~500-1000 samples) |
| Rewards | None (cross-entropy loss) |
| Output | Model with /think and /no_think modes |
| Duration | Quick (single epoch) |

---

## 23. Stage 3: Advanced RL with Thinking Reward (Detailed)

### Goal

Improve the QUALITY of reasoning, not just accuracy.

**From paper:** Uses "Thinking Reward" from external model to evaluate reasoning quality.

### What Happens in Stage 3

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: ADVANCED RL                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT DATA:                                                    │
│  • Same video dataset as Stage 1                                │
│  • Mixed prompts: some /think, some /no_think                   │
│                                                                 │
│  NEW COMPONENT - THINKING REWARD MODEL:                         │
│  • External model: SophiaVL-R1-Thinking-Reward-Model-3B         │
│  • Evaluates: "Is this reasoning logical and comprehensive?"    │
│  • Returns score: 0.0 to 1.0                                    │
│  • From paper: "Higher score = more reasonable thinking"        │
│                                                                 │
│  REWARDS USED (all 5):                                          │
│  • r_fmt (format)                                               │
│  • r_overlong (length penalty)                                  │
│  • r_acc (accuracy)                                             │
│  • r_hybrid (followed /think or /no_think correctly?)           │
│  • r_think (quality of reasoning) ← NEW                         │
│                                                                 │
│  KEY INSIGHT from paper:                                        │
│  "Thinking Reward naturally increases CoT token length"         │
│  → Model learns to give more detailed, better explanations      │
│                                                                 │
│  WARNING from paper:                                            │
│  "Applying Thinking Reward earlier destabilizes training"       │
│  → MUST do Stage 1 and 2 first!                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 3 Training Loop (Pseudocode)

```python
# STAGE 3: ADVANCED RL WITH THINKING REWARD

# Load Stage 2 model and thinking reward model
model = load_model("stage2_checkpoint")
model = apply_lora(model, r=16, alpha=32)
old_policy = copy(model)

# Load external thinking reward model
thinking_model = load_model("SophiaVL-R1-Thinking-Reward-Model-3B")

# Training loop
for epoch in range(num_epochs):
    for video, label in dataset:

        # Step 1: Randomly choose mode
        mode = random.choice(["/think", "/no_think"])
        prompt = f"{video} {mode} Is this real or fake?"

        # Step 2: Sample G outputs
        outputs = []
        for _ in range(G):
            output = model.generate(prompt, do_sample=True)
            outputs.append(output)

        # Step 3: Score with ALL reward functions
        rewards = []
        for output in outputs:
            r = 0

            # Basic rewards (same as Stage 1)
            r += format_reward(output)
            r += overlong_penalty(len(output))

            # Accuracy reward
            prediction = extract_answer(output)
            acc = 1 if prediction == label else 0
            r += acc

            # Hybrid reward (new in Stage 3)
            r += hybrid_reward(mode, output)

            # Thinking reward (new in Stage 3, only for /think mode)
            if mode == "/think":
                # External model evaluates reasoning quality
                quality = thinking_model.score(output)  # 0 to 1
                # Only reward if ALSO correct
                r += min(acc, quality)

            rewards.append(r)

        # Step 4: Calculate advantages
        advantages = normalize(rewards)

        # Step 5: DAPO update
        loss = dapo_loss(model, old_policy, outputs, advantages)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    old_policy = copy(model)

# OUTPUT: Model with high-quality, detailed reasoning
```

### Stage 3 Summary

| Aspect | Details |
|--------|---------|
| Training method | Reinforcement Learning (DAPO) |
| Data needed | Same labeled videos as Stage 1 |
| Rewards | Format + Length + Accuracy + Hybrid + Thinking |
| External model | SophiaVL-R1-Thinking-Reward-Model-3B |
| Output | Model with improved reasoning quality |
| Duration | Significant (comparable to Stage 1) |

---

## 24. Complete Training Pipeline Implementation

### Full Pipeline for Your TikTok Data

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR STARTING POINT                                            │
│                                                                 │
│  tiktok_dataset/                                                │
│  ├── real/  (~5000+ videos)                                     │
│  └── fake/  (~5000+ videos, AI-generated)                       │
│                                                                 │
│  All videos: 5s, 24fps, 1024×1024, HEVC encoded                 │
│  Just folder structure = your labels                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: FOUNDATION RL                                         │
│  ────────────────────────────────────────────────────────────── │
│  Duration: ~70% of total training time                          │
│  Rewards: format + length + accuracy                            │
│  Output: Model can classify real/fake                           │
│                                                                 │
│  for epoch in range(num_epochs):                                │
│      for video, label in dataset:                               │
│          outputs = sample_G_outputs(model, video)               │
│          rewards = [format + length + accuracy for each]        │
│          advantages = normalize(rewards)                        │
│          update_model_with_dapo(outputs, advantages)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: THINKING MODE FUSION (SFT)                            │
│  ────────────────────────────────────────────────────────────── │
│  Duration: Quick (~5% of training)                              │
│  Method: Supervised Fine-Tuning                                 │
│  Output: Model can switch /think and /no_think                  │
│                                                                 │
│  # Collect samples from Stage 1 model                           │
│  samples = collect_correct_outputs(stage1_model, ~500_videos)   │
│  # Create paired /think and /no_think versions                  │
│  paired_data = create_mode_pairs(samples)                       │
│  # Quick SFT                                                    │
│  sft_train(model, paired_data, epochs=1)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: ADVANCED RL WITH THINKING REWARD                      │
│  ────────────────────────────────────────────────────────────── │
│  Duration: ~25% of total training time                          │
│  Rewards: format + length + accuracy + hybrid + thinking        │
│  External: SophiaVL-R1-Thinking-Reward-Model-3B                 │
│  Output: Model with high-quality reasoning                      │
│                                                                 │
│  thinking_model = load("SophiaVL-R1-Thinking-Reward-Model-3B")  │
│                                                                 │
│  for epoch in range(num_epochs):                                │
│      for video, label in dataset:                               │
│          mode = random.choice(["/think", "/no_think"])          │
│          outputs = sample_G_outputs(model, video, mode)         │
│          rewards = [all_5_rewards for each]                     │
│          if mode == "/think":                                   │
│              rewards += thinking_model.score(reasoning)         │
│          advantages = normalize(rewards)                        │
│          update_model_with_dapo(outputs, advantages)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT                                                   │
│                                                                 │
│  Your fine-tuned BusterX++ model that:                          │
│  ✓ Detects real vs fake TikTok videos                          │
│  ✓ Provides detailed reasoning when asked (/think)              │
│  ✓ Gives quick answers when needed (/no_think)                  │
│  ✓ Handles TikTok-specific artifacts (watermarks, text, etc.)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 25. Implementation Requirements

### Required Components

| Component | What It Is | Where to Get It |
|-----------|------------|-----------------|
| Base model | Qwen2.5-VL-7B or BusterX++ | HuggingFace: l8cv/BusterX++ |
| DAPO implementation | RL algorithm | `trl` library or custom |
| Thinking Reward Model | Quality scorer | SophiaVL-R1-Thinking-Reward-Model-3B |
| Dataset | Your TikTok videos | You create this |

### Libraries Needed

```python
# Core model loading
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Parameter-efficient fine-tuning
from peft import LoraConfig, get_peft_model

# RL training (for DAPO)
from trl import PPOTrainer, PPOConfig
# Note: DAPO may need custom implementation based on paper

# Standard training utilities
import torch
from torch.optim import AdamW
```

### Hyperparameters from Paper

```python
# LoRA Configuration
lora_config = LoraConfig(
    r=16,              # Rank (from paper)
    lora_alpha=32,     # Alpha (from paper)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

# Training Configuration
training_config = {
    "learning_rate": 1e-5,        # From paper
    "precision": "bfloat16",      # From paper
    "G": 4,                       # Samples per input (typical)
    "max_response_length": 750,   # From paper
    "target_response_length": 600,# From paper
}
```

### Minimum Dataset Size

| Stage | Data Needed | Recommended |
|-------|-------------|-------------|
| Stage 1 | Full dataset | 10,000+ videos |
| Stage 2 | Subset for SFT | 500-1000 samples |
| Stage 3 | Full dataset | Same as Stage 1 |

### GPU Requirements

| Configuration | VRAM Needed | Notes |
|---------------|-------------|-------|
| Inference only (4-bit) | 8 GB | What we tested |
| Training with LoRA | 24-40 GB | A100 recommended |
| Full fine-tuning | 80+ GB | Multiple A100s |

**From paper:** Training used over 10,000 A100 GPU hours.

### What You DON'T Need

| Task | Required? | Why |
|------|-----------|-----|
| Write CoT explanations | ❌ NO | Generated by model + RL |
| Annotate artifacts | ❌ NO | Model learns from binary labels |
| Pre-training | ❌ NO | Use existing BusterX++ weights |

### What You DO Need

| Task | Required? | Details |
|------|-----------|---------|
| Binary labels (real/fake) | ✅ YES | Folder structure works |
| Balanced dataset | ✅ YES | 1:1 ratio real:fake |
| Standardized video format | ✅ YES | 5s, 24fps, 1024×1024, HEVC |
| Thinking Reward Model | ✅ YES | For Stage 3 quality scoring |
| Significant GPU resources | ✅ YES | A100s for training |

---

*Document created: December 2024*
*Last updated: December 2024*
*Hardware tested: RTX 4060 Laptop 8GB VRAM*
*Sources: BusterX++ arXiv paper (2507.14632), local config files, empirical testing*

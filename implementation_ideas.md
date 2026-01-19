# Implementation Ideas for Deepfake Video Detection

## Original Question

> According to the paper and the tests we've ran so far and the MD file, what is the best approach for me to reproduce this paper step by step on my dataset? It's not necessarily about videos from tik tok but supposing I want to identify deepfake videos of any kind (political propaganda, voice over video which is fake, text over video with fake voice and so on - these are just some examples) how would you set up a plan for this? What should I do step by step? What type of datasets shall I create (exactly taking as example what the authors did), how shall I preprocess this data, do I have to respect a 5s video if I want to use their model? Can I change something along the way? What is the next step after I have the data and the next one and so on? But I need specific information about every step according to the paper. Do you think the MISLnet architecture with that constrained layer can be useful on top of what we have for better accuracy? (And for this I would like you to look for the following paper: "Beyond Deepfake Images: Detecting AI-Generated Videos" from Drexel University, the authors being Danial Samadi Vahdati, Tai D. Nguyen, Aref Azizpour, Matthew C. Stamm). I have also found some repositories I could give you the links for so we can take a look together. Please do not mix the papers and the approaches, I'm trying to see what would be the best option for me right now, no matter the computer, but obtaining very good results. Try to look for what others experimented with and their results, only valid, scientific sources.

---

## Comprehensive Analysis: Reproducing BusterX++ and Alternative Approaches

### Overview of the Two Main Approaches

| Aspect | **BusterX/BusterX++** | **MISLnet (Drexel)** |
|--------|----------------------|----------------------|
| **Core Approach** | MLLM + Reinforcement Learning | Constrained CNN (sub-pixel forensics) |
| **Reported Accuracy** | 88.3-93.9% | 98.3% |
| **Explainability** | Yes (provides reasoning) | No (binary classification) |
| **Training Data Required** | Binary labels only | Binary labels only |
| **Zero-shot Transfer** | Moderate (needs few-shot for new generators) | Poor (0.53 AUC), but excellent with few-shot (0.98) |
| **Compression Resistance** | Good with HEVC standardization | Excellent with robust training |

**Important caveat**: These accuracies are from **different benchmarks** and are not directly comparable.

---

## Part 1: Reproducing BusterX++ Step-by-Step

### Step 1: Dataset Creation (Following Authors' Approach)

Based on [BusterX Paper (arXiv:2505.12620)](https://arxiv.org/abs/2505.12620):

#### 1.1 Real Videos Collection
```
Sources used by authors:
├── OpenVid-1M dataset (100K+ videos)
└── Your equivalent: YouTube, Pexels, stock footage, surveillance
```

**Your approach for political/deepfake videos:**
```
real_dataset/
├── political_speeches/     # Real politicians speaking
├── news_broadcasts/        # Authentic news footage
├── public_appearances/     # Verified real footage
├── voice_recordings/       # Real voice-over-video
└── text_overlays/          # Real infographics/news tickers
```

#### 1.2 Fake Videos Collection
```
Authors used:
├── Open-source: HunyuanVideo, LTX-Video
└── Commercial: Sora, Jimeng (via APIs)

For YOUR use case (political deepfakes):
├── face_swap/              # FaceSwap, DeepFaceLab
├── lip_sync/               # Wav2Lip, SadTalker
├── voice_clone/            # ElevenLabs, XTTS cloned audio
├── full_synthetic/         # Sora, Runway, Pika, Kling
└── text_to_video/          # Generated political content
```

#### 1.3 Required Balance
From paper: **1:1 positive-negative ratio** is required for training.

### Step 2: Video Preprocessing (MANDATORY Specifications)

From [BusterX Paper](https://arxiv.org/html/2505.12620):

| Parameter | **Required Value** | **Can You Change It?** |
|-----------|-------------------|------------------------|
| Resolution | 1024×1024 | Must standardize (can use different resolution if you retrain) |
| Duration | 5 seconds | Model expects this; longer videos need clipping |
| Frame Rate | 24 FPS | Should standardize to remove encoding bias |
| Encoding | HEVC x265, yuv420p10le | Critical - removes codec fingerprints |
| Frame Sampling | 16 frames at 4 FPS | Fixed in model architecture |

**Preprocessing Script (from your knowledge file):**
```python
import subprocess

def standardize_video(input_path, output_path):
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

#### Why HEVC Encoding Matters
> "This standardization eliminates potential biases from underlying encoding preferences." — [BusterX Paper](https://arxiv.org/html/2505.12620)

Different AI generators use different encodings. Without standardization, the model learns to detect H.264 vs HEVC instead of actual deepfake artifacts.

### Step 3: Training Pipeline (3 Stages)

From [BusterX++ Paper (arXiv:2507.14632)](https://arxiv.org/abs/2507.14632):

#### Stage 1: Foundation RL (~70% of training)
```
Input: Your labeled videos (folder structure = labels)
Method: DAPO (Dynamic sAmpling Policy Optimization)
Rewards:
  - Format reward (correct <think>...</think><answer>...</answer>)
  - Length penalty (target ~600 tokens, max 750)
  - Accuracy reward (+1 for correct classification)
Output: Model can classify real/fake
```

#### Stage 2: Thinking Mode Fusion (~5% of training)
```
Input: Outputs from Stage 1 model (~500-1000 samples)
Method: Supervised Fine-Tuning (NOT RL)
Purpose: Teach model to switch between /think and /no_think modes
Output: Model can provide reasoning or quick answers
```

#### Stage 3: Advanced RL (~25% of training)
```
Input: Same videos as Stage 1
Method: DAPO with additional rewards
New Components:
  - Hybrid reward (did it follow /think or /no_think?)
  - Thinking reward from external model: SophiaVL-R1-Thinking-Reward-Model-3B
Output: High-quality reasoning explanations
```

### Step 4: Hyperparameters (From Paper)

```python
# LoRA Configuration
lora_config = {
    "r": 16,              # Rank
    "lora_alpha": 32,     # Alpha
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05
}

# Training Configuration
training_config = {
    "learning_rate": 1e-5,
    "precision": "bfloat16",
    "G": 4,                       # Samples per input for DAPO
    "max_response_length": 750,
    "target_response_length": 600
}
```

### Step 5: Hardware Requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| Inference (4-bit) | 8 GB | RTX 4060 works |
| Training with LoRA | 24-40 GB | A100 or 3090/4090 |
| Full training | 80+ GB | Multiple A100s |

**From paper:** Over **10,000 A100 GPU hours** for full training.

---

## Part 2: MISLnet Constrained Layer Analysis

From [Beyond Deepfake Images (CVPR 2024)](https://arxiv.org/html/2404.15955v1) and [Bayar & Stamm 2018](https://ieeexplore.ieee.org/document/8335799/):

### What is the Constrained Convolutional Layer?

The constrained layer is the **first layer** of the CNN with these properties:

1. **Mathematical Constraint**: The filter weights are constrained so that the center weight = -1 and all other weights sum to 1
2. **Effect**: This forces the network to learn **prediction residuals** (difference between a pixel and its neighbors)
3. **Purpose**: Suppresses image content, reveals manipulation artifacts

```
Standard CNN first layer:
  [learns edges, colors, textures of the IMAGE CONTENT]

Constrained CNN first layer:
  [learns FORENSIC TRACES - artifacts left by manipulation]
```

### MISLnet Performance (from Drexel paper)

| Task | AUC |
|------|-----|
| Patch-level detection | 0.983 |
| Source attribution | 0.991 |
| Zero-shot to new generators | 0.530-0.939 (poor) |
| After few-shot learning (< 1 min video) | 0.982 |
| H.264 compressed (CRF 40) | 0.984 |

### Key Finding
> "Synthetic image detectors are unable to detect synthetic videos... synthetic video generators introduce substantially different traces than those left by image generators." — [Drexel Paper](https://arxiv.org/html/2404.15955v1)

---

## Part 3: Can MISLnet + BusterX++ Be Combined?

### Hybrid Approaches in Literature

Recent research shows promising results with hybrid approaches:

| Framework | Approach | Source |
|-----------|----------|--------|
| **TruthLens** | MLLM global semantics + localized forensic cues | [arXiv 2025](https://arxiv.org/html/2503.15867) |
| **X²-DFD** | MLLM + Specific Feature Detectors (SFDs) | [arXiv 2024](https://arxiv.org/html/2410.06126) |
| **VLF-FFD** | Vision-Language Fusion for bidirectional feature interaction | [arXiv 2025](https://arxiv.org/html/2505.02013) |
| **CAD** | Cross-Modal Alignment + Distillation | [arXiv 2025](https://arxiv.org/html/2505.15233v1) |

### Assessment: Would MISLnet Help BusterX++?

**Potentially YES, but with caveats:**

| Aspect | Benefit | Challenge |
|--------|---------|-----------|
| **Sub-pixel forensics** | MISLnet detects artifacts BusterX might miss | Different input pipelines |
| **Complementary features** | MLLM = semantic reasoning, MISLnet = low-level traces | Integration complexity |
| **Compression robustness** | MISLnet handles H.264 well with robust training | BusterX uses HEVC standardization |
| **Explainability** | BusterX explains, MISLnet scores | Could combine confidence scores |

### Proposed Hybrid Architecture

```
                    ┌─────────────────────────┐
                    │     Input Video         │
                    │  (standardized 5s clip) │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│   MISLnet Branch          │       │   BusterX++ Branch        │
│   (Constrained CNN)       │       │   (MLLM + RL)             │
│                           │       │                           │
│   • Sub-pixel forensics   │       │   • 16 frames @ 4 FPS     │
│   • Patch-level analysis  │       │   • Semantic reasoning    │
│   • Forensic traces       │       │   • Chain-of-thought      │
│                           │       │                           │
│   Output: Score [0-1]     │       │   Output: Reasoning +     │
│                           │       │           Answer          │
└───────────────┬───────────┘       └───────────────┬───────────┘
                │                                   │
                └───────────────┬───────────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │   Fusion Module           │
                    │   • Weighted ensemble     │
                    │   • Confidence calibration│
                    │   • Final: Real/Fake +    │
                    │     Explanation + Score   │
                    └───────────────────────────┘
```

---

## Part 4: What Can You Change?

### Video Duration

| Duration | Implication |
|----------|-------------|
| **< 5 seconds** | Pad or loop to 5s, or modify frame sampling |
| **= 5 seconds** | Ideal (as paper specifies) |
| **> 5 seconds** | Split into 5s clips OR sample 16 frames across entire video |

**If you have longer videos**: You could modify the frame sampling to sample 16 frames across the entire duration (e.g., for a 30s video, sample every ~2 seconds instead of every 0.25s).

### Resolution

| If you change resolution... | Impact |
|-----------------------------|--------|
| Lower (384×384) | Works for inference on 8GB VRAM (tested) |
| Higher (1024×1024) | Better accuracy, needs more VRAM |
| Different aspect ratio | Needs padding/cropping |

**For retraining**: You can use a different resolution, but must be consistent across all training data.

### What You CANNOT Change

- Frame count (16 frames) — hardcoded in visual encoder
- Model architecture (Qwen2.5-VL-7B base)
- Token limit for reasoning (~750 tokens)

---

## Part 5: Recommended Approach for Your Use Case

Given your goals (political propaganda, voice deepfakes, text overlays):

### Option A: Fine-tune BusterX++ (Recommended)

**Pros:**
- Explainability (important for political content analysis)
- State-of-the-art on diverse benchmarks
- Pre-trained weights available

**Cons:**
- May need retraining for your specific domain
- GPU-intensive training

### Option B: MISLnet with Few-Shot Learning

**Pros:**
- Only needs < 1 minute of video per new generator type
- 98%+ accuracy on known generators
- Compression-robust with proper training

**Cons:**
- No explainability
- Poor zero-shot transfer (need samples from each generator)

### Option C: Hybrid Approach (Most Robust)

1. Use MISLnet for initial fast screening (binary score)
2. Use BusterX++ for detailed analysis of flagged content
3. Combine scores with learned fusion weights

---

## Part 6: Step-by-Step Implementation Plan

### Phase 1: Dataset Collection
```
1. Collect 5,000+ real political videos
   - Verified speeches, news broadcasts, public footage

2. Generate/collect 5,000+ fake videos
   - Face-swapped content
   - Lip-sync manipulations
   - Voice-cloned audio over video
   - Fully synthetic (Sora, Runway, etc.)
   - Text overlay manipulations

3. Organize into folders:
   dataset/
   ├── train/
   │   ├── real/     (4,000 videos)
   │   └── fake/     (4,000 videos by type)
   ├── test/
   │   ├── real/     (500 videos)
   │   └── fake/     (500 videos)
   └── benchmark/
       ├── real/     (500 videos)
       └── fake/     (500 from UNSEEN generators)
```

### Phase 2: Preprocessing
```
For each video:
1. Extract 5-second clips
2. Resize to 1024×1024
3. Re-encode to 24 FPS, HEVC x265
4. Verify all clips pass quality check
```

### Phase 3: Baseline Evaluation
```
1. Run existing BusterX++ on your dataset (no training)
2. Record baseline accuracy
3. Identify failure cases
```

### Phase 4: Fine-tuning (if needed)
```
Stage 1: Foundation RL on your dataset
Stage 2: Collect /think and /no_think samples
Stage 3: Advanced RL with thinking reward
```

### Phase 5: Evaluation
```
1. Test on held-out test set (in-distribution)
2. Test on benchmark set (out-of-distribution)
3. Compare to baseline and other methods
```

---

## Sources

### Primary Papers
- [BusterX (arXiv:2505.12620)](https://arxiv.org/abs/2505.12620)
- [BusterX++ (arXiv:2507.14632)](https://arxiv.org/abs/2507.14632)
- [Beyond Deepfake Images - MISLnet (CVPR 2024)](https://arxiv.org/html/2404.15955v1)
- [Constrained CNNs - Bayar & Stamm 2018](https://ieeexplore.ieee.org/document/8335799/)

### Benchmarks
- [DeepfakeBench (GitHub)](https://github.com/SCLBD/DeepfakeBench) - 36 detection methods
- [Deepfake-Eval-2024 (arXiv)](https://arxiv.org/abs/2503.02857) - Real-world benchmark
- [GenVidBench (ICLR 2025)](https://arxiv.org/html/2501.11340v1)

### Hybrid Approaches
- [TruthLens (arXiv 2025)](https://arxiv.org/html/2503.15867)
- [X²-DFD Framework (arXiv 2024)](https://arxiv.org/html/2410.06126)
- [CAD Framework (arXiv 2025)](https://arxiv.org/html/2505.15233v1)

### Preprocessing Best Practices
- [Temporal Consistency Analysis (2024)](https://aimspress.com/article/doi/10.3934/era.2024119)
- [Spatial-Temporal Preprocessing (ACM 2024)](https://dl.acm.org/doi/fullHtml/10.1145/3639592.3639597)

---

## Additional Notes

### Key Metrics for Evaluation

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness |
| **Precision** | TP/(TP+FP) | Of predicted fakes, how many are actually fake |
| **Recall** | TP/(TP+FN) | Of actual fakes, how many did we catch |
| **F1** | Harmonic mean | Balance of precision and recall |
| **AUC** | Area Under ROC Curve | Performance across all thresholds |

### Frame Sampling Best Practices (from literature)

> "Extracting frames at 10-frame intervals was the most suitable. If the time gap between frames is too wide, it leads to a loss of temporal continuity, while if it's too narrow, there is no significant difference between frames." — [ACM 2024](https://dl.acm.org/doi/fullHtml/10.1145/3639592.3639597)

### Temporal Consistency Key Insight

> "The consecutive frames of an original video have natural consistency, but deepfake videos are composed of individual forged images linked together, which disrupts the original spatio-temporal consistency and introduces forgery traces in the temporal domain." — [ERA 2024](https://aimspress.com/article/doi/10.3934/era.2024119)

---

*Document created: December 2024*
*Purpose: Implementation planning for deepfake video detection system*

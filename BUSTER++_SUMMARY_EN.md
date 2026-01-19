# BusterX++ Paper Summary

**Paper:** BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM
**arXiv:** 2507.14632
**Purpose:** This document summarizes each section of the paper for team reference.

---

## 3.2 Benchmark Construction

### Overview

| Attribute | Value |
|-----------|-------|
| Total samples | 4,000 |
| Real images | 1,000 |
| Fake images | 1,000 |
| Real videos | 1,000 |
| Fake videos | 1,000 |

---

### Data Sources

#### Real Content
- **Source:** OpenVid-1M HD dataset
- **Content:** Diverse real-world scenarios
- **Pre-filtering:** Applied for scene variety

#### Fake Content
Two sources:

1. **MagicArena** - curated high-rated samples

2. **Custom Generation Pipeline:**
```
Reddit API (real images)
    → Qwen-2.5-VL (generates captions)
    → Captions as prompts
    → Diffusion models generate fakes
```

**Why Reddit?** Real social media images provide realistic scenarios. Captions describe authentic situations, making generated fakes more challenging to detect.

| Generator Type | Models Used |
|----------------|-------------|
| Images | FLUX, GPT-4o |
| Videos | Seedance 1.0, SkyReels V1 |

---

### Data Filtering

#### Real Samples (3 stages)
1. **Technical filter:** resolution, frame rate, bitrate
2. **Duplicate removal:** same origin clip
3. **Manual review:** remove watermarks, anime, synthetic backgrounds

#### Fake Samples (Novel 2-stage approach)
1. Mix real + fake in blind pool → experts identify "looks real" samples
2. Re-examine to confirm synthetic origin

**Purpose:** Keep only the most convincing fakes (challenging benchmark).

---

### Post-Processing Specifications

| Media | Parameter | Value |
|-------|-----------|-------|
| **Images** | Resolution | 1024 × 1024 |
| **Videos** | Resolution | 1920 × 1080 |
| | Duration | 5 seconds |
| | Frame rate | 24 FPS |
| | Codec | HEVC x265 |

**Why standardize?**
1. Eliminates encoding biases between different generators
2. Ensures consistency across all sources

#### FFmpeg Commands
```bash
# Video
ffmpeg -i input.mp4 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  -t 5 -r 24 -c:v libx265 -pix_fmt yuv420p10le output.mp4

# Image
ffmpeg -i input.jpg \
  -vf "scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:(ow-iw)/2:(oh-ih)/2" \
  output.jpg
```

---

### References Used in This Section

| Ref | Name | Purpose |
|-----|------|---------|
| [5] | Qwen-2.5-VL | Image captioning |
| [17] | Seedance 1.0 | Video generation |
| [29] | FLUX | Image generation |
| [33] | OpenVid-1M HD | Real data source |
| [36] | GPT-4o | Image generation |
| [43] | SkyReels V1 | Video generation |

---

## 4. Method

### How It Connects to Benchmark Construction

After preparing the dataset (Section 3.2), the next step is training. The flow is:

```
Benchmark (4,000 samples)
    → Post-processed (standardized resolution, duration, codec)
    → Fed into training pipeline
    → Model learns to classify Real vs Fake
```

---

## 4.1 Challenges of Cold Start

### What is Cold Start?

Most MLLM+RL methods use a two-phase approach:
1. **SFT Phase (Cold Start):** Supervised fine-tuning with Chain-of-Thought (CoT) examples
2. **RL Phase:** Reinforcement learning to improve

### The Problem

The authors argue this approach is **limited** because:

| Issue | Explanation |
|-------|-------------|
| **CoT quality bottleneck** | Human detection of fakes relies on subtle, intuitive cues (unnatural reflections, lighting inconsistencies, motion artifacts) |
| **Hard to generate good CoT** | Creating quality explanations via prompt engineering is extremely challenging |
| **Risk of degradation** | Poor CoT data can actually degrade the model's reasoning ability |

### BusterX++ Solution

**Abandon cold-start entirely.** Instead, use a multi-stage RL approach that builds capabilities progressively without requiring pre-made CoT data.

---

## 4.2 Multi-Stage Training

### Overview

BusterX++ uses **DAPO** (Dynamic sAmpling Policy Optimization) across three stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

    Standardized Dataset (from Section 3.2)
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   STAGE 1: Foundation RL          │  ~70% of training
    │   Learn basic classification      │
    │   Reward: format + length + acc   │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   STAGE 2: Thinking Mode Fusion   │  ~5% of training
    │   Learn to switch modes           │
    │   Method: Supervised Fine-Tuning  │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   STAGE 3: Advanced RL            │  ~25% of training
    │   Enhance reasoning quality       │
    │   Reward: + hybrid + thinking     │
    └───────────────────────────────────┘
                    │
                    ▼
           Final Model (BusterX++)
```

---

### What is DAPO?

DAPO (Dynamic sAmpling Policy Optimization) is an improved version of GRPO. For each input:

1. Sample multiple outputs from the model
2. Score each output with reward functions
3. Calculate advantage (how much better/worse than average)
4. Update model to favor higher-reward outputs

**Key improvements over GRPO:**
- Asymmetric clipping (better exploration)
- Token-level loss (better for long reasoning)
- Dynamic sampling (removes uninformative samples)

---

### Stage 1: Foundation RL

**Goal:** Learn basic real vs fake classification.

**What happens:**
- Model receives video/image
- Generates response with `<think>...</think><answer>...</answer>` format
- Receives rewards based on correctness

**Reward Function:**
```
R_stage-1 = r_fmt + r_overlong + r_acc
```

| Reward | Value | Condition |
|--------|-------|-----------|
| r_fmt | 0 | Correct format (`<think>...</think><answer>...</answer>`) |
| r_fmt | -1 | Wrong format |
| r_overlong | 0 to -1 | Graduated penalty if response exceeds max length |
| r_acc | +1 | Correct classification |
| r_acc | 0 | Wrong classification |

**Output:** Model can classify but reasoning quality is basic.

---

### Stage 2: Thinking Mode Fusion

**Goal:** Teach model to switch between detailed reasoning and quick answers.

**Method:** Supervised Fine-Tuning (NOT RL)

**Two modes:**

| Mode | Trigger | Output Format |
|------|---------|---------------|
| Thinking | `/think` or no instruction | `<think>{detailed reasoning}</think><answer>{response}</answer>` |
| Non-Thinking | `/no_think` | `<think></think><answer>{response}</answer>` |

**Why needed?**
- Sometimes you need detailed explanation (for reports, evidence)
- Sometimes you need fast classification (batch processing)
- ~0.7% accuracy drop in non-thinking mode, but much faster

**Note:** Ablation shows this stage has minimal impact on accuracy, but is necessary for Stage 3.

---

### Stage 3: Advanced RL

**Goal:** Improve reasoning quality using external evaluation.

**New components:**

#### 1. Thinking Reward
An external model evaluates reasoning quality:
- **Model:** SophiaVL-R1-Thinking-Reward-Model-3B
- **Score:** 0 ≤ r_think ≤ 1

```
r_think = {
    0,                      if /no_think mode
    min(r_acc, M(y_res)),   otherwise (M = external model)
}
```

**Why min(r_acc, ...)?** If classification is wrong, good reasoning doesn't help.

#### 2. Hybrid Reward
Ensures model respects mode instructions:

```
r_hybrid = {
    0,   if response follows correct mode
   -1,   if thinks when told not to, or skips thinking when should
}
```

**Total Reward:**
```
R_stage-3 = r_fmt + r_overlong + r_acc + r_hybrid + r_think
```

**Why Stage 3 is separate:** Applying thinking reward too early (in Stage 1) destabilizes training.

---

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-VL-7B-Instruct |
| External Reward Model | SophiaVL-R1-Thinking-Reward-Model-3B |
| Fine-tuning Method | LoRA (rank=16, alpha=32) |
| Learning Rate | 1×10⁻⁵ |
| Precision | bfloat16 |

---

### Performance Comparison

| Approach | After Stage 1 | After Stage 3 |
|----------|---------------|---------------|
| With Cold-Start | 71.7% | 72.9% |
| Without Cold-Start (BusterX++) | 69.4% | **77.4%** |

**Conclusion:** No cold-start performs worse initially but achieves better final results with superior generalization.

---

### Complete Flow: From Data to Trained Model

```
1. BENCHMARK CONSTRUCTION (Section 3.2)
   │
   ├── Collect real data (OpenVid-1M HD)
   ├── Generate fake data (FLUX, GPT-4o, Seedance, SkyReels)
   ├── Filter (quality + expert review)
   └── Standardize (resolution, duration, codec)
   │
   ▼
2. STAGE 1: FOUNDATION RL
   │
   ├── Input: Standardized images/videos with labels
   ├── Algorithm: DAPO
   ├── Rewards: Format + Length + Accuracy
   └── Output: Basic classifier with reasoning
   │
   ▼
3. STAGE 2: THINKING MODE FUSION
   │
   ├── Input: Outputs from Stage 1 model
   ├── Method: Supervised Fine-Tuning
   ├── Learn: /think and /no_think modes
   └── Output: Model with switchable reasoning
   │
   ▼
4. STAGE 3: ADVANCED RL
   │
   ├── Input: Same data as Stage 1
   ├── Algorithm: DAPO
   ├── Rewards: Stage 1 rewards + Hybrid + Thinking (via SophiaVL)
   └── Output: Final BusterX++ model
   │
   ▼
5. INFERENCE
   │
   ├── /think mode → Detailed explanation + classification
   └── /no_think mode → Fast classification only
```

---

## 4.3 Reward Functions

This section explains **how the model learns** through rewards and penalties. Think of it like training a dog: good behavior gets treats (positive reward), bad behavior gets corrections (negative penalty).

---

### Overview: When Each Reward Is Used

| Reward | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| Format | ✅ | - | ✅ |
| Soft Overlong | ✅ | - | ✅ |
| Accuracy | ✅ | - | ✅ |
| Hybrid Thinking | - | - | ✅ |
| Thinking Reward | - | - | ✅ |

---

### 1. Format Reward (r_fmt)

**What it checks:** Did the model use the correct output structure?

**Expected format:**
```
<think>reasoning here...</think><answer>Real or Fake</answer>
```

**How it works:**

| Situation | Reward | Explanation |
|-----------|--------|-------------|
| Correct format | r_fmt = 0 | No penalty, model followed instructions |
| Wrong format | r_fmt = -1 | Penalty for not following structure |

**Example:**
```
✅ GOOD: <think>The lighting looks unnatural...</think><answer>Fake</answer>
❌ BAD:  I think this is fake because the lighting looks weird.
```

**Why it matters:** Consistent format allows automated parsing and ensures the model always provides reasoning before answering.

---

### 2. Soft Overlong Reward (r_overlong)

**What it checks:** Is the response too long?

**The problem:** Very long responses waste compute and may contain rambling. But cutting off abruptly is also bad.

**How it works:** Uses a "soft" penalty with a buffer zone.

```
|←————————— Lmax (maximum length) ——————————→|
|←—— Safe zone ——→|←— Buffer (Lcache) —→|← Penalty zone →|
     r = 0              r = gradual           r = -1
```

**Formula:**

| Condition | Reward | Meaning |
|-----------|--------|---------|
| L_gen ≤ L_max - L_cache | r_overlong = 0 | Response is short enough, no penalty |
| L_max - L_cache < L_gen ≤ L_max | r_overlong = ((L_max - L_cache) - L_gen) / L_cache | In buffer zone: gradual penalty (between 0 and -1) |
| L_gen > L_max | r_overlong = -1 | Too long, full penalty |

**Intuition:**
- If you're well under the limit → no problem
- If you're approaching the limit → gentle warning (partial penalty)
- If you exceed the limit → full penalty

**Why "soft"?** Instead of a hard cutoff (0 or -1), the gradual penalty teaches the model to naturally stay within reasonable length.

---

### 3. Accuracy Reward (r_acc)

**What it checks:** Did the model classify correctly?

**How it works:**

| Situation | Reward |
|-----------|--------|
| Correct classification (predicted Real when actually Real, or Fake when Fake) | r_acc = +1 |
| Wrong classification | r_acc = 0 |

**Note:** Wrong answers get 0, not -1. This is intentional—the model isn't punished for trying, just not rewarded.

**Example:**
```
Video is actually: FAKE
Model predicts: Fake  → r_acc = +1 ✅
Model predicts: Real  → r_acc = 0  ❌
```

---

### 4. Hybrid Thinking Reward (r_hybrid)

**What it checks:** Did the model respect the thinking mode instruction?

**The two modes:**
- `/think` → Model MUST provide detailed reasoning
- `/no_think` → Model MUST skip reasoning (empty `<think></think>`)

**How it works:**

| Situation | Reward |
|-----------|--------|
| Model follows the mode correctly | r_hybrid = 0 |
| Model thinks when told `/no_think` | r_hybrid = -1 |
| Model skips thinking when told `/think` | r_hybrid = -1 |

**Examples:**
```
Instruction: /think
✅ GOOD: <think>The face shows unnatural movements...</think><answer>Fake</answer>
❌ BAD:  <think></think><answer>Fake</answer>  → Penalty! Should have reasoned.

Instruction: /no_think
✅ GOOD: <think></think><answer>Fake</answer>
❌ BAD:  <think>Let me analyze...</think><answer>Fake</answer>  → Penalty! Should be quick.
```

**Why it matters:** Allows flexibility—detailed analysis when needed, fast responses when speed matters.

---

### 5. Thinking Reward (r_think)

**What it checks:** Is the reasoning actually good quality?

**The challenge:** A model can produce text that looks like reasoning but is actually nonsense. How do we evaluate quality?

**Solution:** Use an external model (SophiaVL-R1-Thinking-Reward-Model-3B) to judge.

**How it works:**

| Mode | Formula | Explanation |
|------|---------|-------------|
| `/no_think` | r_think = 0 | No reasoning to evaluate |
| `/think` | r_think = min(r_acc, M(y_res)) | Quality score from external model, but capped by accuracy |

Where:
- M = external evaluation model (SophiaVL)
- y_res = model's response
- 0 ≤ r_think ≤ 1

**Why min(r_acc, ...)?**
- If classification is WRONG (r_acc = 0), reasoning quality doesn't matter → r_think = 0
- If classification is RIGHT (r_acc = 1), then reasoning quality matters → r_think = M(y_res)

**Intuition:** Good reasoning that leads to wrong answer = useless. We only reward good reasoning when it produces correct results.

**Example:**
```
Video: Actually FAKE

Response A:
<think>The lighting is inconsistent and shadows don't match...</think>
<answer>Fake</answer>
→ Correct + Good reasoning = r_think ≈ 0.9 ✅

Response B:
<think>The lighting is inconsistent and shadows don't match...</think>
<answer>Real</answer>
→ Wrong answer = r_think = 0 (good reasoning wasted) ❌

Response C:
<think>I don't know, maybe fake?</think>
<answer>Fake</answer>
→ Correct but poor reasoning = r_think ≈ 0.3 ⚠️
```

---

### Complete Reward Formulas

**Stage 1 (Foundation RL):**
```
R_stage-1 = r_fmt + r_overlong + r_acc
```
Range: -2 to +1

**Stage 3 (Advanced RL):**
```
R_stage-3 = r_fmt + r_overlong + r_acc + r_hybrid + r_think
```
Range: -3 to +2

---

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    REWARD FUNCTION FLOW                         │
└─────────────────────────────────────────────────────────────────┘

Model generates response
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  FORMAT CHECK   │     │  LENGTH CHECK   │     │ ACCURACY CHECK  │
│                 │     │                 │     │                 │
│ Correct format? │     │ Within limit?   │     │ Correct answer? │
│ Yes → 0         │     │ Yes → 0         │     │ Yes → +1        │
│ No  → -1        │     │ Buffer → partial│     │ No  → 0         │
│                 │     │ Over → -1       │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      STAGE 1 TOTAL      │
                    │  R = fmt + overlong +   │
                    │        accuracy         │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │           STAGE 3 ONLY              │
              ▼                                     ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  HYBRID CHECK   │               │ THINKING QUALITY│
    │                 │               │                 │
    │ Followed mode?  │               │ SophiaVL judges │
    │ Yes → 0         │               │ 0 to 1 score    │
    │ No  → -1        │               │ (only if /think │
    │                 │               │  and correct)   │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             └─────────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              │      STAGE 3 TOTAL      │
              │  R = Stage1 + hybrid +  │
              │       thinking          │
              └─────────────────────────┘
```

---

## 5. Experiments

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-VL-7B-Instruct |
| External Reward Model | SophiaVL-R1-Thinking-Reward-Model-3B |
| Video Sampling | 16 frames at 4 FPS |
| Fine-tuning Method | LoRA (rank=16, alpha=32) |
| Learning Rate | 1×10⁻⁵ |
| Precision | bfloat16 |
| Primary Metric | Accuracy (ACC) per subcategory |

---

## 5.1 Single-Modality Benchmarks

These benchmarks test the model on ONE type of content (either images OR videos, not both).

### So-Fake-Set Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **BusterX++** | **93.9%** | **93.7%** |
| BusterX++ (/no_think) | 92.3% | 92.1% |
| Previous SOTA (So-Fake-R1) | 93.2% | 92.9% |

**Interpretation:** BusterX++ beats the previous state-of-the-art by **+0.7%**. The /no_think mode loses only 1.6% accuracy but is faster.

### GenBuster-200K Results

| Model | Test Set ACC | Out-of-Domain ACC |
|-------|--------------|-------------------|
| **BusterX++** | **88.3%** | **92.4%** |
| BusterX++ (/no_think) | 87.5% | - |
| BusterX (previous) | 85.5% | 84.8% |

**Interpretation:**
- **+2.8%** improvement over BusterX on test set
- **+7.6%** improvement on out-of-domain data (better generalization!)

---

## 5.2 Cross-Modal Performance (GenBuster++ Benchmark)

This is the NEW benchmark created by the authors (Section 3.2) testing BOTH images AND videos together.

### Main Results (Table 4)

| Model | Real Img | Fake Img | Real Vid | Fake Vid | **Overall** |
|-------|----------|----------|----------|----------|-------------|
| **BusterX++** | 80.4% | 76.2% | 95.3% | 57.9% | **77.5%** |
| BusterX++ (/no_think) | 80.5% | 74.4% | 96.4% | 55.9% | 76.8% |
| BusterX (previous) | - | - | - | - | 68.3% |

**Baseline Comparisons (general MLLMs without fine-tuning):**

| Model | Overall Accuracy |
|-------|------------------|
| Qwen2.5-VL-7B | 55.4% |
| InternVL3-8B | 55.5% |
| MiniCPM-o 2.6 | 53.3% |

### Key Observations

| Finding | Value | Interpretation |
|---------|-------|----------------|
| Real videos easiest | 95.3% | Model excels at confirming authentic videos |
| Fake videos hardest | 57.9% | Detecting fake videos is challenging |
| /no_think accuracy drop | -0.7% | Minimal loss for faster inference |
| vs general MLLMs | +22% | Specialized training matters enormously |
| vs BusterX | +9.2% | Multi-stage RL significantly improves |

**Why are fake videos hardest?**
- State-of-the-art generators (Sora, Kling) produce very realistic content
- The benchmark specifically kept only the most convincing fakes
- Video has more dimensions (temporal) to get right/wrong

---

## 5.3 Cold-Start vs Non-Cold-Start Analysis

This proves WHY abandoning cold-start (Section 4.1) was the right choice.

### Comparison Table (Table 5)

| Strategy | Real Img | Fake Img | Real Vid | Fake Vid | **Overall** |
|----------|----------|----------|----------|----------|-------------|
| Cold-start only | 72.4% | 64.7% | 80.5% | 51.9% | 67.4% |
| Cold-start + Stage-3 | 81.0% | 65.9% | 91.4% | 53.2% | 72.9% |
| **No cold-start + Stage-3** | 81.2% | 76.7% | 94.1% | 57.5% | **77.4%** |

### The Crossover Effect

```
Accuracy
    ^
    │                                          ★ 77.4% (No cold-start)
80% │                                    ╱
    │                              ╱────╱
    │                        ╱────╱
75% │                  ╱────╱
    │            ╱────╱─────────────────── 72.9% (Cold-start)
    │      ╱────╱
70% │ ────╱
    │╱ 71.7%  (Cold-start starts higher)
    │  69.4%  (No cold-start starts lower)
    └────────────────────────────────────────> Training Stage
         Stage 1        Stage 2        Stage 3
```

**Key Insight:**
- Cold-start: 71.7% → 72.9% (only +1.2% gain)
- No cold-start: 69.4% → 77.4% (**+8.0% gain**)

No cold-start starts worse but ends MUCH better because it doesn't inherit biases from potentially low-quality CoT data.

---

## 5.4 Ablation Studies

### What Training Data Matters? (Table 6)

| Training Data | Real Img | Fake Img | Real Vid | Fake Vid | Overall |
|---------------|----------|----------|----------|----------|---------|
| Images only | 78.7% | 77.2% | 77.7% | 52.1% | 71.4% |
| Videos only | 75.9% | 67.9% | 95.9% | 51.9% | 72.9% |
| **Both (cross-modal)** | 80.4% | 76.2% | 95.3% | 57.9% | **77.5%** |

**Interpretation:** Training on BOTH modalities gives **+4.6%** over videos-only. Cross-modal learning helps the model generalize better.

### Which Training Stages Matter? (Table 7)

| Configuration | Overall Accuracy |
|---------------|------------------|
| Stage 1 only | 69.4% |
| Stage 1 + Stage 2 | 69.3% |
| Stage 1 + Stage 3 | 77.4% |
| **All three stages** | **77.5%** |

**Interpretation:**
- Stage 2 alone adds almost nothing (+0.0%)
- Stage 3 is critical (+8.0%)
- Stage 2 is only needed to enable Stage 3's hybrid reasoning

---

## 5.5 Robustness Evaluation

Real-world content often has compression, noise, or blur. Does the model still work?

### Perturbation Types Applied

| Perturbation | Settings |
|--------------|----------|
| JPEG compression | quality=70 |
| Gaussian noise | σ=5 |
| Gaussian blur | standard |
| Cascade | All above combined (Real-ESRGAN style) |

### Results Under Perturbations (Table 8)

| Condition | Real Img | Fake Img | Real Vid | Fake Vid | **Overall** |
|-----------|----------|----------|----------|----------|-------------|
| **Clean** | 80.4% | 76.2% | 95.3% | 57.9% | **77.5%** |
| JPEG only | 82.1% | 67.2% | 94.5% | 55.6% | 74.9% |
| Noise only | 76.4% | 66.7% | 95.1% | 49.2% | 71.9% |
| Blur only | 91.6% | 66.4% | 93.9% | 57.6% | 77.4% |
| Cascade (all) | 90.8% | 53.5% | 97.0% | 40.8% | 70.5% |

### Robustness Analysis

| Perturbation | Accuracy Drop | Assessment |
|--------------|---------------|------------|
| JPEG | -2.6% | Good robustness |
| Noise | -5.6% | Moderate impact |
| Blur | -0.1% | Excellent robustness |
| Cascade | -7.0% | Challenging but acceptable |

**Key Observations:**
- **Blur helps real detection** (91.6% vs 80.4%) - blurring makes real content easier to identify
- **Fake images suffer most** under cascade (53.5%) - compression destroys subtle fake artifacts
- **Real videos improve** under cascade (97.0%) - perhaps because degradation makes them look more "natural"

---

## 5.6 Case Study

The paper provides visual examples showing:

1. **Stable Reasoning:** Model consistently identifies the same artifacts across similar content
2. **Low-level Attention:** Model notices subtle details (unnatural reflections, lighting inconsistencies)
3. **Knowledge-based Inference:** Model uses world knowledge (e.g., "this politician wouldn't be in this context")

---

## Summary: Key Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| **Best overall accuracy** | 77.5% | On GenBuster++ (cross-modal) |
| **So-Fake-Set accuracy** | 93.9% | Single-modality benchmark |
| **GenBuster-200K accuracy** | 88.3% | Large-scale benchmark |
| **vs general MLLMs** | +22% | BusterX++ vs Qwen2.5-VL |
| **vs BusterX** | +9.2% | Improvement from multi-stage RL |
| **Cold-start vs no cold-start** | +4.5% | No cold-start wins after Stage 3 |
| **/no_think accuracy drop** | -0.7% | Minimal cost for fast inference |
| **Hardest category** | 57.9% | Fake videos |
| **Easiest category** | 95.3% | Real videos |
| **Worst robustness** | -7.0% | Under cascade perturbation |

---

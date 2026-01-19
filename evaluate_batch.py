"""
BusterX++ Batch Evaluation Script
==================================
Evaluates the model on multiple videos and calculates metrics.
"""

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
from decord import VideoReader, cpu
import os
import random
from collections import defaultdict

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "/home/miriam/Documents/BusterX_plusplus"
DATASET_PATH = "/home/miriam/Documents/BusterX_plusplus/data/GenBuster-200K-mini/GenBuster-200K-mini/benchmark"

# Number of videos to test per category (reduce for faster testing)
SAMPLES_PER_CATEGORY = 3  # 3 real + 3 fake = 6 total

VIDEO_PROMPT = """Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""

# ============================================
# SETUP
# ============================================
print("=" * 60)
print("Setting up model...")
print("=" * 60)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "6GiB", "cpu": "24GiB"},
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    max_pixels=147456  # 384x384 for 8GB VRAM
)

print("Model loaded!")

# ============================================
# HELPER FUNCTIONS
# ============================================
def load_video_frames(video_path, num_frames=16, target_fps=4.0):
    """Load frames from video using decord."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    frame_interval = max(1, int(fps / target_fps))
    frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]

def predict_video(video_path):
    """Run inference on a single video."""
    # Load frames
    pil_frames = load_video_frames(video_path)

    # Prepare input
    content = []
    for frame in pil_frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": VIDEO_PROMPT})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_frames, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    # Generate
    torch.cuda.empty_cache()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    response = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    # Extract prediction
    if "B" in response[-50:] or "fake" in response[-100:].lower():
        return "FAKE", response
    elif "A" in response[-50:] or "real" in response[-100:].lower():
        return "REAL", response
    else:
        return "UNCLEAR", response

def get_sample_videos(dataset_path, samples_per_category):
    """Get random sample of videos from dataset."""
    videos = []

    # Real videos
    real_path = os.path.join(dataset_path, "real")
    real_files = [f for f in os.listdir(real_path) if f.endswith('.mp4')]
    sampled_real = random.sample(real_files, min(samples_per_category, len(real_files)))
    for f in sampled_real:
        videos.append((os.path.join(real_path, f), "REAL", "real"))

    # Fake videos (from first available generator)
    fake_path = os.path.join(dataset_path, "fake")
    generators = [d for d in os.listdir(fake_path) if os.path.isdir(os.path.join(fake_path, d))]

    if generators:
        gen = generators[0]  # Use first generator
        gen_path = os.path.join(fake_path, gen)
        fake_files = [f for f in os.listdir(gen_path) if f.endswith('.mp4')]
        sampled_fake = random.sample(fake_files, min(samples_per_category, len(fake_files)))
        for f in sampled_fake:
            videos.append((os.path.join(gen_path, f), "FAKE", gen))

    return videos

# ============================================
# RUN EVALUATION
# ============================================
print("\n" + "=" * 60)
print(f"Evaluating on {SAMPLES_PER_CATEGORY} real + {SAMPLES_PER_CATEGORY} fake videos")
print("=" * 60)

videos = get_sample_videos(DATASET_PATH, SAMPLES_PER_CATEGORY)
random.shuffle(videos)

results = []
for i, (video_path, ground_truth, source) in enumerate(videos):
    print(f"\n[{i+1}/{len(videos)}] Testing: {os.path.basename(video_path)[:20]}... ({ground_truth})")

    try:
        prediction, response = predict_video(video_path)
        correct = prediction == ground_truth
        results.append({
            "video": os.path.basename(video_path),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": correct,
            "source": source,
            "response_preview": response[:100] + "..." if len(response) > 100 else response
        })
        print(f"    Prediction: {prediction} | Correct: {'YES' if correct else 'NO'}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results.append({
            "video": os.path.basename(video_path),
            "ground_truth": ground_truth,
            "prediction": "ERROR",
            "correct": False,
            "source": source,
            "response_preview": str(e)
        })

# ============================================
# CALCULATE METRICS
# ============================================
print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)

# Basic counts
total = len(results)
correct = sum(1 for r in results if r["correct"])
incorrect = total - correct

# Confusion matrix components
TP = sum(1 for r in results if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
TN = sum(1 for r in results if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
FP = sum(1 for r in results if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
FN = sum(1 for r in results if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")

# Metrics
accuracy = correct / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"""
Confusion Matrix:
                 Predicted
              REAL    FAKE
Actual REAL    {TN}      {FP}     (TN, FP)
Actual FAKE    {FN}      {TP}     (FN, TP)

Metrics:
  Accuracy:  {accuracy:.2%} ({correct}/{total})
  Precision: {precision:.2%} (of predicted fakes, how many were actually fake)
  Recall:    {recall:.2%} (of actual fakes, how many did we catch)
  F1 Score:  {f1:.2%} (harmonic mean of precision and recall)
""")

# ============================================
# DETAILED RESULTS
# ============================================
print("=" * 60)
print("DETAILED RESULTS")
print("=" * 60)

for r in results:
    status = "OK" if r["correct"] else "WRONG"
    print(f"\n[{status}] {r['video'][:40]}...")
    print(f"      Ground Truth: {r['ground_truth']} | Prediction: {r['prediction']} | Source: {r['source']}")
    print(f"      Response: {r['response_preview']}")

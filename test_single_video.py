"""
BusterX++ Single Video Test Script
===================================
This script loads the model with 4-bit quantization and tests on one video.
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

# Pick a test video (this is a REAL video from benchmark)
VIDEO_PATH = "/home/miriam/Documents/BusterX_plusplus/data/GenBuster-200K-mini/GenBuster-200K-mini/benchmark/real/4e4c5bfc2ac9599bd4b77d5f0b2b9ab7ee0a13923239a218278dcf581f8c6712.mp4"

# The prompt from the BusterX++ paper
VIDEO_PROMPT = """Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""

# ============================================
# STEP 1: Set up 4-bit quantization
# ============================================
print("=" * 60)
print("STEP 1: Setting up 4-bit quantization config")
print("=" * 60)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Quantization config created!")

# ============================================
# STEP 2: Load the model
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Loading model (this takes 2-5 minutes)")
print("=" * 60)

# Hybrid CPU+GPU: Limit GPU to 6GB, rest goes to CPU RAM
# This is slower but allows full 16 frames at original resolution
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "6GiB", "cpu": "24GiB"},  # GPU 0 limited, overflow to CPU
    trust_remote_code=True
)

print("Model loaded successfully!")

# ============================================
# STEP 3: Load the processor
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Loading processor")
print("=" * 60)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, max_pixels=147456)
print("Processor loaded! (max_pixels=147456 / 384x384 for 8GB VRAM)")

# ============================================
# STEP 4: Check GPU memory usage
# ============================================
print("\n" + "=" * 60)
print("STEP 4: GPU Memory Usage")
print("=" * 60)

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  GPU Memory Allocated: {allocated:.2f} GB")
    print(f"  GPU Memory Reserved:  {reserved:.2f} GB")

# ============================================
# STEP 5: Load video frames using decord
# ============================================
print("\n" + "=" * 60)
print("STEP 5: Loading video frames")
print("=" * 60)
print(f"Video: {VIDEO_PATH}")

# Use decord to read video frames
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)
fps = vr.get_avg_fps()
print(f"  Total frames: {total_frames}")
print(f"  FPS: {fps:.2f}")

# Sample 16 frames at 4 FPS (as per BusterX++ paper)
target_fps = 4.0
num_frames = 16
frame_interval = max(1, int(fps / target_fps))
frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

# Get frames as PIL images
frames = vr.get_batch(frame_indices).asnumpy()
pil_frames = [Image.fromarray(frame) for frame in frames]
print(f"  Sampled {len(pil_frames)} frames")

# ============================================
# STEP 6: Prepare the input
# ============================================
print("\n" + "=" * 60)
print("STEP 6: Preparing model input")
print("=" * 60)

# Build conversation with video frames as images
content = []
for i, frame in enumerate(pil_frames):
    content.append({"type": "image", "image": frame})
content.append({"type": "text", "text": VIDEO_PROMPT})

messages = [{"role": "user", "content": content}]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process inputs
inputs = processor(
    text=[text],
    images=pil_frames,
    return_tensors="pt",
    padding=True
)

# Move to GPU
inputs = inputs.to(model.device)
print("Input prepared!")

# ============================================
# STEP 7: Generate response
# ============================================
print("\n" + "=" * 60)
print("STEP 7: Generating response (this takes ~30-120 seconds)")
print("=" * 60)

# Clear cache before inference to maximize available memory
torch.cuda.empty_cache()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

# Decode the response
response = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

# ============================================
# STEP 8: Display results
# ============================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("\nModel's Response:")
print("-" * 40)
print(response)
print("-" * 40)

# Extract the answer
if "B" in response[-50:] or "fake" in response[-100:].lower():
    prediction = "FAKE"
elif "A" in response[-50:] or "real" in response[-100:].lower():
    prediction = "REAL"
else:
    prediction = "UNCLEAR"

print(f"\nPrediction: {prediction}")
print(f"Ground Truth: REAL (this is a real video from benchmark)")
print(f"Correct: {'YES' if prediction == 'REAL' else 'NO'}")

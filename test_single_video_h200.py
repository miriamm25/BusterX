"""
BusterX++ Single Video Test Script
"""

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

MODEL_PATH = "l8cv/BusterX_plusplus"
VIDEO_PATH = "/home/miriam/data/GenBuster-200K-mini/test/fake/hunyuanvideo/040ad7f712cb29195faae064ee23d5ff22cf608825aa5419753205820c3eed4a.mp4"

VIDEO_PROMPT = """Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?

Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc.

Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake"""

print("=" * 60)
print("STEP 1: Loading model (H200 - bfloat16)")
print("=" * 60)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully!")

print("\n" + "=" * 60)
print("STEP 2: Loading processor")
print("=" * 60)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Processor loaded!")

print("\n" + "=" * 60)
print("STEP 3: GPU Memory Usage")
print("=" * 60)

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  GPU Memory Allocated: {allocated:.2f} GB")
    print(f"  GPU Memory Reserved:  {reserved:.2f} GB")

print("\n" + "=" * 60)
print("STEP 4: Loading video frames")
print("=" * 60)
print(f"Video: {VIDEO_PATH}")

vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)
fps = vr.get_avg_fps()
print(f"  Total frames: {total_frames}")
print(f"  FPS: {fps:.2f}")

target_fps = 4.0
num_frames = 16
frame_interval = max(1, int(fps / target_fps))
frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]

frames = vr.get_batch(frame_indices).asnumpy()
pil_frames = [Image.fromarray(frame) for frame in frames]
print(f"  Sampled {len(pil_frames)} frames")

print("\n" + "=" * 60)
print("STEP 5: Preparing model input")
print("=" * 60)

content = []
for i, frame in enumerate(pil_frames):
    content.append({"type": "image", "image": frame})
content.append({"type": "text", "text": VIDEO_PROMPT})

messages = [{"role": "user", "content": content}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[text],
    images=pil_frames,
    return_tensors="pt",
    padding=True
)
inputs = inputs.to(model.device)
print("Input prepared!")

print("\n" + "=" * 60)
print("STEP 6: Generating response")
print("=" * 60)

torch.cuda.empty_cache()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=750,
        do_sample=True,
        temperature=1e-06,
        repetition_penalty=1.05
    )

response = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("\nModel's Response:")
print("-" * 40)
print(response)
print("-" * 40)

if "B" in response[-50:] or "fake" in response[-100:].lower():
    prediction = "FAKE"
elif "A" in response[-50:] or "real" in response[-100:].lower():
    prediction = "REAL"
else:
    prediction = "UNCLEAR"

ground_truth = "FAKE" if "/fake/" in VIDEO_PATH else "REAL"
print(f"\nPrediction: {prediction}")
print(f"Ground Truth: {ground_truth}")
print(f"Correct: {'YES' if prediction == ground_truth else 'NO'}")

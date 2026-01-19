"""
BusterX++ Single Video Test Script - DEBUG VERSION
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
print("DEBUG: Loading model")
print("=" * 60)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# DEBUG: Print model config
print("\n[DEBUG] Model config:")
print(f"  - Model type: {model.config.model_type}")
print(f"  - Hidden size: {model.config.hidden_size}")
print(f"  - Num layers: {model.config.num_hidden_layers}")
print(f"  - Vocab size: {model.config.vocab_size}")

print("\n" + "=" * 60)
print("DEBUG: Loading processor")
print("=" * 60)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

# DEBUG: Print processor info
print("\n[DEBUG] Processor info:")
print(f"  - Processor class: {type(processor)}")
print(f"  - Tokenizer class: {type(processor.tokenizer)}")
print(f"  - Image processor class: {type(processor.image_processor)}")

print("\n" + "=" * 60)
print("DEBUG: Loading video frames")
print("=" * 60)

vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)
fps = vr.get_avg_fps()
print(f"[DEBUG] Video: {VIDEO_PATH}")
print(f"[DEBUG] Total frames: {total_frames}, FPS: {fps:.2f}")

target_fps = 4.0
num_frames = 16
frame_interval = max(1, int(fps / target_fps))
frame_indices = [min(i * frame_interval, total_frames - 1) for i in range(num_frames)]
print(f"[DEBUG] Frame indices: {frame_indices}")

frames = vr.get_batch(frame_indices).asnumpy()
pil_frames = [Image.fromarray(frame) for frame in frames]
print(f"[DEBUG] Sampled {len(pil_frames)} frames, size: {pil_frames[0].size}")

print("\n" + "=" * 60)
print("DEBUG: Preparing input")
print("=" * 60)

content = []
for i, frame in enumerate(pil_frames):
    content.append({"type": "image", "image": frame})
content.append({"type": "text", "text": VIDEO_PROMPT})

messages = [{"role": "user", "content": content}]

# DEBUG: Print message structure
print(f"\n[DEBUG] Message structure:")
print(f"  - Number of content items: {len(content)}")
print(f"  - Content types: {[c['type'] for c in content]}")

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# DEBUG: Print chat template output
print(f"\n[DEBUG] Chat template output (first 500 chars):")
print(text[:500])
print("...")
print(f"\n[DEBUG] Chat template output (last 500 chars):")
print(text[-500:])

inputs = processor(
    text=[text],
    images=pil_frames,
    return_tensors="pt",
    padding=True
)

# DEBUG: Print input shapes
print(f"\n[DEBUG] Input tensors:")
for key, value in inputs.items():
    if hasattr(value, 'shape'):
        print(f"  - {key}: {value.shape}")
    else:
        print(f"  - {key}: {type(value)}")

inputs = inputs.to(model.device)
print(f"[DEBUG] Inputs moved to device: {model.device}")

print("\n" + "=" * 60)
print("DEBUG: Generating response")
print("=" * 60)

# DEBUG: Print generation config
print(f"\n[DEBUG] Model generation config:")
print(f"  - do_sample: {model.generation_config.do_sample}")
print(f"  - temperature: {model.generation_config.temperature}")
print(f"  - max_new_tokens (default): {getattr(model.generation_config, 'max_new_tokens', 'not set')}")
print(f"  - repetition_penalty: {getattr(model.generation_config, 'repetition_penalty', 'not set')}")

torch.cuda.empty_cache()

print("\n[DEBUG] Starting generation with:")
print("  - max_new_tokens=750")
print("  - do_sample=True")
print("  - temperature=1e-06")
print("  - repetition_penalty=1.05")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=750,
        do_sample=True,
        temperature=1e-06,
        repetition_penalty=1.05
    )

# DEBUG: Print output info
print(f"\n[DEBUG] Output shape: {output_ids.shape}")
print(f"[DEBUG] Input length: {inputs.input_ids.shape[1]}")
print(f"[DEBUG] Generated tokens: {output_ids.shape[1] - inputs.input_ids.shape[1]}")

# DEBUG: Print raw generated token IDs
generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
print(f"[DEBUG] Generated token IDs: {generated_ids.tolist()}")

response = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

# DEBUG: Also decode without skipping special tokens
response_with_special = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=False
)[0]

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("\n[DEBUG] Response (with special tokens):")
print("-" * 40)
print(response_with_special)
print("-" * 40)

print("\n[DEBUG] Response (without special tokens):")
print("-" * 40)
print(response)
print("-" * 40)

print(f"\n[DEBUG] Response length: {len(response)} chars")

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

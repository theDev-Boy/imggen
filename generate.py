#!/usr/bin/env python3
"""
FAST & QUALITY Image Generator
Model: Realistic Vision v5.1 (2GB) - Great faces, fast!
"""

import torch
import time
import argparse
from diffusers import DiffusionPipeline
from PIL import Image, ImageEnhance

def generate_single_image(pipe, prompt, steps=8, index=1):
    """Generate one high-quality image"""
    print(f"\n🎨 Image {index}: {prompt}")
    
    # Enhance prompt for better humans
    enhanced = f"{prompt}, photorealistic, highly detailed face, detailed skin, professional photography, 8k, sharp focus, masterpiece"
    negative = "cartoon, painting, blurry, distorted face, bad anatomy, extra limbs, ugly, deformed, disfigured, bad hands, missing fingers, watermark, text"
    
    gen_start = time.time()
    
    with torch.no_grad():
        image = pipe(
            prompt=enhanced,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    
    # Save
    filename = f"output_{index}.png"
    image.save(filename, "PNG")
    
    gen_time = time.time() - gen_start
    print(f"✅ Image {index} done in {gen_time:.1f}s")
    
    return gen_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prompt1", type=str, default="")
    parser.add_argument("--prompt2", type=str, default="")
    parser.add_argument("--prompt3", type=str, default="")
    parser.add_argument("--prompt4", type=str, default="")
    parser.add_argument("--prompt5", type=str, default="")
    
    args = parser.parse_args()
    
    # Collect prompts
    prompts = []
    for i in range(1, args.count + 1):
        prompt = getattr(args, f"prompt{i}", "")
        if prompt:
            prompts.append(prompt)
        else:
            prompts.append(prompts[0] if prompts else "a boy jumping from airplane, photorealistic, cinematic")
    
    print("="*50)
    print("🚀 FAST QUALITY IMAGE GENERATOR")
    print(f"📸 Images: {len(prompts)}")
    print(f"⚙️  Steps: {args.steps}")
    print("="*50)
    
    # Load model - FAST 2GB model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    
    print("📥 Loading model (~2GB, first time only)...")
    load_start = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # CPU optimizations
    pipe.enable_attention_slicing()
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    # Generate all images
    total_gen = 0
    for i, prompt in enumerate(prompts, 1):
        gen_time = generate_single_image(pipe, prompt, args.steps, i)
        total_gen += gen_time
    
    # Save info
    with open("info.txt", "w") as f:
        f.write(f"Model: Realistic Vision V5.1\n")
        f.write(f"Steps per image: {args.steps}\n")
        f.write(f"Total images: {args.count}\n")
        f.write(f"Model load: {load_time:.1f}s\n")
        f.write(f"Total generation: {total_gen:.1f}s\n")
        f.write("\nPrompts:\n")
        for i, p in enumerate(prompts, 1):
            f.write(f"{i}. {p}\n")
    
    print("\n" + "="*50)
    print(f"🎉 ALL DONE!")
    print(f"📸 Generated {args.count} image(s)")
    print(f"⏱️  Total time: {load_time + total_gen:.1f}s")
    print("="*50)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
⚡ FAST Image Generation using LCM-LoRA + Stable Diffusion 1.5
Optimized for speed on CPU
Model size: ~2.5GB (cached after first download)
Generation time: 25-40 seconds on CPU
"""

import torch
import sys
import time
import argparse
import os
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image

def generate_image(prompt, num_inference_steps=4, seed=None):
    """
    Generate image using LCM-LoRA for fast inference
    """
    print("🚀 Starting FAST Image Generation")
    print(f"📝 Prompt: {prompt}")
    print(f"⚙️  Steps: {num_inference_steps}")
    
    start_time = time.time()
    
    # Model ID - LCM is fastest
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    
    # Check if offline mode
    offline_mode = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
    if offline_mode:
        print("📦 Using cached model (offline mode)")
    
    print("📥 Loading model...")
    load_start = time.time()
    
    # Load pipeline with optimizations
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=offline_mode,  # Use cache if available
    )
    
    # Speed optimizations
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # These make it faster on CPU
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_slicing()
    
    # Optional: Model CPU offload (saves RAM but slower)
    # pipe.enable_sequential_cpu_offload()
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    # Set seed
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        print(f"🎲 Using seed: {seed}")
    else:
        generator = torch.Generator().manual_seed(torch.initial_seed())
        print(f"🎲 Random seed generated")
    
    # Generate image
    print("🎨 Generating image...")
    gen_start = time.time()
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,  # LCM works best with 1.0-2.0
            generator=generator,
            height=512,
            width=512,
            num_images_per_prompt=1,
        ).images[0]
    
    gen_time = time.time() - gen_start
    total_time = time.time() - start_time
    
    # Save image
    output_path = "output.png"
    image.save(output_path, "PNG", optimize=True)
    
    # Save generation info
    with open("generation-info.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Steps: {num_inference_steps}\n")
        f.write(f"Seed: {seed if seed else 'random'}\n")
        f.write(f"Model Load Time: {load_time:.1f}s\n")
        f.write(f"Generation Time: {gen_time:.1f}s\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"Image Size: 512x512\n")
        f.write(f"Offline Mode: {offline_mode}\n")
    
    print(f"✨ Image generated in {gen_time:.1f}s!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"💾 Saved to: {output_path}")
    print(f"📊 Speed: {gen_time/num_inference_steps:.1f}s per step")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast AI Image Generation")
    parser.add_argument("prompt", type=str, help="Image description")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (2-8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Limit steps for LCM
    if args.steps > 8:
        print("⚠️  LCM works best with 2-8 steps. Using 8.")
        args.steps = 8
    
    generate_image(args.prompt, args.steps, args.seed)

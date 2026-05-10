#!/usr/bin/env python3
import torch
import sys
import time
import argparse
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image

def generate_image(prompt, num_inference_steps=4, seed=None):
    print("🚀 Starting Fast Image Generation")
    print(f"📝 Prompt: {prompt}")
    print(f"⚙️  Steps: {num_inference_steps}")
    
    start_time = time.time()
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    
    print("📥 Loading model (first time downloads ~2.5GB)...")
    load_start = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    
    try:
        pipe.enable_sequential_cpu_offload()
    except:
        print("⚠️  Could not enable CPU offload, using default")
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        print(f"🎲 Using seed: {seed}")
    else:
        generator = torch.Generator().manual_seed(torch.seed())
        print(f"🎲 Random seed generated")
    
    print("🎨 Generating image...")
    gen_start = time.time()
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.5,
            generator=generator,
            height=512,
            width=512,
        ).images[0]
    
    gen_time = time.time() - gen_start
    total_time = time.time() - start_time
    
    output_path = "output.png"
    image.save(output_path, "PNG")
    
    with open("generation-info.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Steps: {num_inference_steps}\n")
        f.write(f"Seed: {seed if seed else 'random'}\n")
        f.write(f"Model Load Time: {load_time:.1f}s\n")
        f.write(f"Generation Time: {gen_time:.1f}s\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"Image Size: 512x512\n")
    
    print(f"✨ Image generated in {gen_time:.1f}s!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"💾 Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast AI Image Generation")
    parser.add_argument("prompt", type=str, help="Image description")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (2-8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.steps > 8:
        print("⚠️  LCM works best with 2-8 steps. Using 8.")
        args.steps = 8
    
    generate_image(args.prompt, args.steps, args.seed)

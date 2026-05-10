#!/usr/bin/env python3
"""
QUALITY Image Generator - Great faces and bodies!
Uses SDXL Turbo or Realistic Vision for humans
"""

import torch
import time
import argparse
import os
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from PIL import Image

def generate_image(prompt, style="photorealistic", num_steps=1):
    print("🚀 QUALITY Image Generator")
    print(f"📝 Prompt: {prompt}")
    print(f"🎨 Style: {style}")
    print(f"⚙️  Steps: {num_steps}")
    
    start = time.time()
    
    # Choose model based on style
    if style == "anime":
        model_id = "cagliostrolab/animagine-xl-3.1"
        print("🌸 Loading Anime model...")
    elif style == "artistic":
        model_id = "stabilityai/sdxl-turbo"
        print("🎨 Loading Artistic model...")
    else:
        # BEST FOR REALISTIC HUMANS
        model_id = "stabilityai/sdxl-turbo"
        print("📸 Loading Photorealistic model...")
    
    # Load with optimizations
    load_start = time.time()
    
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
        )
    except:
        # Fallback to basic SDXL
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float32,
        )
    
    # CPU optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    print(f"✅ Model loaded in {time.time()-load_start:.1f}s")
    
    # Enhance prompt for better humans
    if style == "photorealistic":
        enhanced_prompt = f"{prompt}, photorealistic, highly detailed face, detailed eyes, detailed skin texture, professional photography, 8k uhd, sharp focus, realistic human proportions"
        negative_prompt = "cartoon, painting, blurry, distorted face, bad anatomy, extra limbs, ugly, deformed, disfigured, bad proportions, unnatural body"
    elif style == "anime":
        enhanced_prompt = f"{prompt}, anime style, studio ghibli, detailed, high quality"
        negative_prompt = "photorealistic, ugly, deformed"
    else:
        enhanced_prompt = f"{prompt}, artistic, beautiful, detailed"
        negative_prompt = "ugly, deformed, bad quality"
    
    print("🎨 Creating your image...")
    gen_start = time.time()
    
    with torch.no_grad():
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0 if num_steps <= 2 else 1.0,
            height=768,
            width=768,
        ).images[0]
    
    gen_time = time.time() - gen_start
    
    # Enhance image quality
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    # Save
    output = "output.png"
    image.save(output, "PNG", quality=95)
    
    total = time.time() - start
    print(f"✨ DONE! Generated in {gen_time:.1f}s")
    print(f"⏱️  Total: {total:.1f}s")
    print(f"📁 Saved: {output}")
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--style", type=str, default="photorealistic")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    
    generate_image(args.prompt, args.style, args.steps)

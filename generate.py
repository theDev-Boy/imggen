#!/usr/bin/env python3
"""
SMART Image Generator - OpenRouter enhances prompts first!
Model: baidu/cobuddy + DreamShaper LCM
"""

import torch
import time
import argparse
import requests
import json
import os
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image, ImageEnhance

# OpenRouter API Config
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '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')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "inclusionai/ring-2.6-1t:free"

def enhance_prompt_with_ai(user_prompt):
    """Use OpenRouter to create better image prompts"""
    
    print(f"🧠 Enhancing: {user_prompt[:80]}...")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "AI Image Generator"
    }
    
    system_prompt = """You are an expert at creating image generation prompts. 
Transform the user's simple prompt into a detailed, photorealistic image prompt.
Follow these rules:
1. Add photography terms (8k, cinematic, professional, sharp focus, detailed)
2. Describe lighting, environment, atmosphere
3. Keep it under 200 characters
4. Make it work great for Stable Diffusion
5. Output ONLY the enhanced prompt, nothing else
6. NO quotes, NO explanations, just the prompt"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Enhance this for image generation: {user_prompt}"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            enhanced = response.json()['choices'][0]['message']['content'].strip()
            # Clean up any quotes
            enhanced = enhanced.replace('"', '').replace("'", "")
            print(f"✅ Enhanced: {enhanced[:100]}...")
            return enhanced
        else:
            print(f"⚠️ API error, using original prompt")
            return user_prompt
            
    except Exception as e:
        print(f"⚠️ Connection error: {e}, using original prompt")
        return user_prompt

def generate_image(pipe, prompt, steps, index, original_prompt):
    """Generate image with enhanced prompt"""
    
    print(f"\n🎨 Image {index}")
    print(f"📝 Original: {original_prompt[:60]}...")
    print(f"🧠 Enhanced: {prompt[:80]}...")
    
    # Add quality boosters
    final_prompt = f"{prompt}, highly detailed, masterpiece, professional photography, 8k"
    negative = "blurry, ugly, deformed, bad anatomy, cartoon, painting, watermark, text, low quality, distorted face"
    
    start = time.time()
    
    with torch.no_grad():
        image = pipe(
            prompt=final_prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=1.5,
            height=512,
            width=512,
        ).images[0]
    
    # Enhance image quality
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    filename = f"output_{index}.png"
    image.save(filename, "PNG", optimize=True)
    
    gen_time = time.time() - start
    print(f"✅ Generated in {gen_time:.1f}s")
    
    return gen_time, original_prompt, prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--steps", type=str, default="fast")
    parser.add_argument("--p1", type=str, default="")
    parser.add_argument("--p2", type=str, default="")
    parser.add_argument("--p3", type=str, default="")
    parser.add_argument("--p4", type=str, default="")
    parser.add_argument("--p5", type=str, default="")
    args = parser.parse_args()
    
    steps_map = {"fast": 4, "balanced": 6, "quality": 8}
    steps = steps_map.get(args.steps, 4)
    
    # Get prompts
    user_prompts = [p for p in [args.p1, args.p2, args.p3, args.p4, args.p5][:args.count] if p]
    
    print("="*60)
    print("🧠 SMART IMAGE GENERATOR WITH AI ENHANCEMENT")
    print(f"🤖 Enhancer: {MODEL_NAME}")
    print(f"📸 Images: {len(user_prompts)}")
    print(f"⚙️  Steps: {steps}")
    print("="*60)
    
    # Step 1: Enhance all prompts with AI
    print("\n📡 Enhancing prompts with OpenRouter...")
    enhanced_prompts = []
    for i, prompt in enumerate(user_prompts, 1):
        enhanced = enhance_prompt_with_ai(prompt)
        enhanced_prompts.append(enhanced)
    
    # Step 2: Load image generation model
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    print(f"\n📥 Loading image model...")
    t0 = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    load_time = time.time() - t0
    print(f"✅ Image model loaded in {load_time:.1f}s")
    
    # Step 3: Generate images
    print(f"\n🎨 Generating {len(enhanced_prompts)} image(s)...")
    
    prompt_info = []
    total_gen = 0
    
    for i, (original, enhanced) in enumerate(zip(user_prompts, enhanced_prompts), 1):
        gen_time, orig, enh = generate_image(pipe, enhanced, steps, i, original)
        total_gen += gen_time
        prompt_info.append(f"Image {i}:")
        prompt_info.append(f"  Original: {orig}")
        prompt_info.append(f"  Enhanced: {enh}")
        prompt_info.append("")
    
    # Save prompt info
    with open("prompts_info.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write("PROMPT ENHANCEMENT REPORT\n")
        f.write(f"Enhancer Model: {MODEL_NAME}\n")
        f.write("="*50 + "\n\n")
        f.write("\n".join(prompt_info))
    
    total_time = load_time + total_gen
    print(f"\n{'='*60}")
    print(f"🎉 ALL DONE!")
    print(f"📸 Generated: {len(user_prompts)} image(s)")
    print(f"🧠 Prompt enhancement: ~2-3s each")
    print(f"🎨 Image generation: {total_gen:.1f}s total")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"📄 Check 'prompts_info.txt' for enhancement details")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SMART Image Generator with OpenRouter Enhancement
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
    
    print(f"🧠 Original: {user_prompt}")
    print(f"📡 Sending to OpenRouter using {MODEL_NAME}...")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simplified system prompt that works better
    system_prompt = "You enhance image prompts. Add visual details, lighting, atmosphere, and quality terms like photorealistic, 8k, cinematic. Output ONLY the enhanced prompt with no other text."
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"Enhance this image prompt with vivid visual details: '{user_prompt}'. Return ONLY the enhanced prompt text."}
        ],
        "temperature": 0.9,
        "max_tokens": 200,
        "top_p": 0.95
    }
    
    try:
        print(f"🔑 Using API Key: {OPENROUTER_API_KEY[:20]}...")
        print(f"📤 Sending request...")
        
        response = requests.post(
            OPENROUTER_URL, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Full Response: {json.dumps(result, indent=2)[:500]}")
            
            if 'choices' in result and len(result['choices']) > 0:
                enhanced = result['choices'][0]['message']['content'].strip()
                enhanced = enhanced.replace('"', '').replace("'", "").strip()
                
                # Check if enhancement worked
                if enhanced and len(enhanced) > len(user_prompt) + 10:
                    print(f"✅ Enhanced: {enhanced}")
                    return enhanced
                else:
                    print(f"⚠️ Enhancement too short or empty: '{enhanced}'")
                    # Manual enhancement as fallback
                    return manual_enhance(user_prompt)
            else:
                print(f"❌ No choices in response")
                return manual_enhance(user_prompt)
        elif response.status_code == 429:
            print(f"⏳ Rate limited! Using manual enhancement")
            return manual_enhance(user_prompt)
        elif response.status_code == 401:
            print(f"🔒 Authentication failed! Check your API key")
            return manual_enhance(user_prompt)
        else:
            print(f"❌ API Error {response.status_code}: {response.text[:200]}")
            return manual_enhance(user_prompt)
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out")
        return manual_enhance(user_prompt)
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        return manual_enhance(user_prompt)

def manual_enhance(prompt):
    """Manual prompt enhancement when API fails"""
    enhancements = [
        "photorealistic, highly detailed, professional photography, 8k resolution, sharp focus, masterpiece",
        "cinematic lighting, dramatic atmosphere, intricate details, award winning photo, hyperrealistic",
        "stunning visual, perfect composition, vibrant colors, depth of field, ultra detailed, breathtaking"
    ]
    
    import random
    booster = random.choice(enhancements)
    enhanced = f"{prompt}, {booster}"
    print(f"🔧 Manual enhancement: {enhanced[:100]}...")
    return enhanced

def generate_image(pipe, prompt, steps, index, original_prompt):
    """Generate image with enhanced prompt"""
    
    print(f"\n{'='*50}")
    print(f"🎨 GENERATING IMAGE {index}")
    print(f"📝 Original: {original_prompt}")
    print(f"🧠 Using: {prompt[:150]}")
    
    # Add quality boosters
    final_prompt = f"{prompt}, highly detailed, sharp focus"
    negative = "blurry, ugly, deformed, bad anatomy, cartoon, painting, watermark, text, low quality, distorted face, extra fingers, bad hands"
    
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
    
    # Enhance image
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    filename = f"output_{index}.png"
    image.save(filename, "PNG", optimize=True)
    
    gen_time = time.time() - start
    print(f"✅ Generated in {gen_time:.1f}s")
    print(f"💾 Saved as: {filename}")
    
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
    
    user_prompts = [p for p in [args.p1, args.p2, args.p3, args.p4, args.p5][:args.count] if p]
    
    print("="*60)
    print("🧠 SMART IMAGE GENERATOR WITH AI ENHANCEMENT")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"📸 Images: {len(user_prompts)}")
    print(f"⚙️  Steps: {steps}")
    print("="*60)
    
    # Step 1: Enhance prompts
    print("\n📡 PHASE 1: Enhancing prompts...")
    enhanced_prompts = []
    for i, prompt in enumerate(user_prompts, 1):
        print(f"\n--- Prompt {i}/{len(user_prompts)} ---")
        enhanced = enhance_prompt_with_ai(prompt)
        enhanced_prompts.append(enhanced)
    
    # Step 2: Load model
    print("\n📥 PHASE 2: Loading image generation model...")
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
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
    print(f"✅ Loaded in {load_time:.1f}s")
    
    # Step 3: Generate images
    print(f"\n🎨 PHASE 3: Generating images...")
    
    prompt_info = []
    total_gen = 0
    
    for i, (original, enhanced) in enumerate(zip(user_prompts, enhanced_prompts), 1):
        gen_time, orig, enh = generate_image(pipe, enhanced, steps, i, original)
        total_gen += gen_time
        prompt_info.append(f"Image {i}:")
        prompt_info.append(f"  Original: {orig}")
        prompt_info.append(f"  Enhanced: {enh}")
        prompt_info.append("")
    
    # Save info
    with open("prompts_info.txt", "w", encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("PROMPT ENHANCEMENT REPORT\n")
        f.write(f"Enhancer Model: {MODEL_NAME}\n")
        f.write(f"Total Time: {load_time + total_gen:.1f}s\n")
        f.write("="*50 + "\n\n")
        f.write("\n".join(prompt_info))
    
    total_time = load_time + total_gen
    print(f"\n{'='*60}")
    print(f"🎉 GENERATION COMPLETE!")
    print(f"📸 {len(user_prompts)} image(s)")
    print(f"⏱️  Total: {total_time:.1f}s")
    print(f"📄 Report: prompts_info.txt")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

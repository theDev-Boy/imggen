#!/usr/bin/env python3
import torch
import time
import argparse
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image, ImageEnhance

def generate_fast(pipe, prompt, steps, index):
    print(f"\n🎨 Image {index}: {prompt[:60]}...")
    enhanced = f"{prompt}, highly detailed, sharp focus, professional photography, 8k, masterpiece"
    negative = "blurry, ugly, deformed, bad anatomy, cartoon, painting, watermark, text, low quality"
    start = time.time()
    with torch.no_grad():
        image = pipe(
            prompt=enhanced,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=1.5,
            height=512,
            width=512,
        ).images[0]
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    filename = f"output_{index}.png"
    image.save(filename, "PNG", optimize=True)
    t = time.time() - start
    print(f"✅ Done in {t:.1f}s")
    return t

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
    
    prompts = [p for p in [args.p1, args.p2, args.p3, args.p4, args.p5][:args.count] if p]
    
    print("="*50)
    print("⚡ FAST IMAGE GENERATOR")
    print(f"📸 Images: {len(prompts)}")
    print(f"⚙️  Steps: {steps}")
    print("="*50)
    
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    print("📥 Loading model...")
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
    print(f"✅ Loaded in {load_time:.1f}s\n")
    
    total_gen = 0
    for i, prompt in enumerate(prompts, 1):
        t = generate_fast(pipe, prompt, steps, i)
        total_gen += t
    
    total = load_time + total_gen
    print(f"\n{'='*50}")
    print(f"🎉 COMPLETE! {len(prompts)} image(s)")
    print(f"⏱️  Total: {total:.1f}s")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

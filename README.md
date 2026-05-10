# 🎨 Fast AI Image Generator

Generate images using GitHub Actions with LCM (Latent Consistency Models)!

## 🚀 Quick Start

1. Go to **Actions** tab
2. Click **"Generate AI Image"**
3. Click **"Run workflow"**
4. Enter your prompt: `boy jumping from airplane`
5. Choose steps (4 is fast, 8 is better quality)
6. Click **"Run workflow"**

## ⏱️ Performance

- **Model**: LCM Dreamshaper v7 (~2.5GB)
- **Generation Time**: 2-5 minutes on CPU
- **Steps**: 4-8 (vs 20-50 for regular SD)
- **Resolution**: 512x512

## 💡 Tips

- Use 4 steps for speed, 8 for quality
- LCM works best with simple, clear prompts
- First run downloads model (~2.5GB), cached after
- Images stored as artifacts for 7 days

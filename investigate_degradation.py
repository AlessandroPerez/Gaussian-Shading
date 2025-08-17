#!/usr/bin/env python3
"""
Investigation Script - Find what's breaking the watermark detection
This script will save images and investigate each step of the pipeline
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import transform_img, set_random_seed, image_distortion
from watermark import Gaussian_Shading

def investigate_pipeline_degradation():
    """Investigate what's causing the watermark detection degradation"""
    print("🔍 INVESTIGATING PIPELINE DEGRADATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pipeline_investigation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    print(f"📁 Saving results to: {output_dir}")
    
    # Initialize pipeline exactly like working tests
    scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    if device == 'cuda':
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base',
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
        )
    else:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base',
            scheduler=scheduler,
            torch_dtype=torch.float32,
        )
    
    pipe.safety_checker = None
    pipe = pipe.to(device)
    
    # Initialize watermark
    watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
    
    print(f"✅ Threshold: {watermark.tau_onebit:.6f}")
    
    # Test with 5 images to investigate the issue
    num_test_images = 5
    
    for i in range(num_test_images):
        print(f"\n🧪 TESTING IMAGE {i+1}/{num_test_images}")
        print("-" * 40)
        
        prompt = "A beautiful landscape painting"
        
        # STEP 1: Generate watermarked image
        print("1️⃣ Generating watermarked image...")
        set_random_seed(i)
        init_latents_w = watermark.create_watermark_and_return_w()
        
        if device == 'cuda':
            init_latents_w = init_latents_w.to(torch.float16).to(device)
        else:
            init_latents_w = init_latents_w.to(torch.float32).to(device)
        
        outputs = pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=7.5,
            num_inference_steps=20,
            height=512,
            width=512,
            latents=init_latents_w,
        )
        
        original_image = outputs.images[0]
        original_image.save(f"{output_dir}/images/original_{i:02d}.png")
        print(f"   💾 Saved: original_{i:02d}.png")
        
        # STEP 2: Test immediate detection
        print("2️⃣ Testing immediate detection...")
        image_tensor = transform_img(original_image).unsqueeze(0)
        if device == 'cuda':
            image_tensor = image_tensor.to(torch.float16).to(device)
        else:
            image_tensor = image_tensor.to(torch.float32).to(device)
        
        image_latents = pipe.get_image_latents(image_tensor, sample=False)
        text_embeddings = pipe.get_text_embedding('')
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        immediate_accuracy = watermark.eval_watermark(reversed_latents)
        immediate_detected = immediate_accuracy >= watermark.tau_onebit
        
        print(f"   ✅ Immediate: Acc={immediate_accuracy:.6f}, Detected={immediate_detected}")
        
        # STEP 3: Apply "none" attack (save and reload)
        print("3️⃣ Applying 'none' attack (save/reload cycle)...")
        
        # This mimics what happens in attack testing
        temp_path = f"{output_dir}/images/temp_{i:02d}.png"
        original_image.save(temp_path)
        reloaded_image = Image.open(temp_path)
        reloaded_image.save(f"{output_dir}/images/reloaded_{i:02d}.png")
        print(f"   💾 Saved: reloaded_{i:02d}.png")
        
        # Test detection on reloaded image
        print("4️⃣ Testing detection on reloaded image...")
        reloaded_tensor = transform_img(reloaded_image).unsqueeze(0)
        if device == 'cuda':
            reloaded_tensor = reloaded_tensor.to(torch.float16).to(device)
        else:
            reloaded_tensor = reloaded_tensor.to(torch.float32).to(device)
        
        reloaded_latents = pipe.get_image_latents(reloaded_tensor, sample=False)
        reloaded_text_embeddings = pipe.get_text_embedding('')
        reloaded_reversed_latents = pipe.forward_diffusion(
            latents=reloaded_latents,
            text_embeddings=reloaded_text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        reloaded_accuracy = watermark.eval_watermark(reloaded_reversed_latents)
        reloaded_detected = reloaded_accuracy >= watermark.tau_onebit
        
        print(f"   ✅ Reloaded: Acc={reloaded_accuracy:.6f}, Detected={reloaded_detected}")
        
        # STEP 5: Apply actual distortion
        print("5️⃣ Testing with actual distortion...")
        
        class MockArgs:
            def __init__(self):
                self.jpeg_ratio = 75
                self.gaussian_blur_r = None
                self.gaussian_std = None
                self.resize_ratio = None
                self.random_crop_ratio = None
                self.brightness_factor = None
                self.median_blur_k = None
                self.random_drop_ratio = None
                self.sp_prob = None
        
        mock_args = MockArgs()
        distorted_image = image_distortion(original_image, i, mock_args)
        distorted_image.save(f"{output_dir}/images/distorted_{i:02d}.png")
        print(f"   💾 Saved: distorted_{i:02d}.png")
        
        # Test detection on distorted image
        distorted_tensor = transform_img(distorted_image).unsqueeze(0)
        if device == 'cuda':
            distorted_tensor = distorted_tensor.to(torch.float16).to(device)
        else:
            distorted_tensor = distorted_tensor.to(torch.float32).to(device)
        
        distorted_latents = pipe.get_image_latents(distorted_tensor, sample=False)
        distorted_text_embeddings = pipe.get_text_embedding('')
        distorted_reversed_latents = pipe.forward_diffusion(
            latents=distorted_latents,
            text_embeddings=distorted_text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        distorted_accuracy = watermark.eval_watermark(distorted_reversed_latents)
        distorted_detected = distorted_accuracy >= watermark.tau_onebit
        
        print(f"   ✅ Distorted: Acc={distorted_accuracy:.6f}, Detected={distorted_detected}")
        
        # STEP 6: Analyze degradation
        print("6️⃣ Analysis:")
        immediate_to_reloaded_drop = immediate_accuracy - reloaded_accuracy
        reloaded_to_distorted_drop = reloaded_accuracy - distorted_accuracy
        total_drop = immediate_accuracy - distorted_accuracy
        
        print(f"   📉 Immediate → Reloaded drop: {immediate_to_reloaded_drop:.6f}")
        print(f"   📉 Reloaded → Distorted drop: {reloaded_to_distorted_drop:.6f}")
        print(f"   📉 Total drop: {total_drop:.6f}")
        
        if abs(immediate_to_reloaded_drop) > 0.01:
            print("   🚨 Major degradation during save/reload!")
        if abs(reloaded_to_distorted_drop) > 0.01:
            print("   🚨 Major degradation during distortion!")
        
        # Check image differences
        print("7️⃣ Image analysis:")
        original_array = np.array(original_image)
        reloaded_array = np.array(reloaded_image)
        distorted_array = np.array(distorted_image)
        
        original_vs_reloaded_diff = np.mean(np.abs(original_array.astype(float) - reloaded_array.astype(float)))
        reloaded_vs_distorted_diff = np.mean(np.abs(reloaded_array.astype(float) - distorted_array.astype(float)))
        
        print(f"   📊 Original vs Reloaded pixel diff: {original_vs_reloaded_diff:.3f}")
        print(f"   📊 Reloaded vs Distorted pixel diff: {reloaded_vs_distorted_diff:.3f}")
    
    print(f"\n✅ Investigation complete! Check {output_dir}/ for saved images")
    print(f"\n🎯 KEY INSIGHTS:")
    print("   - Compare immediate vs reloaded accuracy drops")
    print("   - Check pixel differences between image versions")
    print("   - Look for patterns in accuracy degradation")

if __name__ == "__main__":
    investigate_pipeline_degradation()

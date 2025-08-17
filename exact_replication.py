#!/usr/bin/env python3
"""
Exact replication of final_fresh_test.py methodology to find the failure point
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import transform_img, set_random_seed
from watermark import Gaussian_Shading

def run_exact_test():
    """Run exact replication with different seed ranges"""
    print("🔍 EXACT REPLICATION OF FAILED TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"exact_replication_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # Initialize pipeline - EXACT same as final_fresh_test
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
    
    # Initialize watermark - EXACT same parameters
    watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
    
    print(f"✅ tau_onebit: {watermark.tau_onebit:.6f}")
    print(f"✅ tau_bits: {watermark.tau_bits:.6f}")
    
    # EXACT same prompts as final_fresh_test
    prompts = [
        "A serene mountain landscape at sunset",
        "A modern city skyline with glass buildings", 
        "A colorful flower garden in spring",
        "An abstract geometric pattern",
        "A peaceful forest with tall trees",
        "A futuristic spacecraft in space",
        "A vintage car on a country road", 
        "A cozy cottage with a garden",
        "A stormy ocean with large waves",
        "A desert scene with sand dunes"
    ]
    
    # Test different seed ranges
    seed_ranges = [
        {"name": "Original range (42-61)", "start": 42, "count": 20},
        {"name": "Higher range (100-119)", "start": 100, "count": 20},
        {"name": "Very high range (1000-1019)", "start": 1000, "count": 20},
        {"name": "Timestamp-like range", "start": int(time.time()) % 10000, "count": 20},
    ]
    
    for seed_range in seed_ranges:
        print(f"\n🧪 TESTING: {seed_range['name']}")
        print("-" * 50)
        
        watermarked_images = []
        generation_accuracies = []
        generation_detections = []
        
        for i in range(seed_range['count']):
            prompt = prompts[i % len(prompts)]
            seed = seed_range['start'] + i
            
            print(f"  🎨 Image {i+1}: seed={seed}, prompt='{prompt[:30]}...'")
            
            # EXACT same generation process
            set_random_seed(seed)
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
            
            image = outputs.images[0]
            watermarked_images.append(image)
            
            # Save image
            range_name = seed_range['name'].replace(" ", "_").replace("(", "").replace(")", "")
            image_path = f"{output_dir}/images/{range_name}_{i:02d}_seed{seed}.png"
            image.save(image_path)
            
            # EXACT same detection process
            image_tensor = transform_img(image).unsqueeze(0)
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
            
            accuracy = watermark.eval_watermark(reversed_latents)
            detected = accuracy >= watermark.tau_onebit
            generation_accuracies.append(accuracy)
            generation_detections.append(detected)
            
            if i < 5:  # Show details for first few
                print(f"    ✅ Acc: {accuracy:.6f}, Detected: {detected}")
        
        # Summary for this range
        avg_accuracy = np.mean(generation_accuracies)
        detection_rate = sum(generation_detections) / len(generation_detections)
        
        print(f"\n📊 SUMMARY for {seed_range['name']}:")
        print(f"   🎯 Average Accuracy: {avg_accuracy:.6f}")
        print(f"   ✅ Detection Rate: {detection_rate:.3f} ({sum(generation_detections)}/{len(generation_detections)})")
        
        if detection_rate < 1.0:
            failed_indices = [i for i, detected in enumerate(generation_detections) if not detected]
            print(f"   ❌ Failed images: {failed_indices}")
            for idx in failed_indices:
                seed = seed_range['start'] + idx
                print(f"     Image {idx}: seed={seed}, acc={generation_accuracies[idx]:.6f}")
    
    print(f"\n✅ Analysis complete! Images saved in {output_dir}/images/")
    print(f"\n🎯 This should reveal if the issue was:")
    print("   - Specific seed values")
    print("   - Pipeline state changes")
    print("   - Random statistical variation")

if __name__ == "__main__":
    run_exact_test()

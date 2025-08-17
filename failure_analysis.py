#!/usr/bin/env python3
"""
Failure Analysis - Test the specific seeds/prompts that caused low detection
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import transform_img, set_random_seed
from watermark import Gaussian_Shading

def test_specific_failures():
    """Test the specific seed/prompt combinations that failed"""
    print("🔍 TESTING SPECIFIC FAILURE CASES")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"failure_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # Initialize pipeline
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
    
    # Test different seeding strategies and prompts
    test_cases = [
        # Working case (from investigation)
        {"name": "Sequential seed + Fixed prompt", "seeds": [0, 1, 2, 3, 4], "prompts": ["A beautiful landscape painting"] * 5},
        
        # Fresh test case (that failed)
        {"name": "Time-based seed + Varied prompts", 
         "seeds": [42 + i for i in range(5)], 
         "prompts": [
             "A serene mountain landscape at sunset",
             "A modern city skyline with glass buildings",
             "A colorful flower garden in spring",
             "An abstract geometric pattern",
             "A peaceful forest with tall trees"
         ]},
        
        # Mixed cases
        {"name": "Sequential seed + Varied prompts", 
         "seeds": [0, 1, 2, 3, 4], 
         "prompts": [
             "A serene mountain landscape at sunset",
             "A modern city skyline with glass buildings", 
             "A colorful flower garden in spring",
             "An abstract geometric pattern",
             "A peaceful forest with tall trees"
         ]},
        
        {"name": "Time-based seed + Fixed prompt", 
         "seeds": [42 + i for i in range(5)], 
         "prompts": ["A beautiful landscape painting"] * 5},
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 TESTING: {test_case['name']}")
        print("-" * 50)
        
        case_results = []
        
        for i in range(5):
            seed = test_case['seeds'][i]
            prompt = test_case['prompts'][i]
            
            print(f"   Image {i+1}: seed={seed}, prompt='{prompt[:30]}...'")
            
            # Generate watermarked image
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
            
            # Save image
            case_name = test_case['name'].replace(" ", "_").replace("+", "and")
            image_path = f"{output_dir}/images/{case_name}_{i:02d}.png"
            image.save(image_path)
            
            # Test detection immediately
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
            case_results.append(accuracy)
            
            print(f"     → Acc: {accuracy:.6f}, Detected: {detected}")
        
        # Summary for this case
        avg_accuracy = np.mean(case_results)
        detection_rate = sum(1 for acc in case_results if acc >= watermark.tau_onebit) / len(case_results)
        
        print(f"   📊 SUMMARY:")
        print(f"     Average accuracy: {avg_accuracy:.6f}")
        print(f"     Detection rate: {detection_rate:.3f} ({sum(1 for acc in case_results if acc >= watermark.tau_onebit)}/{len(case_results)})")
        
        if detection_rate < 1.0:
            failed_indices = [i for i, acc in enumerate(case_results) if acc < watermark.tau_onebit]
            print(f"     ❌ Failed images: {failed_indices}")
            for idx in failed_indices:
                print(f"       Image {idx}: seed={test_case['seeds'][idx]}, acc={case_results[idx]:.6f}")
    
    print(f"\n✅ Analysis complete! Images saved in {output_dir}/images/")
    print(f"\n🎯 This will help identify if the issue is:")
    print("   - Specific prompts causing issues")
    print("   - Certain seed values being problematic") 
    print("   - Random generation variability")

if __name__ == "__main__":
    test_specific_failures()

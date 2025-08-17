#!/usr/bin/env python3
"""
DETAILED DIAGNOSTIC TEST FOR GAUSSIAN SHADING SYSTEM
===================================================

This script runs detailed diagnostics to identify issues with the watermarking system.
"""

import torch
import numpy as np
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import set_random_seed, transform_img
from PIL import Image


def test_watermark_generation_and_detection():
    """Test the core watermark generation and detection loop"""
    print("🔍 DETAILED WATERMARK DIAGNOSTIC")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize watermark system
    print("\n1. Initializing watermark system...")
    watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
    print(f"   Detection threshold: {watermark.tau_onebit}")
    print(f"   Traceability threshold: {watermark.tau_bits}")
    
    # Initialize diffusion pipeline
    print("\n2. Loading diffusion pipeline...")
    try:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base', subfolder='scheduler'
        )
        
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
        print("   ✅ Pipeline loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Pipeline loading failed: {e}")
        return False
    
    # Test watermark creation
    print("\n3. Testing watermark creation...")
    try:
        set_random_seed(42)
        init_latents_w = watermark.create_watermark_and_return_w()
        print(f"   ✅ Watermarked latents shape: {init_latents_w.shape}")
        print(f"   ✅ Watermarked latents dtype: {init_latents_w.dtype}")
        print(f"   ✅ Watermarked latents device: {init_latents_w.device}")
        
        # Check latent values
        print(f"   Latents min: {init_latents_w.min():.4f}, max: {init_latents_w.max():.4f}")
        print(f"   Latents mean: {init_latents_w.mean():.4f}, std: {init_latents_w.std():.4f}")
        
    except Exception as e:
        print(f"   ❌ Watermark creation failed: {e}")
        return False
    
    # Test image generation
    print("\n4. Testing image generation...")
    try:
        prompt = "A beautiful sunset over mountains"
        with torch.no_grad():
            outputs = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                num_inference_steps=20,  # Reduced for faster testing
                height=512,
                width=512,
                latents=init_latents_w,
            )
        
        image_w = outputs.images[0]
        print(f"   ✅ Generated image size: {image_w.size}")
        
        # Save test image
        image_w.save("diagnostic_watermarked.png")
        print("   ✅ Image saved as diagnostic_watermarked.png")
        
    except Exception as e:
        print(f"   ❌ Image generation failed: {e}")
        return False
    
    # Test watermark detection process step by step
    print("\n5. Testing watermark detection (step by step)...")
    try:
        # Convert image to tensor
        image_tensor = transform_img(image_w, 512).unsqueeze(0)
        if device == 'cuda':
            image_tensor = image_tensor.half()
        image_tensor = image_tensor.to(device)
        print(f"   ✅ Image tensor shape: {image_tensor.shape}")
        
        # Get image latents
        with torch.no_grad():
            image_latents = pipe.get_image_latents(image_tensor, sample=False)
            print(f"   ✅ Image latents shape: {image_latents.shape}")
            print(f"   Image latents min: {image_latents.min():.4f}, max: {image_latents.max():.4f}")
            
            # Get text embeddings for empty prompt
            text_embeddings = pipe.get_text_embedding('')
            print(f"   ✅ Text embeddings shape: {text_embeddings.shape}")
            
            # Forward diffusion (inversion)
            print("   Running DDIM inversion...")
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=20,
            )
            print(f"   ✅ Reversed latents shape: {reversed_latents.shape}")
            print(f"   Reversed latents min: {reversed_latents.min():.4f}, max: {reversed_latents.max():.4f}")
            
            # Test watermark evaluation
            print("   Evaluating watermark...")
            accuracy = watermark.eval_watermark(reversed_latents)
            print(f"   ✅ Watermark accuracy: {accuracy:.6f}")
            print(f"   Detection threshold: {watermark.tau_onebit:.6f}")
            print(f"   Detected: {'YES' if accuracy >= watermark.tau_onebit else 'NO'}")
            
            # Check if this is the issue - compare original and recovered
            print("\n6. Comparing original and recovered latents...")
            diff = torch.abs(init_latents_w - reversed_latents)
            print(f"   L1 difference mean: {diff.mean():.6f}")
            print(f"   L1 difference max: {diff.max():.6f}")
            
            # Test direct evaluation on original latents (should be perfect)
            print("\n7. Testing on original watermarked latents (sanity check)...")
            direct_accuracy = watermark.eval_watermark(init_latents_w)
            print(f"   Direct accuracy: {direct_accuracy:.6f}")
            print(f"   Direct detected: {'YES' if direct_accuracy >= watermark.tau_onebit else 'NO'}")
            
    except Exception as e:
        print(f"   ❌ Watermark detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test on clean image for comparison
    print("\n8. Testing on clean (non-watermarked) image...")
    try:
        set_random_seed(123)
        with torch.no_grad():
            clean_outputs = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                num_inference_steps=20,
                height=512,
                width=512,
            )
        
        clean_image = clean_outputs.images[0]
        clean_image.save("diagnostic_clean.png")
        
        # Test detection on clean image
        clean_tensor = transform_img(clean_image, 512).unsqueeze(0)
        if device == 'cuda':
            clean_tensor = clean_tensor.half()
        clean_tensor = clean_tensor.to(device)
        
        with torch.no_grad():
            clean_latents = pipe.get_image_latents(clean_tensor, sample=False)
            clean_reversed = pipe.forward_diffusion(
                latents=clean_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=20,
            )
            clean_accuracy = watermark.eval_watermark(clean_reversed)
            
        print(f"   Clean image accuracy: {clean_accuracy:.6f}")
        print(f"   Clean detected: {'YES' if clean_accuracy >= watermark.tau_onebit else 'NO'}")
        
    except Exception as e:
        print(f"   ❌ Clean image test failed: {e}")
    
    # Summary
    print(f"\n{'='*40}")
    print("DIAGNOSTIC SUMMARY:")
    print(f"{'='*40}")
    print(f"Watermarked image accuracy: {accuracy:.6f}")
    print(f"Clean image accuracy: {clean_accuracy:.6f}")
    print(f"Detection threshold: {watermark.tau_onebit:.6f}")
    print(f"Direct watermark accuracy: {direct_accuracy:.6f}")
    print(f"")
    
    if direct_accuracy >= watermark.tau_onebit:
        print("✅ Watermark embedding is working correctly")
    else:
        print("❌ Watermark embedding has issues")
    
    if accuracy >= watermark.tau_onebit:
        print("✅ Watermark detection after diffusion is working")
    else:
        print("❌ Watermark detection after diffusion has issues")
        print("   This could be due to:")
        print("   - DDIM inversion not being perfect")
        print("   - Too few inversion steps")
        print("   - Threshold being too high")
    
    if clean_accuracy < watermark.tau_onebit:
        print("✅ False positive rate is low (good)")
    else:
        print("❌ False positive rate is high (bad)")
    
    return True


if __name__ == "__main__":
    test_watermark_generation_and_detection()

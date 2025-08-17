#!/usr/bin/env python3
"""
Side-by-side comparison: clean_only_test approach vs fresh test approach
"""

import torch
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Gaussian_Shading
from image_utils import set_random_seed, transform_img

def compare_approaches():
    """Compare clean_only_test approach vs fresh test approach"""
    print("🔍 Side-by-side comparison of approaches...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize pipeline exactly like clean_only_test
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
    
    print(f"✅ Setup complete. Testing with 5 images...")
    
    # APPROACH 1: Exact clean_only_test method
    print(f"\n🧪 APPROACH 1: Exact clean_only_test replication")
    
    prompt = "A beautiful landscape painting"  # Fixed prompt like clean_only_test
    text_embeddings = pipe.get_text_embedding('')
    
    approach1_accuracies = []
    approach1_detections = []
    
    for i in range(5):
        print(f"   Image {i+1}/5", end=" ")
        
        # Generate watermarked image (exact same method)
        set_random_seed(i)  # Sequential seeds like clean_only_test
        init_latents_w = watermark.create_watermark_and_return_w()
        
        # Ensure correct dtype to match pipeline
        if device == 'cuda':
            init_latents_w = init_latents_w.to(torch.float16).to(device)
        else:
            init_latents_w = init_latents_w.to(torch.float32).to(device)
        
        outputs = pipe(
            prompt,  # Fixed prompt
            num_images_per_prompt=1,
            guidance_scale=7.5,
            num_inference_steps=20,
            height=512,
            width=512,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        
        # Convert to tensor and get latents (exact same method)
        image_w_tensor = transform_img(image_w).unsqueeze(0)
        if device == 'cuda':
            image_w_tensor = image_w_tensor.to(torch.float16).to(device)
        else:
            image_w_tensor = image_w_tensor.to(torch.float32).to(device)
            
        image_latents = pipe.get_image_latents(image_w_tensor, sample=False)
        
        # Reverse diffusion (exact same method)
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        # Evaluate watermark
        accuracy = watermark.eval_watermark(reversed_latents)
        detected = accuracy >= watermark.tau_onebit
        
        approach1_accuracies.append(accuracy)
        approach1_detections.append(detected)
        
        print(f"Acc: {accuracy:.4f}, Detected: {detected}")
    
    print(f"   📊 Approach 1 - Avg Accuracy: {sum(approach1_accuracies)/len(approach1_accuracies):.4f}")
    print(f"   📊 Approach 1 - Detection Rate: {sum(approach1_detections)/len(approach1_detections):.3f}")
    
    # APPROACH 2: Fresh test method (varied prompts, different seeds)
    print(f"\n🧪 APPROACH 2: Fresh test method")
    
    prompts = [
        "A serene mountain landscape at sunset",
        "A modern city skyline with glass buildings", 
        "A colorful flower garden in spring",
        "An abstract geometric pattern",
        "A peaceful forest with tall trees"
    ]
    
    approach2_accuracies = []
    approach2_detections = []
    
    for i in range(5):
        print(f"   Image {i+1}/5", end=" ")
        
        # Generate watermarked image (fresh test method)
        set_random_seed(int(12345) + i * 17)  # Different seed calculation
        init_latents_w = watermark.create_watermark_and_return_w()
        
        # Ensure correct dtype
        if device == 'cuda':
            init_latents_w = init_latents_w.to(torch.float16).to(device)
        else:
            init_latents_w = init_latents_w.to(torch.float32).to(device)
        
        outputs = pipe(
            prompts[i],  # Variable prompts
            num_images_per_prompt=1,
            guidance_scale=7.5,
            num_inference_steps=20,
            height=512,
            width=512,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        
        # Convert to tensor and get latents
        image_w_tensor = transform_img(image_w).unsqueeze(0)
        if device == 'cuda':
            image_w_tensor = image_w_tensor.to(torch.float16).to(device)
        else:
            image_w_tensor = image_w_tensor.to(torch.float32).to(device)
            
        image_latents = pipe.get_image_latents(image_w_tensor, sample=False)
        
        # Reverse diffusion
        text_embeddings = pipe.get_text_embedding('')  # Get fresh each time
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        # Evaluate watermark
        accuracy = watermark.eval_watermark(reversed_latents)
        detected = accuracy >= watermark.tau_onebit
        
        approach2_accuracies.append(accuracy)
        approach2_detections.append(detected)
        
        print(f"Acc: {accuracy:.4f}, Detected: {detected}")
    
    print(f"   📊 Approach 2 - Avg Accuracy: {sum(approach2_accuracies)/len(approach2_accuracies):.4f}")
    print(f"   📊 Approach 2 - Detection Rate: {sum(approach2_detections)/len(approach2_detections):.3f}")
    
    print(f"\n🎯 COMPARISON SUMMARY:")
    print(f"   Approach 1 (clean_only_test): {sum(approach1_detections)/len(approach1_detections):.3f} detection rate")
    print(f"   Approach 2 (fresh test): {sum(approach2_detections)/len(approach2_detections):.3f} detection rate")
    
    diff = (sum(approach1_accuracies)/len(approach1_accuracies)) - (sum(approach2_accuracies)/len(approach2_accuracies))
    print(f"   Accuracy difference: {diff:.6f}")

if __name__ == "__main__":
    compare_approaches()

#!/usr/bin/env python3
"""
Debug script to understand the pipeline method differences
"""

import torch
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Gaussian_Shading
from image_utils import set_random_seed, transform_img

def debug_pipeline_methods():
    """Debug what methods are actually available"""
    print("🔍 Debugging Pipeline Methods...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize pipeline like clean_only_test
    scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base',
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        revision='fp16' if device == 'cuda' else None,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)
    
    print(f"📋 Pipeline type: {type(pipe)}")
    
    # Check available methods
    methods = [method for method in dir(pipe) if not method.startswith('_')]
    diffusion_methods = [method for method in methods if 'diffusion' in method.lower()]
    
    print(f"🎯 Diffusion-related methods: {diffusion_methods}")
    
    # Test if methods exist
    has_forward = hasattr(pipe, 'forward_diffusion')
    has_backward = hasattr(pipe, 'backward_diffusion')
    
    print(f"✅ Has forward_diffusion: {has_forward}")
    print(f"✅ Has backward_diffusion: {has_backward}")
    
    # Generate a test image and try both approaches
    watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
    
    print(f"\n🧪 Testing watermark pipeline...")
    
    # Generate watermarked image
    set_random_seed(42)
    init_latents_w = watermark.create_watermark_and_return_w()
    
    if device == 'cuda':
        init_latents_w = init_latents_w.to(torch.float16).to(device)
    else:
        init_latents_w = init_latents_w.to(torch.float32).to(device)
    
    # Generate image
    outputs = pipe(
        "A beautiful landscape painting",
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
    text_embeddings = pipe.get_text_embedding('')
    
    print(f"🔍 Image latents shape: {image_latents.shape}")
    print(f"🔍 Text embeddings shape: {text_embeddings.shape}")
    
    # Try backward_diffusion (what we used)
    try:
        print(f"\n🧪 Testing backward_diffusion...")
        reversed_latents_backward = pipe.backward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        accuracy_backward = watermark.eval_watermark(reversed_latents_backward)
        print(f"✅ backward_diffusion accuracy: {accuracy_backward:.6f}")
        
    except Exception as e:
        print(f"❌ backward_diffusion failed: {e}")
    
    # Try forward_diffusion (what clean_only_test uses)
    try:
        print(f"\n🧪 Testing forward_diffusion...")
        reversed_latents_forward = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        accuracy_forward = watermark.eval_watermark(reversed_latents_forward)
        print(f"✅ forward_diffusion accuracy: {accuracy_forward:.6f}")
        
    except Exception as e:
        print(f"❌ forward_diffusion failed: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
    
    # Test if it's a typo and actually calling something else
    try:
        print(f"\n🧪 Testing if forward_diffusion exists via getattr...")
        forward_method = getattr(pipe, 'forward_diffusion', None)
        print(f"forward_diffusion method: {forward_method}")
    except Exception as e:
        print(f"❌ getattr failed: {e}")

if __name__ == "__main__":
    debug_pipeline_methods()

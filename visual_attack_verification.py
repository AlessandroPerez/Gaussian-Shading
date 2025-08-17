#!/usr/bin/env python3
"""
Visual verification of attack effects
"""

import torch
from pathlib import Path
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from watermark import Gaussian_Shading
from image_utils import *
import numpy as np

class VisualAttackVerification:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Initialize pipeline
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base', subfolder='scheduler'
        )
        
        if self.device == 'cuda':
            self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base',
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
            )
        else:
            self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base',
                scheduler=scheduler,
                torch_dtype=torch.float32,
            )
        
        self.pipe.safety_checker = None
        self.pipe = self.pipe.to(self.device)
        
        # Initialize watermark
        self.watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
        
        # Create output directory
        self.output_dir = Path("./visual_attack_verification")
        self.output_dir.mkdir(exist_ok=True)
        
    def apply_attack_visual(self, image, attack_name, attack_params):
        """Apply attack and return modified image"""
        class MockArgs:
            def __init__(self, **kwargs):
                # Set all parameters to None first
                for param in ['jpeg_ratio', 'gaussian_blur_r', 'gaussian_std', 'resize_ratio', 
                             'random_crop_ratio', 'random_drop_ratio', 'median_blur_k', 
                             'sp_prob', 'brightness_factor']:
                    setattr(self, param, None)
                # Set specific parameters
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        args = MockArgs(**attack_params)
        seed = 42
        
        attacked_image = image_distortion(image, seed, args)
        return attacked_image
    
    def test_visual_attacks(self):
        """Generate and save images to visually verify attack effects"""
        print(f"\n🎨 Generating test image...")
        
        # Generate base watermarked image
        prompt = "A beautiful landscape painting"
        set_random_seed(42)
        init_latents_w = self.watermark.create_watermark_and_return_w()
        
        outputs = self.pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=7.5,
            num_inference_steps=20,
            height=512,
            width=512,
            latents=init_latents_w,
        )
        base_image = outputs.images[0]
        
        # Save base image
        base_image.save(self.output_dir / "00_base_watermarked.png")
        print(f"   💾 Saved: 00_base_watermarked.png")
        
        # Define attacks to test
        attacks = {
            "01_jpeg_high": {"jpeg_ratio": 85},
            "02_jpeg_low": {"jpeg_ratio": 25},
            "03_blur_mild": {"gaussian_blur_r": 1},
            "04_blur_strong": {"gaussian_blur_r": 3},
            "05_noise_mild": {"gaussian_std": 0.02},
            "06_noise_strong": {"gaussian_std": 0.08},
            "07_resize_90": {"resize_ratio": 0.9},
            "08_resize_70": {"resize_ratio": 0.7},
            "09_bright_120": {"brightness_factor": 1.2},
            "10_bright_80": {"brightness_factor": 0.8},
        }
        
        # Apply each attack and save result
        for attack_name, attack_params in attacks.items():
            print(f"   ⚔️  Applying {attack_name}...")
            try:
                attacked_image = self.apply_attack_visual(base_image, attack_name, attack_params)
                attacked_image.save(self.output_dir / f"{attack_name}.png")
                print(f"      💾 Saved: {attack_name}.png")
                
                # Check if image actually changed
                base_array = np.array(base_image)
                attacked_array = np.array(attacked_image)
                
                # Calculate difference
                diff = np.mean(np.abs(base_array.astype(float) - attacked_array.astype(float)))
                print(f"      📊 Mean pixel difference: {diff:.2f}")
                
                if diff < 1.0:
                    print(f"      ⚠️  WARNING: Very small change detected!")
                    
            except Exception as e:
                print(f"      ❌ Error applying {attack_name}: {e}")
        
        print(f"\n✅ Visual verification complete! Check {self.output_dir} for results")
        print(f"   📁 Compare images to see if attacks are actually being applied")

if __name__ == "__main__":
    verifier = VisualAttackVerification()
    verifier.test_visual_attacks()

#!/usr/bin/env python3
"""
Clean-only watermark test to verify high accuracy on unattacked images
"""

import torch
import time
from pathlib import Path
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from watermark import Gaussian_Shading
from image_utils import *

class CleanOnlyTest:
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
        
        print(f"✅ Watermark thresholds:")
        print(f"   📊 tau_onebit: {self.watermark.tau_onebit:.6f}")
        print(f"   📊 tau_bits: {self.watermark.tau_bits:.6f}")
        
    def test_clean_detection(self, num_images=10):
        """Test watermark detection on clean (unattacked) images"""
        print(f"\n🧪 Testing {num_images} clean watermarked images...")
        
        prompt = "A beautiful landscape painting"
        text_embeddings = self.pipe.get_text_embedding('')
        
        accuracies = []
        detections = []
        
        for i in range(num_images):
            print(f"   🎨 Image {i+1}/{num_images}", end=" ")
            
            # Generate watermarked image
            set_random_seed(i)
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
            image_w = outputs.images[0]
            
            # Convert to tensor and get latents (NO ATTACKS)
            image_w_tensor = transform_img(image_w).unsqueeze(0)
            if self.device == 'cuda':
                image_w_tensor = image_w_tensor.to(torch.float16).to(self.device)
            else:
                image_w_tensor = image_w_tensor.to(torch.float32).to(self.device)
                
            image_latents = self.pipe.get_image_latents(image_w_tensor, sample=False)
            
            # Reverse diffusion
            reversed_latents = self.pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=20,
            )
            
            # Evaluate watermark
            accuracy = self.watermark.eval_watermark(reversed_latents)
            detected = accuracy >= self.watermark.tau_onebit
            
            accuracies.append(accuracy)
            detections.append(detected)
            
            print(f"✅ Acc: {accuracy:.4f}, Detected: {detected}")
        
        # Results
        avg_accuracy = sum(accuracies) / len(accuracies)
        detection_rate = sum(detections) / len(detections)
        
        print(f"\n📊 CLEAN IMAGE RESULTS:")
        print(f"   🎯 Average Accuracy: {avg_accuracy:.4f}")
        print(f"   ✅ Detection Rate: {detection_rate:.4f} ({sum(detections)}/{len(detections)})")
        print(f"   📏 Threshold: {self.watermark.tau_onebit:.4f}")
        
        return accuracies, detections

if __name__ == "__main__":
    test = CleanOnlyTest()
    test.test_clean_detection(10)

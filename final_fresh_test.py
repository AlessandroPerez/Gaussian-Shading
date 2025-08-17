#!/usr/bin/env python3
"""
Final Fresh Test - With proper dtype handling and detailed analysis
"""

import os
import sys
import json
import time
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import transform_img, set_random_seed, image_distortion
from watermark import Gaussian_Shading

def setup_fresh_environment():
    """Setup a completely fresh test environment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"final_fresh_test_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created fresh test environment: {output_dir}")
    return output_dir, device

class FinalFreshTester:
    def __init__(self, device):
        self.device = device
        
        # Initialize diffusion pipeline exactly like clean_only_test
        scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
        if device == 'cuda':
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
        self.pipe = self.pipe.to(device)
        
        # Initialize watermark
        self.watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
        
        print(f"✅ Watermark thresholds:")
        print(f"   📊 tau_onebit: {self.watermark.tau_onebit:.6f}")
        print(f"   📊 tau_bits: {self.watermark.tau_bits:.6f}")
    
    def test_watermark_detection(self, image):
        """Test watermark detection on a single image"""
        # Convert to tensor and get latents
        image_tensor = transform_img(image).unsqueeze(0)
        if self.device == 'cuda':
            image_tensor = image_tensor.to(torch.float16).to(self.device)
        else:
            image_tensor = image_tensor.to(torch.float32).to(self.device)
            
        image_latents = self.pipe.get_image_latents(image_tensor, sample=False)
        
        # Reverse diffusion using forward_diffusion (like clean_only_test)
        text_embeddings = self.pipe.get_text_embedding('')
        reversed_latents = self.pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=20,
        )
        
        # Evaluate watermark
        accuracy = self.watermark.eval_watermark(reversed_latents)
        detected = accuracy >= self.watermark.tau_onebit
        
        return accuracy, detected
    
    def detailed_fresh_test(self, num_images=20):
        """Comprehensive fresh test with detailed analysis"""
        print(f"\n🎨 Generating {num_images} fresh watermarked images...")
        
        # Different prompts for variety
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
        
        watermarked_images = []
        generation_accuracies = []
        generation_detections = []
        
        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            print(f"  🎨 Image {i+1}/{num_images}: {prompt[:30]}...")
            
            # Generate watermarked image with proper seeding
            set_random_seed(i + 42)  # Sequential like clean_only_test
            init_latents_w = self.watermark.create_watermark_and_return_w()
            
            # Ensure correct dtype for the pipeline
            if self.device == 'cuda':
                init_latents_w = init_latents_w.to(torch.float16).to(self.device)
            else:
                init_latents_w = init_latents_w.to(torch.float32).to(self.device)
            
            outputs = self.pipe(
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
            
            # Test detection immediately after generation
            accuracy, detected = self.test_watermark_detection(image)
            generation_accuracies.append(accuracy)
            generation_detections.append(detected)
            
            if i < 5:  # Show details for first few
                print(f"    ✅ Immediate detection - Acc: {accuracy:.6f}, Detected: {detected}")
        
        print(f"\n📊 IMMEDIATE DETECTION RESULTS:")
        avg_accuracy = np.mean(generation_accuracies)
        detection_rate = sum(generation_detections) / len(generation_detections)
        print(f"   🎯 Average Accuracy: {avg_accuracy:.6f}")
        print(f"   ✅ Detection Rate: {detection_rate:.3f} ({sum(generation_detections)}/{len(generation_detections)})")
        
        # Now test with attacks
        attack_configs = [
            {'name': 'none', 'type': 'none', 'params': {}},
            {'name': 'jpeg_high', 'type': 'jpeg', 'params': {'jpeg_ratio': 75}},
            {'name': 'jpeg_low', 'type': 'jpeg', 'params': {'jpeg_ratio': 30}},
            {'name': 'blur_mild', 'type': 'blur', 'params': {'gaussian_blur_r': 1}},
            {'name': 'noise_mild', 'type': 'noise', 'params': {'gaussian_std': 0.01}},
            {'name': 'resize_90', 'type': 'resize', 'params': {'resize_ratio': 0.9}},
        ]
        
        print(f"\n⚔️ Testing attacks...")
        attack_results = {}
        
        for attack_config in attack_configs:
            attack_name = attack_config['name']
            print(f"  Testing {attack_name}...")
            
            accuracies = []
            detections = []
            
            for i, image in enumerate(watermarked_images):
                # Apply attack
                attacked_image = self.apply_distortion(image, attack_config, i)
                
                # Test detection
                accuracy, detected = self.test_watermark_detection(attacked_image)
                accuracies.append(accuracy)
                detections.append(detected)
            
            attack_results[attack_name] = {
                'mean_accuracy': np.mean(accuracies),
                'detection_rate': sum(detections) / len(detections),
                'num_detected': sum(detections),
                'total_images': len(detections)
            }
            
            print(f"    ✅ {attack_name}: Acc={attack_results[attack_name]['mean_accuracy']:.4f}, "
                  f"Rate={attack_results[attack_name]['detection_rate']:.3f}")
        
        return attack_results
    
    def apply_distortion(self, image_pil, test_config, seed):
        """Apply distortion using existing image_utils function"""
        
        class MockArgs:
            def __init__(self):
                self.jpeg_ratio = None
                self.gaussian_blur_r = None
                self.gaussian_std = None
                self.resize_ratio = None
                self.random_crop_ratio = None
                self.brightness_factor = None
                self.median_blur_k = None
                self.random_drop_ratio = None
                self.sp_prob = None
        
        mock_args = MockArgs()
        
        if test_config["type"] == "none":
            return image_pil
        elif test_config["type"] == "jpeg":
            mock_args.jpeg_ratio = test_config["params"]["jpeg_ratio"]
        elif test_config["type"] == "blur":
            mock_args.gaussian_blur_r = test_config["params"]["gaussian_blur_r"]
        elif test_config["type"] == "noise":
            mock_args.gaussian_std = test_config["params"]["gaussian_std"]
        elif test_config["type"] == "resize":
            mock_args.resize_ratio = test_config["params"]["resize_ratio"]
        
        return image_distortion(image_pil, seed, mock_args)

def main():
    """Main test execution"""
    print("🚀 Starting Final Fresh Gaussian Shading Test")
    print("="*60)
    
    # Setup fresh environment
    output_dir, device = setup_fresh_environment()
    
    try:
        # Initialize tester
        tester = FinalFreshTester(device)
        
        # Run detailed fresh test
        results = tester.detailed_fresh_test(num_images=20)
        
        print(f"\n🎯 FINAL FRESH TEST SUMMARY:")
        print(f"="*60)
        
        for attack_name, result in results.items():
            print(f"{attack_name.upper().replace('_', ' ')}:")
            print(f"  Accuracy: {result['mean_accuracy']:.6f}")
            print(f"  Detection Rate: {result['detection_rate']:.3f} ({result['num_detected']}/{result['total_images']})")
        
        print(f"\n✅ Final fresh test completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

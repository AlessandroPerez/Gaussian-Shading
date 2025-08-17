#!/usr/bin/env python3
"""
Simple Fresh Test - Based on proven clean_only_test approach
Ensures completely new image generation and tests with minimal complexity.
"""

import os
import sys
import json
import time
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import io_utils
import image_utils
from watermark import Gaussian_Shading
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import transform_img, set_random_seed, image_distortion

def setup_fresh_environment():
    """Setup a completely fresh test environment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simple_fresh_test_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created fresh test environment: {output_dir}")
    return output_dir, device

class SimpleFreshTester:
    def __init__(self, device):
        self.device = device
        
        # Initialize diffusion pipeline
        scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", 
            scheduler=scheduler,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            revision="fp16" if device == 'cuda' else None,
            use_auth_token=None
        ).to(device)
        
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        
        # Initialize watermark
        self.watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
        
        print(f"✅ Watermark thresholds:")
        print(f"   📊 tau_onebit: {self.watermark.tau_onebit:.6f}")
        print(f"   📊 tau_bits: {self.watermark.tau_bits:.6f}")
    
    def generate_fresh_watermarked_images(self, num_images=30):
        """Generate fresh watermarked images"""
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
        
        text_embeddings = self.pipe.get_text_embedding('')
        watermarked_images = []
        
        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            print(f"  🎨 Image {i+1}/{num_images}: {prompt[:30]}...")
            
            # Generate watermarked image with unique seed
            set_random_seed(int(time.time() * 1000) % 100000 + i * 17)
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
            watermarked_images.append(outputs.images[0])
        
        print(f"✅ Generated {len(watermarked_images)} fresh watermarked images")
        return watermarked_images
    
    def test_watermark_detection(self, image):
        """Test watermark detection on a single image"""
        # Convert to tensor and get latents
        image_tensor = transform_img(image).unsqueeze(0)
        if self.device == 'cuda':
            image_tensor = image_tensor.to(torch.float16).to(self.device)
        else:
            image_tensor = image_tensor.to(torch.float32).to(self.device)
            
        image_latents = self.pipe.get_image_latents(image_tensor, sample=False)
        
        # Reverse diffusion (using forward_diffusion like clean_only_test)
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
    
    def apply_attacks_and_test(self, watermarked_images):
        """Apply attacks and test detection"""
        print(f"\n⚔️ Applying attacks and testing detection...")
        
        attack_configs = [
            {'name': 'none', 'type': 'none', 'params': {}},  # No attack baseline
            {'name': 'jpeg_high', 'type': 'jpeg', 'params': {'jpeg_ratio': 75}},
            {'name': 'jpeg_low', 'type': 'jpeg', 'params': {'jpeg_ratio': 30}},
            {'name': 'blur_mild', 'type': 'blur', 'params': {'gaussian_blur_r': 1}},
            {'name': 'blur_strong', 'type': 'blur', 'params': {'gaussian_blur_r': 3}},
            {'name': 'noise_mild', 'type': 'noise', 'params': {'gaussian_std': 0.01}},
            {'name': 'noise_strong', 'type': 'noise', 'params': {'gaussian_std': 0.05}},
            {'name': 'resize_90', 'type': 'resize', 'params': {'resize_ratio': 0.9}},
            {'name': 'resize_70', 'type': 'resize', 'params': {'resize_ratio': 0.7}},
            {'name': 'brightness_120', 'type': 'brightness', 'params': {'brightness_factor': 1.2}},
            {'name': 'brightness_80', 'type': 'brightness', 'params': {'brightness_factor': 0.8}},
        ]
        
        results = {}
        
        for attack_config in attack_configs:
            attack_name = attack_config['name']
            print(f"  Testing {attack_name}...")
            
            accuracies = []
            detections = []
            
            for i, image in enumerate(watermarked_images):
                if i % 10 == 0:
                    print(f"    Processing image {i+1}/{len(watermarked_images)}")
                
                # Apply attack using the image_distortion function
                attacked_image = self.apply_distortion(image, attack_config, i)
                
                # Test detection
                accuracy, detected = self.test_watermark_detection(attacked_image)
                accuracies.append(accuracy)
                detections.append(detected)
            
            results[attack_name] = {
                'accuracies': accuracies,
                'mean_accuracy': np.mean(accuracies),
                'detection_rate': sum(detections) / len(detections),
                'num_detected': sum(detections),
                'total_images': len(detections)
            }
            
            print(f"    ✅ {attack_name}: Acc={results[attack_name]['mean_accuracy']:.4f}, "
                  f"Rate={results[attack_name]['detection_rate']:.3f}")
        
        return results
    
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
        elif test_config["type"] == "brightness":
            mock_args.brightness_factor = test_config["params"]["brightness_factor"]
        
        return image_distortion(image_pil, seed, mock_args)
    
    def generate_clean_images_and_test(self, num_images=10):
        """Generate clean (non-watermarked) images for false positive testing"""
        print(f"\n🧽 Generating {num_images} clean images for false positive testing...")
        
        # Different prompts
        prompts = [
            "A beautiful mountain vista",
            "A modern building",
            "A flower field",
            "An abstract design",
            "A forest scene"
        ]
        
        accuracies = []
        detections = []
        
        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            print(f"  🧽 Clean image {i+1}/{num_images}: {prompt[:25]}...")
            
            # Generate clean image (no watermark)
            set_random_seed(50000 + i * 23)  # Different seed range
            outputs = self.pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                num_inference_steps=20,
                height=512,
                width=512,
            )
            clean_image = outputs.images[0]
            
            # Test detection
            accuracy, detected = self.test_watermark_detection(clean_image)
            accuracies.append(accuracy)
            detections.append(detected)
        
        results = {
            'accuracies': accuracies,
            'mean_accuracy': np.mean(accuracies),
            'detection_rate': sum(detections) / len(detections),
            'num_detected': sum(detections),
            'total_images': len(detections)
        }
        
        print(f"  ✅ Clean images: Acc={results['mean_accuracy']:.4f}, "
              f"Rate={results['detection_rate']:.3f}")
        
        return results

def display_results(watermarked_results, clean_results, threshold):
    """Display comprehensive results"""
    print("\n" + "="*80)
    print("🎯 SIMPLE FRESH TEST RESULTS")
    print("="*80)
    
    print(f"Detection Threshold (tau_onebit): {threshold:.6f}")
    
    print(f"\n📊 CLEAN IMAGES (should have LOW detection rate):")
    print(f"  Mean Accuracy: {clean_results['mean_accuracy']:.6f}")
    print(f"  Detection Rate: {clean_results['detection_rate']:.3f} ({clean_results['detection_rate']*100:.1f}%)")
    print(f"  False Positives: {clean_results['num_detected']}/{clean_results['total_images']}")
    
    print(f"\n🎨 WATERMARKED IMAGES:")
    
    # Baseline (no attack)
    baseline = watermarked_results['none']
    print(f"\nBASELINE (no attack):")
    print(f"  Mean Accuracy: {baseline['mean_accuracy']:.6f}")
    print(f"  Detection Rate: {baseline['detection_rate']:.3f} ({baseline['detection_rate']*100:.1f}%)")
    
    print(f"\n⚔️ ATTACK ROBUSTNESS:")
    
    for attack_name, results in watermarked_results.items():
        if attack_name == 'none':
            continue
            
        accuracy_drop = baseline['mean_accuracy'] - results['mean_accuracy']
        detection_drop = baseline['detection_rate'] - results['detection_rate']
        
        print(f"\n{attack_name.upper().replace('_', ' ')}:")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.6f} (drop: {accuracy_drop:.6f})")
        print(f"  Detection Rate: {results['detection_rate']:.3f} (drop: {detection_drop:.3f})")
        print(f"  Still Detected: {results['num_detected']}/{results['total_images']}")

def main():
    """Main test execution"""
    print("🚀 Starting Simple Fresh Gaussian Shading Test")
    print("="*60)
    
    # Setup fresh environment
    output_dir, device = setup_fresh_environment()
    
    try:
        # Initialize tester
        tester = SimpleFreshTester(device)
        
        # Generate fresh watermarked images
        watermarked_images = tester.generate_fresh_watermarked_images(num_images=30)
        
        # Test with attacks
        watermarked_results = tester.apply_attacks_and_test(watermarked_images)
        
        # Generate and test clean images
        clean_results = tester.generate_clean_images_and_test(num_images=10)
        
        # Display results
        display_results(watermarked_results, clean_results, tester.watermark.tau_onebit)
        
        # Save results
        all_results = {
            'watermarked': watermarked_results,
            'clean': clean_results,
            'threshold': tester.watermark.tau_onebit,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = f"{output_dir}/results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON
            serializable_results = {}
            for key, value in all_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict) and 'accuracies' in v:
                            serializable_results[key][k] = {
                                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                                for kk, vv in v.items()
                            }
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Simple fresh test completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

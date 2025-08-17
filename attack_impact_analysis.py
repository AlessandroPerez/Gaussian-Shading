#!/usr/bin/env python3
"""
Detailed attack impact analysis to verify if attacks destroy watermarks or if detection system has issues
"""

import torch
import time
from pathlib import Path
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from watermark import Gaussian_Shading
from image_utils import *
import numpy as np

class AttackImpactAnalysis:
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
        
    def apply_attack(self, image, attack_type, params):
        """Apply specific attack to image"""
        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # Set all attack parameters to None first
        args = MockArgs(
            jpeg_ratio=None,
            gaussian_blur_r=None,
            gaussian_std=None,
            resize_ratio=None,
            random_crop_ratio=None,
            random_drop_ratio=None,
            median_blur_k=None,
            sp_prob=None,
            brightness_factor=None
        )
        
        # Set specific attack parameter
        for param, value in params.items():
            setattr(args, param, value)
        
        # Apply attack using image_distortion
        seed = 42  # Fixed seed for reproducibility
        attacked_image = image_distortion(image, seed, args)
        
        return attacked_image
    
    def analyze_single_attack(self, attack_name, attack_config, num_tests=5):
        """Analyze impact of a single attack type"""
        print(f"\n🔍 ANALYZING: {attack_name}")
        print("-" * 40)
        
        prompt = "A beautiful landscape painting"
        text_embeddings = self.pipe.get_text_embedding('')
        
        results = {
            'clean_accuracies': [],
            'attacked_accuracies': [],
            'clean_detected': [],
            'attacked_detected': []
        }
        
        for i in range(num_tests):
            print(f"   🧪 Test {i+1}/{num_tests}")
            
            # Generate watermarked image
            set_random_seed(i + 100)  # Different seed range
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
            
            # Test 1: Clean image detection
            clean_accuracy, clean_detected = self.test_detection(image_w, text_embeddings)
            results['clean_accuracies'].append(clean_accuracy)
            results['clean_detected'].append(clean_detected)
            
            # Test 2: Attacked image detection
            if attack_config["type"] != "none":
                attacked_image = self.apply_attack(image_w, attack_config["type"], attack_config["params"])
                attacked_accuracy, attacked_detected = self.test_detection(attacked_image, text_embeddings)
                results['attacked_accuracies'].append(attacked_accuracy)
                results['attacked_detected'].append(attacked_detected)
            else:
                # For "none" attack, same as clean
                results['attacked_accuracies'].append(clean_accuracy)
                results['attacked_detected'].append(clean_detected)
            
            print(f"      📊 Clean: {clean_accuracy:.4f} ({'✅' if clean_detected else '❌'})")
            print(f"      ⚔️  Attack: {results['attacked_accuracies'][-1]:.4f} ({'✅' if results['attacked_detected'][-1] else '❌'})")
        
        # Calculate statistics
        clean_avg = np.mean(results['clean_accuracies'])
        attacked_avg = np.mean(results['attacked_accuracies'])
        clean_detection_rate = np.mean(results['clean_detected'])
        attacked_detection_rate = np.mean(results['attacked_detected'])
        accuracy_drop = clean_avg - attacked_avg
        detection_drop = clean_detection_rate - attacked_detection_rate
        
        print(f"\n   📈 RESULTS:")
        print(f"      🧹 Clean Avg Accuracy: {clean_avg:.4f} (Detection: {clean_detection_rate:.2f})")
        print(f"      ⚔️  Attacked Avg Accuracy: {attacked_avg:.4f} (Detection: {attacked_detection_rate:.2f})")
        print(f"      📉 Accuracy Drop: {accuracy_drop:.4f}")
        print(f"      📉 Detection Drop: {detection_drop:.2f}")
        
        return {
            'attack': attack_name,
            'clean_avg': clean_avg,
            'attacked_avg': attacked_avg,
            'accuracy_drop': accuracy_drop,
            'clean_detection_rate': clean_detection_rate,
            'attacked_detection_rate': attacked_detection_rate,
            'detection_drop': detection_drop
        }
    
    def test_detection(self, image, text_embeddings):
        """Test watermark detection on an image"""
        try:
            # Convert to tensor
            image_tensor = transform_img(image).unsqueeze(0)
            if self.device == 'cuda':
                image_tensor = image_tensor.to(torch.float16).to(self.device)
            else:
                image_tensor = image_tensor.to(torch.float32).to(self.device)
                
            # Get latents
            image_latents = self.pipe.get_image_latents(image_tensor, sample=False)
            
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
            
            return accuracy, detected
            
        except Exception as e:
            print(f"      ❌ Detection error: {e}")
            return 0.0, False
    
    def run_comprehensive_analysis(self):
        """Run comprehensive attack impact analysis"""
        print(f"\n🚀 ATTACK IMPACT ANALYSIS")
        print("=" * 50)
        
        # Define attacks to test
        attack_configs = {
            "clean": {"type": "none", "params": {}},
            "jpeg_mild": {"type": "jpeg", "params": {"jpeg_ratio": 85}},
            "jpeg_strong": {"type": "jpeg", "params": {"jpeg_ratio": 25}},
            "blur_mild": {"type": "blur", "params": {"gaussian_blur_r": 1}},
            "blur_strong": {"type": "blur", "params": {"gaussian_blur_r": 3}},
            "noise_mild": {"type": "noise", "params": {"gaussian_std": 0.02}},
            "noise_strong": {"type": "noise", "params": {"gaussian_std": 0.08}},
            "resize_mild": {"type": "resize", "params": {"resize_ratio": 0.9}},
            "resize_strong": {"type": "resize", "params": {"resize_ratio": 0.7}},
        }
        
        all_results = []
        
        for attack_name, attack_config in attack_configs.items():
            result = self.analyze_single_attack(attack_name, attack_config, num_tests=3)
            all_results.append(result)
        
        # Summary
        print(f"\n🏆 SUMMARY ANALYSIS")
        print("=" * 50)
        print(f"{'Attack':<15} {'Clean Acc':<10} {'Attack Acc':<11} {'Acc Drop':<9} {'Det Drop':<9}")
        print("-" * 55)
        
        for result in all_results:
            print(f"{result['attack']:<15} {result['clean_avg']:<10.4f} {result['attacked_avg']:<11.4f} "
                  f"{result['accuracy_drop']:<9.4f} {result['detection_drop']:<9.2f}")
        
        # Analysis conclusions
        print(f"\n🔍 ANALYSIS CONCLUSIONS:")
        
        # Check if clean detection is consistently high
        clean_results = [r for r in all_results if r['attack'] == 'clean'][0]
        if clean_results['clean_avg'] > 0.9:
            print(f"   ✅ Clean detection working perfectly: {clean_results['clean_avg']:.4f}")
        else:
            print(f"   ❌ ISSUE: Clean detection is low: {clean_results['clean_avg']:.4f}")
        
        # Check attack impact patterns
        attack_results = [r for r in all_results if r['attack'] != 'clean']
        significant_drops = [r for r in attack_results if r['accuracy_drop'] > 0.1]
        
        if len(significant_drops) > 0:
            print(f"   ⚔️  Attacks are destroying watermarks (as expected)")
            print(f"      📉 {len(significant_drops)}/{len(attack_results)} attacks cause >0.1 accuracy drop")
        else:
            print(f"   🤔 ISSUE: Attacks not significantly impacting watermarks")
        
        # Check for detection system issues
        detection_consistency = all([r['clean_detection_rate'] > 0.8 for r in all_results if r['attack'] == 'clean'])
        if detection_consistency:
            print(f"   ✅ Detection system is consistent and reliable")
        else:
            print(f"   ❌ ISSUE: Detection system showing inconsistency")

if __name__ == "__main__":
    analyzer = AttackImpactAnalysis()
    analyzer.run_comprehensive_analysis()

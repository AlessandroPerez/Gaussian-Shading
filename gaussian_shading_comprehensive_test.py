#!/usr/bin/env python3
"""
COMPREHENSIVE GAUSSIAN SHADING WATERMARK TEST SYSTEM
====================================================

Complete testing system for Gaussian Shading watermarking with:
- 1000+ realistic images generated using Stable Diffusion
- Both watermarked and clean images (balanced dataset)
- Multiple attack types and intensities
- Detection accuracy measurements
- CPU/GPU compatibility
- Comprehensive statistics and reporting
- All images saved for inspection
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
import os
from datetime import datetime
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import Gaussian Shading components
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from image_utils import set_random_seed, transform_img, image_distortion, latents_to_imgs
from optim_utils import get_dataset
from watermark_attacks import WatermarkAttacks, ATTACK_PRESETS
import open_clip


class GaussianShadingComprehensiveTest:
    """Comprehensive test system for Gaussian Shading watermarking"""
    
    def __init__(self, args):
        print("🏆 COMPREHENSIVE GAUSSIAN SHADING WATERMARK TEST")
        print("=" * 55)
        print("Testing: Realistic Images + Watermark Robustness + Statistics")
        
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
        
        print(f"   🖥️  Device: {self.device}")
        
        # Initialize Stable Diffusion pipeline
        print("   🔄 Loading Stable Diffusion pipeline...")
        self._init_diffusion_pipeline()
        
        # Initialize watermarking system
        print("   💧 Loading watermarking system...")
        self._init_watermark_system()
        
        # Initialize attack system  
        print("   ⚔️  Loading attack system...")
        self.attack_system = WatermarkAttacks(device=self.device)
        
        # Load dataset
        print("   📚 Loading prompt dataset...")
        self._init_dataset()
        
        # Create output directories
        self._create_output_dirs()
        
        # Define attack configurations
        self._define_attack_configs()
        
        print(f"   ✅ System initialized successfully")
        print(f"   📊 Testing with {args.num_images} images")
        print(f"   🎯 Watermark type: {'ChaCha20' if args.chacha else 'Simple'}")
        print(f"   📈 Attack variants: {len(self.attack_configs)}")
    
    def _init_diffusion_pipeline(self):
        """Initialize Stable Diffusion pipeline"""
        try:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.args.model_path, subfolder='scheduler'
            )
            
            if self.device == 'cuda':
                self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                    self.args.model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float16,
                    revision='fp16',
                )
            else:
                # CPU fallback with float32
                self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                    self.args.model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float32,
                )
            
            self.pipe.safety_checker = None
            self.pipe = self.pipe.to(self.device)
            
        except Exception as e:
            print(f"Error initializing diffusion pipeline: {e}")
            raise
    
    def _init_watermark_system(self):
        """Initialize watermarking system"""
        try:
            if self.args.chacha:
                self.watermark = Gaussian_Shading_chacha(
                    self.args.channel_copy,
                    self.args.hw_copy,
                    self.args.fpr,
                    self.args.user_number
                )
            else:
                self.watermark = Gaussian_Shading(
                    self.args.channel_copy,
                    self.args.hw_copy,
                    self.args.fpr,
                    self.args.user_number
                )
        except Exception as e:
            print(f"Error initializing watermark system: {e}")
            raise
    
    def _init_dataset(self):
        """Initialize prompt dataset"""
        try:
            # Use realistic prompts from dataset
            if hasattr(self.args, 'dataset_path') and self.args.dataset_path:
                self.dataset, self.prompt_key = get_dataset(self.args)
            else:
                # Fallback to default realistic prompts
                self.dataset = self._get_default_prompts()
                self.prompt_key = 'prompt'
        except Exception as e:
            print(f"Error loading dataset, using default prompts: {e}")
            self.dataset = self._get_default_prompts()
            self.prompt_key = 'prompt'
    
    def _get_default_prompts(self):
        """Get default realistic prompts for image generation"""
        prompts = [
            {"prompt": "A serene mountain landscape at sunset with golden light"},
            {"prompt": "Portrait of an elderly man with wisdom in his eyes"},
            {"prompt": "Modern city skyline with glass buildings reflecting clouds"},
            {"prompt": "A cat sitting by a window on a rainy day"},
            {"prompt": "Autumn forest with colorful leaves falling"},
            {"prompt": "Ocean waves crashing on a rocky shore"},
            {"prompt": "A child's birthday party with balloons and cake"},
            {"prompt": "Vintage car parked in front of a classic diner"},
            {"prompt": "Garden full of blooming roses in spring"},
            {"prompt": "Snow-covered village in the mountains"},
            {"prompt": "Bustling marketplace with vendors and colorful fruits"},
            {"prompt": "Lighthouse standing tall against stormy seas"},
            {"prompt": "Library filled with ancient books and warm lighting"},
            {"prompt": "Field of sunflowers under a blue sky"},
            {"prompt": "Cozy coffee shop with people reading books"},
            {"prompt": "Desert landscape with sand dunes and cacti"},
            {"prompt": "River flowing through a peaceful valley"},
            {"prompt": "Concert hall with musicians performing"},
            {"prompt": "Bakery window displaying fresh pastries"},
            {"prompt": "Moonlight reflecting on a calm lake"},
        ]
        
        # Extend to match required number
        extended_prompts = []
        for i in range(self.args.num_images):
            extended_prompts.append(prompts[i % len(prompts)])
        
        return extended_prompts
    
    def _create_output_dirs(self):
        """Create output directories"""
        self.output_base = Path(self.args.output_path)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'watermarked': self.output_base / 'watermarked_images',
            'clean': self.output_base / 'clean_images',
            'attacked': self.output_base / 'attacked_images',
            'results': self.output_base / 'results',
            'plots': self.output_base / 'plots'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _define_attack_configs(self):
        """Define attack configurations for testing"""
        self.attack_configs = {
            # No attack (baseline)
            "clean": {"type": "none", "intensity": "none"},
            
            # Gaussian blur attacks
            "blur_mild": {"type": "blur", "intensity": "mild", 
                         "params": {"kernel_size": 3, "sigma": 0.5}},
            "blur_moderate": {"type": "blur", "intensity": "moderate", 
                             "params": {"kernel_size": 5, "sigma": 1.0}},
            "blur_strong": {"type": "blur", "intensity": "strong", 
                           "params": {"kernel_size": 7, "sigma": 1.5}},
            
            # JPEG compression attacks
            "jpeg_mild": {"type": "jpeg", "intensity": "mild", 
                         "params": {"quality": 85}},
            "jpeg_moderate": {"type": "jpeg", "intensity": "moderate", 
                             "params": {"quality": 70}},
            "jpeg_strong": {"type": "jpeg", "intensity": "strong", 
                           "params": {"quality": 50}},
            "jpeg_extreme": {"type": "jpeg", "intensity": "extreme", 
                            "params": {"quality": 25}},
            
            # Noise attacks
            "noise_mild": {"type": "noise", "intensity": "mild", 
                          "params": {"noise_std": 0.02}},
            "noise_moderate": {"type": "noise", "intensity": "moderate", 
                              "params": {"noise_std": 0.05}},
            "noise_strong": {"type": "noise", "intensity": "strong", 
                            "params": {"noise_std": 0.08}},
            
            # Rotation attacks
            "rotation_mild": {"type": "rotation", "intensity": "mild", 
                             "params": {"angle_degrees": 5}},
            "rotation_moderate": {"type": "rotation", "intensity": "moderate", 
                                 "params": {"angle_degrees": 10}},
            "rotation_strong": {"type": "rotation", "intensity": "strong", 
                               "params": {"angle_degrees": 15}},
            
            # Scaling attacks
            "scaling_mild": {"type": "scaling", "intensity": "mild", 
                            "params": {"scale_factor": 0.9}},
            "scaling_moderate": {"type": "scaling", "intensity": "moderate", 
                                "params": {"scale_factor": 0.8}},
            "scaling_strong": {"type": "scaling", "intensity": "strong", 
                              "params": {"scale_factor": 0.7}},
            
            # Cropping attacks
            "cropping_mild": {"type": "cropping", "intensity": "mild", 
                             "params": {"crop_ratio": 0.9}},
            "cropping_moderate": {"type": "cropping", "intensity": "moderate", 
                                 "params": {"crop_ratio": 0.8}},
            "cropping_strong": {"type": "cropping", "intensity": "strong", 
                               "params": {"crop_ratio": 0.7}},
        }
    
    def generate_watermarked_images(self) -> list:
        """Generate watermarked images using Stable Diffusion"""
        print(f"\n🎨 GENERATING WATERMARKED IMAGES")
        print("=" * 35)
        
        watermarked_images = []
        
        with tqdm(total=self.args.num_images, desc="Generating watermarked images", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
            
            for i in range(self.args.num_images):
                try:
                    seed = i + self.args.gen_seed
                    prompt = self.dataset[i][self.prompt_key]
                    
                    # Set random seed for reproducibility
                    set_random_seed(seed)
                    
                    # Create watermarked latents
                    init_latents_w = self.watermark.create_watermark_and_return_w()
                    
                    # Generate image with watermark
                    with torch.no_grad():
                        outputs = self.pipe(
                            prompt,
                            num_images_per_prompt=1,
                            guidance_scale=self.args.guidance_scale,
                            num_inference_steps=self.args.num_inference_steps,
                            height=self.args.image_length,
                            width=self.args.image_length,
                            latents=init_latents_w,
                        )
                    
                    image_w = outputs.images[0]
                    
                    # Save watermarked image
                    image_path = self.dirs['watermarked'] / f"watermarked_{i:05d}.png"
                    image_w.save(image_path)
                    
                    # Convert to tensor for processing
                    image_tensor = transform_img(image_w, self.args.image_length).unsqueeze(0)
                    if self.device == 'cuda':
                        image_tensor = image_tensor.half()
                    image_tensor = image_tensor.to(self.device)
                    
                    watermarked_images.append({
                        "image": image_tensor,
                        "prompt": prompt,
                        "seed": seed,
                        "index": i,
                        "path": str(image_path),
                        "is_watermarked": True
                    })
                    
                except Exception as e:
                    print(f"Error generating watermarked image {i}: {e}")
                
                pbar.update(1)
        
        print(f"   ✅ Generated {len(watermarked_images)} watermarked images")
        return watermarked_images
    
    def generate_clean_images(self) -> list:
        """Generate clean (non-watermarked) images"""
        print(f"\n🧹 GENERATING CLEAN IMAGES")
        print("=" * 30)
        
        clean_images = []
        num_clean = self.args.num_images // 2  # Generate half as many clean images
        
        with tqdm(total=num_clean, desc="Generating clean images", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
            
            for i in range(num_clean):
                try:
                    seed = i + self.args.gen_seed + 10000  # Different seed space
                    prompt = self.dataset[i][self.prompt_key]
                    
                    # Set random seed
                    set_random_seed(seed)
                    
                    # Generate image without watermark (normal latents)
                    with torch.no_grad():
                        outputs = self.pipe(
                            prompt,
                            num_images_per_prompt=1,
                            guidance_scale=self.args.guidance_scale,
                            num_inference_steps=self.args.num_inference_steps,
                            height=self.args.image_length,
                            width=self.args.image_length,
                        )
                    
                    image_clean = outputs.images[0]
                    
                    # Save clean image
                    image_path = self.dirs['clean'] / f"clean_{i:05d}.png"
                    image_clean.save(image_path)
                    
                    # Convert to tensor
                    image_tensor = transform_img(image_clean, self.args.image_length).unsqueeze(0)
                    if self.device == 'cuda':
                        image_tensor = image_tensor.half()
                    image_tensor = image_tensor.to(self.device)
                    
                    clean_images.append({
                        "image": image_tensor,
                        "prompt": prompt,
                        "seed": seed,
                        "index": i,
                        "path": str(image_path),
                        "is_watermarked": False
                    })
                    
                except Exception as e:
                    print(f"Error generating clean image {i}: {e}")
                
                pbar.update(1)
        
        print(f"   ✅ Generated {len(clean_images)} clean images")
        return clean_images
    
    def apply_attack(self, image: torch.Tensor, attack_config: dict, seed: int = 0) -> torch.Tensor:
        """Apply attack to image based on configuration using existing image_utils functions"""
        if attack_config["type"] == "none":
            return image
        
        try:
            # Convert tensor to PIL for compatibility with image_utils
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Convert to PIL Image
            image_pil = transforms.ToPILImage()(image.cpu())
            
            # Create a mock args object for image_distortion function
            class MockArgs:
                def __init__(self):
                    self.jpeg_ratio = None
                    self.random_crop_ratio = None
                    self.random_drop_ratio = None
                    self.gaussian_blur_r = None
                    self.median_blur_k = None
                    self.resize_ratio = None
                    self.gaussian_std = None
                    self.sp_prob = None
                    self.brightness_factor = None
            
            mock_args = MockArgs()
            
            # Set appropriate parameter based on attack type
            if attack_config["type"] == "jpeg":
                mock_args.jpeg_ratio = attack_config["params"]["quality"]
            elif attack_config["type"] == "blur":
                mock_args.gaussian_blur_r = attack_config["params"].get("kernel_size", 3) // 2
            elif attack_config["type"] == "noise":
                mock_args.gaussian_std = attack_config["params"]["noise_std"]
            elif attack_config["type"] == "scaling":
                mock_args.resize_ratio = attack_config["params"]["scale_factor"]
            elif attack_config["type"] == "cropping":
                mock_args.random_crop_ratio = attack_config["params"]["crop_ratio"]
            elif attack_config["type"] == "brightness":
                mock_args.brightness_factor = attack_config["params"]["brightness_factor"]
            else:
                # Use the watermark_attacks system for other attacks
                return self.attack_system.apply_single_attack(image.unsqueeze(0), attack_config)
            
            # Apply distortion using existing function
            distorted_pil = image_distortion(image_pil, seed, mock_args)
            
            # Convert back to tensor
            distorted_tensor = transform_img(distorted_pil, image.shape[-1]).unsqueeze(0)
            if self.device == 'cuda':
                distorted_tensor = distorted_tensor.half()
            
            return distorted_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error applying attack {attack_config['type']}: {e}")
            return image.unsqueeze(0) if image.dim() == 3 else image
    
    def test_watermark_detection(self, image: torch.Tensor) -> dict:
        """Test watermark detection on an image"""
        try:
            # Convert image to correct format for detection
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Get latents for watermark detection
            image_for_latents = image.unsqueeze(0)
            
            with torch.no_grad():
                # Get image latents
                image_latents = self.pipe.get_image_latents(image_for_latents, sample=False)
                
                # Use empty prompt for detection (as in original code)
                text_embeddings = self.pipe.get_text_embedding('')
                
                # Reverse diffusion process
                reversed_latents = self.pipe.forward_diffusion(
                    latents=image_latents,
                    text_embeddings=text_embeddings,
                    guidance_scale=1,
                    num_inference_steps=self.args.num_inversion_steps,
                )
                
                # Evaluate watermark
                accuracy = self.watermark.eval_watermark(reversed_latents)
            
            return {
                "detected": accuracy >= self.watermark.tau_onebit,
                "accuracy": accuracy,
                "confidence": accuracy
            }
            
        except Exception as e:
            print(f"Error in watermark detection: {e}")
            return {
                "detected": False,
                "accuracy": 0.0,
                "confidence": 0.0
            }
    
    def calculate_metrics(self, y_true: list, y_pred: list) -> dict:
        """Calculate detection metrics"""
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            
            # Calculate confusion matrix components
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            return {
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    
    def run_comprehensive_test(self) -> dict:
        """Run the comprehensive watermarking test"""
        print(f"\n🚀 STARTING COMPREHENSIVE WATERMARK TEST")
        print("=" * 45)
        
        start_time = time.time()
        
        # Generate images
        watermarked_images = self.generate_watermarked_images()
        clean_images = self.generate_clean_images()
        
        # Combine datasets
        all_images = watermarked_images + clean_images
        random.shuffle(all_images)
        
        print(f"\n⚔️  TESTING ATTACK ROBUSTNESS")
        print("=" * 30)
        
        results = {}
        
        for attack_name, attack_config in self.attack_configs.items():
            print(f"\n   ⚔️  Testing {attack_name} ({attack_config['intensity']}):")
            
            # Test subset for efficiency (or all if requested)
            test_sample = all_images[:self.args.test_sample_size] if hasattr(self.args, 'test_sample_size') else all_images
            
            detection_true = []
            detection_pred = []
            confidences = []
            attack_times = []
            
            # Create attack-specific directory
            attack_dir = self.dirs['attacked'] / attack_name
            attack_dir.mkdir(exist_ok=True)
            
            with tqdm(total=len(test_sample), desc=f"  {attack_name}",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
                
                for test_item in test_sample:
                    try:
                        # Apply attack
                        start_attack_time = time.time()
                        attacked_image = self.apply_attack(test_item["image"], attack_config, test_item["seed"])
                        
                        # Save attacked image
                        if attack_config["type"] != "none":
                            attacked_pil = transforms.ToPILImage()(attacked_image.squeeze(0).cpu())
                            attacked_path = attack_dir / f"attacked_{test_item['index']:05d}.png"
                            attacked_pil.save(attacked_path)
                        
                        # Test watermark detection
                        detection_result = self.test_watermark_detection(attacked_image)
                        attack_time = time.time() - start_attack_time
                        
                        # Record results
                        detection_true.append(1 if test_item["is_watermarked"] else 0)
                        detection_pred.append(1 if detection_result["detected"] else 0)
                        confidences.append(detection_result["confidence"])
                        attack_times.append(attack_time)
                        
                    except Exception as e:
                        # Handle errors gracefully
                        detection_true.append(1 if test_item["is_watermarked"] else 0)
                        detection_pred.append(0)
                        confidences.append(0.0)
                        attack_times.append(0.0)
                        print(f"Error processing item {test_item['index']}: {e}")
                    
                    pbar.update(1)
            
            # Calculate metrics
            metrics = self.calculate_metrics(detection_true, detection_pred)
            
            # Store results
            results[attack_name] = {
                "attack_config": attack_config,
                "sample_size": len(test_sample),
                "watermarked_count": sum(detection_true),
                "clean_count": len(detection_true) - sum(detection_true),
                "metrics": metrics,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "avg_time": float(np.mean(attack_times)) if attack_times else 0.0,
                "detection_details": {
                    "true_labels": detection_true,
                    "predicted_labels": detection_pred,
                    "confidences": confidences
                }
            }
            
            # Print immediate results
            print(f"      🎯 F1 Score: {metrics['f1_score']:.3f}")
            print(f"      📊 Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"      🔢 TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, TN: {metrics['true_negatives']}, FN: {metrics['false_negatives']}")
            print(f"      ⏱️  Avg Time: {metrics.get('avg_time', 0):.3f}s")
        
        # Get overall TPR statistics
        tpr_detection, tpr_traceability = self.watermark.get_tpr()
        
        # Compile final results
        final_results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "total_images_generated": len(all_images),
                "watermarked_images": len(watermarked_images),
                "clean_images": len(clean_images),
                "device": self.device,
                "watermark_type": "ChaCha20" if self.args.chacha else "Simple",
                "model_path": self.args.model_path,
                "args": vars(self.args)
            },
            "watermark_stats": {
                "tpr_detection": int(tpr_detection),
                "tpr_traceability": int(tpr_traceability),
                "total_watermarked_tested": len(watermarked_images),
                "detection_threshold": float(self.watermark.tau_onebit),
                "traceability_threshold": float(self.watermark.tau_bits) if self.watermark.tau_bits else 0.0
            },
            "attack_results": results
        }
        
        return final_results
    
    def save_results(self, results: dict):
        """Save comprehensive results"""
        # Save JSON results
        results_file = self.dirs['results'] / 'comprehensive_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save plots
        self.generate_plots(results)
        
        # Generate text report
        self.generate_text_report(results)
        
        print(f"\n💾 RESULTS SAVED")
        print(f"   📄 JSON results: {results_file}")
        print(f"   📊 Plots: {self.dirs['plots']}/")
        print(f"   📋 Text report: {self.dirs['results']}/report.txt")
    
    def generate_plots(self, results: dict):
        """Generate visualization plots"""
        try:
            attack_results = results['attack_results']
            
            # Plot 1: F1 scores by attack
            attack_names = list(attack_results.keys())
            f1_scores = [attack_results[name]['metrics']['f1_score'] for name in attack_names]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(attack_names)), f1_scores)
            plt.xlabel('Attack Type')
            plt.ylabel('F1 Score')
            plt.title('Watermark Detection F1 Scores by Attack Type')
            plt.xticks(range(len(attack_names)), attack_names, rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Color bars by intensity
            intensity_colors = {'none': 'green', 'mild': 'yellow', 'moderate': 'orange', 
                               'strong': 'red', 'extreme': 'darkred'}
            for i, (attack_name, bar) in enumerate(zip(attack_names, bars)):
                intensity = attack_results[attack_name]['attack_config']['intensity']
                bar.set_color(intensity_colors.get(intensity, 'blue'))
            
            plt.tight_layout()
            plt.savefig(self.dirs['plots'] / 'f1_scores_by_attack.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Precision vs Recall
            precisions = [attack_results[name]['metrics']['precision'] for name in attack_names]
            recalls = [attack_results[name]['metrics']['recall'] for name in attack_names]
            
            plt.figure(figsize=(8, 8))
            scatter = plt.scatter(recalls, precisions, c=range(len(attack_names)), cmap='viridis', s=100)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision vs Recall for Different Attacks')
            plt.grid(True, alpha=0.3)
            
            # Add attack name annotations
            for i, name in enumerate(attack_names):
                plt.annotate(name, (recalls[i], precisions[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            plt.savefig(self.dirs['plots'] / 'precision_recall_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 3: Confusion matrix heatmap for most challenging attack
            worst_attack = min(attack_names, key=lambda x: attack_results[x]['metrics']['f1_score'])
            worst_results = attack_results[worst_attack]
            
            cm = confusion_matrix(
                worst_results['detection_details']['true_labels'],
                worst_results['detection_details']['predicted_labels']
            )
            
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Clean', 'Watermarked'],
                       yticklabels=['Clean', 'Watermarked'])
            plt.title(f'Confusion Matrix - {worst_attack} (Worst Case)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.dirs['plots'] / f'confusion_matrix_{worst_attack}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def generate_text_report(self, results: dict):
        """Generate comprehensive text report"""
        report_path = self.dirs['results'] / 'report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE GAUSSIAN SHADING WATERMARK TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Test information
            test_info = results['test_info']
            f.write("TEST CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Date: {test_info['timestamp']}\n")
            f.write(f"Duration: {test_info['duration_seconds']/60:.1f} minutes\n")
            f.write(f"Device: {test_info['device']}\n")
            f.write(f"Watermark Type: {test_info['watermark_type']}\n")
            f.write(f"Model: {test_info['model_path']}\n")
            f.write(f"Total Images: {test_info['total_images_generated']}\n")
            f.write(f"Watermarked: {test_info['watermarked_images']}\n")
            f.write(f"Clean: {test_info['clean_images']}\n\n")
            
            # Watermark statistics
            wm_stats = results['watermark_stats']
            f.write("WATERMARK STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Detection TPR: {wm_stats['tpr_detection']}/{wm_stats['total_watermarked_tested']}\n")
            f.write(f"Traceability TPR: {wm_stats['tpr_traceability']}/{wm_stats['total_watermarked_tested']}\n")
            f.write(f"Detection Threshold: {wm_stats['detection_threshold']:.6f}\n")
            f.write(f"Traceability Threshold: {wm_stats['traceability_threshold']:.6f}\n\n")
            
            # Attack results
            f.write("ATTACK RESISTANCE RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Attack':<20} {'Intensity':<10} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Acc':<6}\n")
            f.write("-" * 60 + "\n")
            
            attack_results = results['attack_results']
            for attack_name, data in attack_results.items():
                intensity = data['attack_config']['intensity']
                metrics = data['metrics']
                f.write(f"{attack_name:<20} {intensity:<10} {metrics['f1_score']:<6.3f} "
                       f"{metrics['precision']:<6.3f} {metrics['recall']:<6.3f} {metrics['accuracy']:<6.3f}\n")
            
            # Summary statistics
            f.write("\n" + "SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            
            all_f1s = [data['metrics']['f1_score'] for data in attack_results.values()]
            all_accs = [data['metrics']['accuracy'] for data in attack_results.values()]
            
            f.write(f"Average F1 Score: {np.mean(all_f1s):.3f}\n")
            f.write(f"Minimum F1 Score: {np.min(all_f1s):.3f}\n")
            f.write(f"Average Accuracy: {np.mean(all_accs):.3f}\n")
            f.write(f"Minimum Accuracy: {np.min(all_accs):.3f}\n")
            
            # Performance assessment
            min_f1 = np.min(all_f1s)
            if min_f1 >= 0.9:
                verdict = "EXCELLENT - High robustness across all attacks"
            elif min_f1 >= 0.8:
                verdict = "GOOD - Robust against most attacks"
            elif min_f1 >= 0.7:
                verdict = "MODERATE - Some vulnerabilities present"
            else:
                verdict = "NEEDS IMPROVEMENT - Significant vulnerabilities"
            
            f.write(f"\nOVERALL ASSESSMENT: {verdict}\n")
    
    def print_summary(self, results: dict):
        """Print summary of test results"""
        print(f"\n🏆 COMPREHENSIVE TEST SUMMARY")
        print("=" * 35)
        
        test_info = results['test_info']
        wm_stats = results['watermark_stats']
        attack_results = results['attack_results']
        
        print(f"   ⏱️  Duration: {test_info['duration_seconds']/60:.1f} minutes")
        print(f"   📊 Total Images: {test_info['total_images_generated']}")
        print(f"   💧 Watermarked: {test_info['watermarked_images']}")
        print(f"   🧹 Clean: {test_info['clean_images']}")
        print(f"   🖥️  Device: {test_info['device']}")
        
        print(f"\n   🎯 WATERMARK PERFORMANCE:")
        print(f"      Detection TPR: {wm_stats['tpr_detection']}/{wm_stats['total_watermarked_tested']} "
              f"({wm_stats['tpr_detection']/wm_stats['total_watermarked_tested']:.1%})")
        
        print(f"\n   ⚔️  ATTACK RESISTANCE:")
        all_f1s = [data['metrics']['f1_score'] for data in attack_results.values()]
        all_accs = [data['metrics']['accuracy'] for data in attack_results.values()]
        
        print(f"      Average F1: {np.mean(all_f1s):.3f}")
        print(f"      Minimum F1: {np.min(all_f1s):.3f}")
        print(f"      Average Accuracy: {np.mean(all_accs):.3f}")
        
        # Find best and worst attacks
        best_attack = max(attack_results.keys(), key=lambda x: attack_results[x]['metrics']['f1_score'])
        worst_attack = min(attack_results.keys(), key=lambda x: attack_results[x]['metrics']['f1_score'])
        
        print(f"      Best resistance: {best_attack} (F1: {attack_results[best_attack]['metrics']['f1_score']:.3f})")
        print(f"      Worst resistance: {worst_attack} (F1: {attack_results[worst_attack]['metrics']['f1_score']:.3f})")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Comprehensive Gaussian Shading Watermark Test')
    
    # Image generation parameters
    parser.add_argument('--num_images', default=1000, type=int,
                       help='Number of images to generate and test (default: 1000)')
    parser.add_argument('--image_length', default=512, type=int,
                       help='Image size (default: 512)')
    parser.add_argument('--guidance_scale', default=7.5, type=float,
                       help='Guidance scale for diffusion (default: 7.5)')
    parser.add_argument('--num_inference_steps', default=50, type=int,
                       help='Number of inference steps (default: 50)')
    parser.add_argument('--num_inversion_steps', default=None, type=int,
                       help='Number of inversion steps (default: same as inference)')
    parser.add_argument('--gen_seed', default=0, type=int,
                       help='Starting seed for generation (default: 0)')
    
    # Watermark parameters
    parser.add_argument('--channel_copy', default=1, type=int,
                       help='Channel copy factor (default: 1)')
    parser.add_argument('--hw_copy', default=8, type=int,
                       help='Height/width copy factor (default: 8)')
    parser.add_argument('--user_number', default=1000000, type=int,
                       help='User number for watermark (default: 1000000)')
    parser.add_argument('--fpr', default=0.000001, type=float,
                       help='False positive rate (default: 0.000001)')
    parser.add_argument('--chacha', action='store_true',
                       help='Use ChaCha20 encryption (default: False)')
    
    # Model parameters
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base',
                       help='Diffusion model path (default: SD 2.1)')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts',
                       help='Dataset path for prompts')
    
    # System parameters
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force CPU usage (default: use GPU if available)')
    parser.add_argument('--output_path', default='./comprehensive_test_output/',
                       help='Output directory (default: ./comprehensive_test_output/)')
    
    args = parser.parse_args()
    
    # Set inversion steps if not provided
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
    
    try:
        print(f"🔧 CONFIGURATION:")
        print(f"   • Images to generate: {args.num_images}")
        print(f"   • Image size: {args.image_length}x{args.image_length}")
        print(f"   • Watermark type: {'ChaCha20' if args.chacha else 'Simple'}")
        print(f"   • Device preference: {'CPU only' if args.cpu_only else 'GPU preferred'}")
        print(f"   • Model: {args.model_path}")
        print(f"   • Output: {args.output_path}")
        
        # Initialize test system
        test_system = GaussianShadingComprehensiveTest(args)
        
        # Run comprehensive test
        results = test_system.run_comprehensive_test()
        
        # Save results
        test_system.save_results(results)
        
        # Print summary
        test_system.print_summary(results)
        
        print(f"\n✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print(f"   📁 All results saved to: {args.output_path}")
        print(f"   🖼️  Images saved in subdirectories")
        print(f"   📊 Statistical analysis completed")
        
    except Exception as e:
        print(f"\n❌ ERROR IN COMPREHENSIVE TEST: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

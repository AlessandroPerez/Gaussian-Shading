#!/usr/bin/env python3
"""
Fresh Comprehensive Test - Ensures completely new image generation
This test explicitly generates new images each time to avoid any caching issues.
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import io_utils
import image_utils
import optim_utils
from watermark import Gaussian_Shading

def setup_fresh_environment():
    """Setup a completely fresh test environment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"fresh_test_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/clean_images", exist_ok=True)
    os.makedirs(f"{output_dir}/watermarked_images", exist_ok=True)
    os.makedirs(f"{output_dir}/attacked_images", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    print(f"Created fresh test environment: {output_dir}")
    return output_dir, device

def generate_fresh_prompts(num_images):
    """Generate diverse prompts to ensure image variety"""
    base_prompts = [
        "a serene mountain landscape at sunset",
        "a modern city skyline with glass buildings",
        "a colorful flower garden in spring",
        "an abstract geometric pattern",
        "a peaceful forest with tall trees",
        "a futuristic spacecraft in space",
        "a vintage car on a country road",
        "a cozy cottage with a garden",
        "a stormy ocean with large waves",
        "a desert scene with sand dunes"
    ]
    
    # Generate variations
    prompts = []
    for i in range(num_images):
        base = base_prompts[i % len(base_prompts)]
        if i >= len(base_prompts):
            variations = [
                f"{base} with dramatic lighting",
                f"{base} in watercolor style",
                f"{base} during golden hour",
                f"{base} with vibrant colors",
                f"{base} in minimalist style"
            ]
            prompts.append(variations[(i - len(base_prompts)) % len(variations)])
        else:
            prompts.append(base)
    
    return prompts

def fresh_image_generation(output_dir, device, num_images=30):
    """Generate completely fresh images"""
    print(f"\n🎨 Generating {num_images} fresh images...")
    
    # Setup components
    args = type('Args', (), {
        'stable_diffusion_version': '2-1',
        'seed': int(time.time()),  # Use timestamp for seed variety
        'w_channel': 1,
        'w_pattern': 8,
        'w_mask_shape': 'circle',
        'w_radius': 10,
        'w_measurement': 'l1_complex',
        'w_injection': 'complex',
        'w_pattern_const': 0.1,
        'reference_model': None,
        'reference_model_pretrain': None,
        'max_num_log_image': 100,
        'gen_seed': int(time.time()) % 1000000
    })()
    
    # Initialize watermark embedder
    watermark_embedder = WatermarkEmbedder(args).to(device)
    
    # Generate diverse prompts
    prompts = generate_fresh_prompts(num_images)
    
    clean_images = []
    watermarked_images = []
    
    for i, prompt in enumerate(prompts):
        print(f"  Generating image {i+1}/{num_images}: {prompt[:50]}...")
        
        # Generate with different seeds for variety
        seed = args.gen_seed + i * 13  # Prime number spacing for variety
        
        # Generate clean image
        clean_image = io_utils.generate_image(
            prompt=prompt,
            args=args,
            watermarked=False,
            seed=seed
        )
        
        # Generate watermarked image
        watermarked_image = io_utils.generate_image(
            prompt=prompt,
            args=args,
            watermarked=True,
            seed=seed
        )
        
        # Save images
        clean_path = f"{output_dir}/clean_images/clean_{i:03d}.png"
        watermarked_path = f"{output_dir}/watermarked_images/watermarked_{i:03d}.png"
        
        clean_image.save(clean_path)
        watermarked_image.save(watermarked_path)
        
        clean_images.append(clean_path)
        watermarked_images.append(watermarked_path)
    
    print(f"✅ Generated {len(clean_images)} clean and {len(watermarked_images)} watermarked images")
    return clean_images, watermarked_images

def apply_fresh_attacks(watermarked_images, output_dir):
    """Apply attacks to fresh watermarked images"""
    print("\n⚔️ Applying attacks to fresh watermarked images...")
    
    attack_configs = [
        {'name': 'jpeg_high', 'func': 'jpeg_compress', 'params': {'quality': 75}},
        {'name': 'jpeg_low', 'func': 'jpeg_compress', 'params': {'quality': 30}},
        {'name': 'blur_mild', 'func': 'gaussian_blur', 'params': {'radius': 1}},
        {'name': 'blur_strong', 'func': 'gaussian_blur', 'params': {'radius': 3}},
        {'name': 'noise_mild', 'func': 'gaussian_noise', 'params': {'std': 0.01}},
        {'name': 'noise_strong', 'func': 'gaussian_noise', 'params': {'std': 0.05}},
        {'name': 'resize_90', 'func': 'resize', 'params': {'scale': 0.9}},
        {'name': 'resize_70', 'func': 'resize', 'params': {'scale': 0.7}},
        {'name': 'crop_90', 'func': 'center_crop', 'params': {'scale': 0.9}},
        {'name': 'brightness_120', 'func': 'brightness', 'params': {'factor': 1.2}},
        {'name': 'brightness_80', 'func': 'brightness', 'params': {'factor': 0.8}},
    ]
    
    attacked_image_sets = {}
    
    for attack_config in attack_configs:
        attack_name = attack_config['name']
        attack_func = getattr(image_utils, attack_config['func'])
        attack_params = attack_config['params']
        
        print(f"  Applying {attack_name}...")
        
        attacked_images = []
        attack_dir = f"{output_dir}/attacked_images/{attack_name}"
        os.makedirs(attack_dir, exist_ok=True)
        
        for i, watermarked_path in enumerate(watermarked_images):
            # Load image
            image = Image.open(watermarked_path)
            
            # Apply attack
            attacked_image = attack_func(image, **attack_params)
            
            # Save attacked image
            attacked_path = f"{attack_dir}/attacked_{i:03d}.png"
            attacked_image.save(attacked_path)
            attacked_images.append(attacked_path)
        
        attacked_image_sets[attack_name] = attacked_images
        print(f"    ✅ Applied {attack_name} to {len(attacked_images)} images")
    
    return attacked_image_sets

def fresh_watermark_detection(clean_images, watermarked_images, attacked_image_sets, output_dir, device):
    """Perform watermark detection on fresh images"""
    print("\n🔍 Performing watermark detection on fresh images...")
    
    # Setup watermark detector
    args = type('Args', (), {
        'stable_diffusion_version': '2-1',
        'w_channel': 1,
        'w_pattern': 8,
        'w_mask_shape': 'circle',
        'w_radius': 10,
        'w_measurement': 'l1_complex',
        'w_injection': 'complex',
        'w_pattern_const': 0.1,
        'reference_model': None,
        'reference_model_pretrain': None
    })()
    
    watermark_detector = WatermarkDetector(args).to(device)
    
    results = {}
    
    # Test clean images (should have low detection rate)
    print("  Testing clean images...")
    clean_accuracies = []
    for i, clean_path in enumerate(clean_images):
        if i % 5 == 0:
            print(f"    Processing clean image {i+1}/{len(clean_images)}")
        
        image = Image.open(clean_path)
        accuracy = watermark_detector.detect_watermark(image)
        clean_accuracies.append(accuracy)
    
    results['clean'] = {
        'accuracies': clean_accuracies,
        'mean_accuracy': np.mean(clean_accuracies),
        'detection_rate': sum(1 for acc in clean_accuracies if acc > watermark_detector.tau_onebit) / len(clean_accuracies)
    }
    
    # Test watermarked images (should have high detection rate)
    print("  Testing watermarked images...")
    watermarked_accuracies = []
    for i, watermarked_path in enumerate(watermarked_images):
        if i % 5 == 0:
            print(f"    Processing watermarked image {i+1}/{len(watermarked_images)}")
        
        image = Image.open(watermarked_path)
        accuracy = watermark_detector.detect_watermark(image)
        watermarked_accuracies.append(accuracy)
    
    results['watermarked'] = {
        'accuracies': watermarked_accuracies,
        'mean_accuracy': np.mean(watermarked_accuracies),
        'detection_rate': sum(1 for acc in watermarked_accuracies if acc > watermark_detector.tau_onebit) / len(watermarked_accuracies)
    }
    
    # Test attacked images
    print("  Testing attacked images...")
    for attack_name, attacked_images in attacked_image_sets.items():
        print(f"    Testing {attack_name}...")
        
        attacked_accuracies = []
        for i, attacked_path in enumerate(attacked_images):
            image = Image.open(attacked_path)
            accuracy = watermark_detector.detect_watermark(image)
            attacked_accuracies.append(accuracy)
        
        results[attack_name] = {
            'accuracies': attacked_accuracies,
            'mean_accuracy': np.mean(attacked_accuracies),
            'detection_rate': sum(1 for acc in attacked_accuracies if acc > watermark_detector.tau_onebit) / len(attacked_accuracies)
        }
    
    # Calculate F1 scores for comprehensive evaluation
    print("  Calculating F1 scores...")
    f1_scores = {}
    
    for attack_name in attacked_image_sets.keys():
        # True positives: watermarked images detected as watermarked
        tp = sum(1 for acc in results['watermarked']['accuracies'] if acc > watermark_detector.tau_onebit)
        
        # False positives: clean images detected as watermarked
        fp = sum(1 for acc in results['clean']['accuracies'] if acc > watermark_detector.tau_onebit)
        
        # False negatives: attacked watermarked images not detected
        fn = sum(1 for acc in results[attack_name]['accuracies'] if acc <= watermark_detector.tau_onebit)
        
        # Calculate F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[attack_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    results['f1_scores'] = f1_scores
    results['detection_threshold'] = watermark_detector.tau_onebit
    
    return results

def save_results(results, output_dir):
    """Save comprehensive results"""
    results_file = f"{output_dir}/results/comprehensive_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            if 'accuracies' in value:
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")

def display_results(results):
    """Display comprehensive results"""
    print("\n" + "="*80)
    print("🎯 FRESH COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    threshold = results['detection_threshold']
    print(f"Detection Threshold (tau_onebit): {threshold:.6f}")
    
    print(f"\n📊 WATERMARK DETECTION PERFORMANCE:")
    print(f"Clean Images (should be LOW):")
    print(f"  Mean Accuracy: {results['clean']['mean_accuracy']:.6f}")
    print(f"  Detection Rate: {results['clean']['detection_rate']:.3f} ({results['clean']['detection_rate']*100:.1f}%)")
    
    print(f"\nWatermarked Images (should be HIGH):")
    print(f"  Mean Accuracy: {results['watermarked']['mean_accuracy']:.6f}")
    print(f"  Detection Rate: {results['watermarked']['detection_rate']:.3f} ({results['watermarked']['detection_rate']*100:.1f}%)")
    
    print(f"\n⚔️ ATTACK ROBUSTNESS ANALYSIS:")
    attack_names = [k for k in results.keys() if k not in ['clean', 'watermarked', 'f1_scores', 'detection_threshold']]
    
    for attack_name in sorted(attack_names):
        attack_results = results[attack_name]
        print(f"\n{attack_name.upper().replace('_', ' ')}:")
        print(f"  Mean Accuracy: {attack_results['mean_accuracy']:.6f}")
        print(f"  Detection Rate: {attack_results['detection_rate']:.3f} ({attack_results['detection_rate']*100:.1f}%)")
        print(f"  Accuracy Drop: {results['watermarked']['mean_accuracy'] - attack_results['mean_accuracy']:.6f}")
    
    print(f"\n🎯 F1 SCORE ANALYSIS:")
    for attack_name in sorted(attack_names):
        f1_data = results['f1_scores'][attack_name]
        print(f"\n{attack_name.upper().replace('_', ' ')}:")
        print(f"  Precision: {f1_data['precision']:.6f}")
        print(f"  Recall: {f1_data['recall']:.6f}")
        print(f"  F1 Score: {f1_data['f1']:.6f}")
        print(f"  TP/FP/FN: {f1_data['tp']}/{f1_data['fp']}/{f1_data['fn']}")

def main():
    """Main test execution"""
    print("🚀 Starting Fresh Comprehensive Gaussian Shading Test")
    print("="*60)
    
    # Setup fresh environment
    output_dir, device = setup_fresh_environment()
    
    try:
        # Generate fresh images
        clean_images, watermarked_images = fresh_image_generation(output_dir, device, num_images=30)
        
        # Apply attacks
        attacked_image_sets = apply_fresh_attacks(watermarked_images, output_dir)
        
        # Perform detection
        results = fresh_watermark_detection(clean_images, watermarked_images, attacked_image_sets, output_dir, device)
        
        # Save and display results
        save_results(results, output_dir)
        display_results(results)
        
        print(f"\n✅ Fresh comprehensive test completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

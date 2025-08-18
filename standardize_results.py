#!/usr/bin/env python3
"""
Standardize test_results.json for cross-system comparison
"""
import json
import sys
from pathlib import Path

# Mapping for attack names and intensities
ATTACK_NAME_MAP = {
    # JPEG
    'jpeg_high':   ('jpeg', 'mild', {'quality': 85}),
    'jpeg_medium': ('jpeg', 'moderate', {'quality': 70}),
    'jpeg_low':    ('jpeg', 'strong', {'quality': 50}),
    'jpeg_very_low': ('jpeg', 'extreme', {'quality': 25}),
    # Blur
    'blur_mild':   ('blur', 'mild', {'kernel_size': 3, 'sigma': 0.5}),
    'blur_moderate': ('blur', 'moderate', {'kernel_size': 5, 'sigma': 1.0}),
    'blur_strong': ('blur', 'strong', {'kernel_size': 7, 'sigma': 1.5}),
    # Noise
    'noise_mild':   ('awgn', 'mild', {'noise_std': 0.02}),
    'noise_moderate': ('awgn', 'moderate', {'noise_std': 0.05}),
    'noise_strong': ('awgn', 'extreme', {'noise_std': 0.08}),
    # Resize
    'resize_90': ('scaling', 'mild', {'scale_factor': 0.9}),
    'resize_80': ('scaling', 'moderate', {'scale_factor': 0.8}),
    'resize_70': ('scaling', 'strong', {'scale_factor': 0.7}),
    # Crop
    'crop_90': ('cropping', 'mild', {'crop_ratio': 0.9}),
    'crop_80': ('cropping', 'moderate', {'crop_ratio': 0.8}),
    'crop_70': ('cropping', 'strong', {'crop_ratio': 0.7}),
    # Brightness
    'bright_120': ('brightness', 'mild', {'brightness_factor': 1.2}),
    'bright_80': ('brightness', 'strong', {'brightness_factor': 0.8}),
    # Clean
    'clean': ('none', 'none', {}),
}

# Standard output order
STANDARD_ORDER = [
    'clean',
    'jpeg_mild', 'jpeg_moderate', 'jpeg_strong', 'jpeg_extreme',
    'blur_mild', 'blur_moderate', 'blur_strong',
    'noise_mild', 'noise_moderate', 'noise_strong', 'noise_extreme',
    'scaling_mild', 'scaling_moderate', 'scaling_strong',
    'cropping_mild', 'cropping_moderate', 'cropping_strong', 'cropping_extreme',
    'brightness_mild', 'brightness_strong',
]

# Standardize a single entry
def standardize_entry(name, entry):
    if name not in ATTACK_NAME_MAP:
        return None, None
    attack_type, intensity, params = ATTACK_NAME_MAP[name]
    std_name = f"{attack_type}_{intensity}"
    std_entry = dict(entry)
    std_entry['attack_config'] = {
        'type': attack_type,
        'intensity': intensity,
        'params': params
    }
    std_entry.pop('test_config', None)
    return std_name, std_entry

def main():
    if len(sys.argv) < 3:
        print("Usage: python standardize_results.py <input_json> <output_json>")
        sys.exit(1)
    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])
    with open(infile, 'r') as f:
        data = json.load(f)
    test_results = data.get('test_results', {})
    standardized = {}
    for name, entry in test_results.items():
        std_name, std_entry = standardize_entry(name, entry)
        if std_name:
            standardized[std_name] = std_entry
    # Copy metadata if present
    if 'test_info' in data:
        standardized['benchmark_info'] = data['test_info']
    if 'watermark_stats' in data:
        standardized['watermark_stats'] = data['watermark_stats']
    with open(outfile, 'w') as f:
        json.dump(standardized, f, indent=2)
    print(f"Standardized results written to {outfile}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify that all required dependencies are properly installed.
Run this after setting up the conda environment and installing requirements.txt
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def main():
    print("🧪 Testing Gaussian Shading Dependencies")
    print("=" * 50)
    
    # Test core packages
    print("\n📦 Core Packages:")
    success = True
    success &= test_import("torch", "PyTorch")
    success &= test_import("torchvision", "TorchVision") 
    success &= test_import("numpy", "NumPy")
    success &= test_import("scipy", "SciPy")
    success &= test_import("PIL", "Pillow")
    success &= test_import("matplotlib", "Matplotlib")
    
    # Test ML packages
    print("\n🤖 ML Packages:")
    success &= test_import("diffusers", "Diffusers")
    success &= test_import("transformers", "Transformers")
    success &= test_import("einops", "Einops")
    
    # Test utilities
    print("\n🛠️ Utilities:")
    success &= test_import("tqdm", "TQDM")
    success &= test_import("Crypto.Cipher", "PyCryptodome")
    success &= test_import("requests", "Requests")
    success &= test_import("yaml", "PyYAML")
    
    # Test GPU availability
    print("\n🖥️ GPU Check:")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
        else:
            print("⚠️ CUDA not available (CPU only)")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
    
    # Test Gaussian Shading specific imports
    print("\n🎯 Gaussian Shading Components:")
    try:
        from watermark import Gaussian_Shading
        print("✅ Watermark module")
    except Exception as e:
        print(f"❌ Watermark module: {e}")
        success = False
        
    try:
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        print("✅ Inverse Stable Diffusion")
    except Exception as e:
        print(f"❌ Inverse Stable Diffusion: {e}")
        success = False
    
    try:
        import image_utils
        print("✅ Image utilities")
    except Exception as e:
        print(f"❌ Image utilities: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All dependencies are properly installed!")
        print("You can now run:")
        print("   python simplified_gaussian_test.py --num_images 5")
    else:
        print("⚠️ Some dependencies are missing. Please check the installation.")
        print("Make sure you:")
        print("1. Activated the conda environment: conda activate gs")
        print("2. Installed PyTorch: conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("3. Installed requirements: pip install -r requirements.txt")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

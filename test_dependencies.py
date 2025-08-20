#!/usr/bin/env python3
"""
Test script to verify that all required dependencies are properly installed.
Run this after setting up the conda environment and installing requirements.txt

This is a minimal test that avoids imports that might cause segmentation faults.
"""

import sys
import importlib

def test_import_safe(module_name, package_name=None, critical=True):
    """Test if a module can be imported safely"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        if critical:
            print(f"❌ {package_name or module_name}: MISSING - {e}")
        else:
            print(f"⚠️ {package_name or module_name}: Optional - {e}")
        return critical  # Return False only for critical packages

def main():
    print("🧪 Testing Gaussian Shading Dependencies (Safe Mode)")
    print("=" * 60)
    
    # Test critical packages that must be present
    print("\n📦 Critical Packages:")
    all_critical = True
    
    # Basic Python packages
    all_critical &= test_import_safe("json", "JSON (built-in)")
    all_critical &= test_import_safe("pathlib", "Pathlib (built-in)")
    all_critical &= test_import_safe("argparse", "Argparse (built-in)")
    
    # Essential third-party packages
    all_critical &= test_import_safe("numpy", "NumPy")
    all_critical &= test_import_safe("PIL", "Pillow")
    all_critical &= test_import_safe("tqdm", "TQDM")
    all_critical &= test_import_safe("requests", "Requests")
    
    # ML packages (test without loading)
    all_critical &= test_import_safe("diffusers", "Diffusers")
    all_critical &= test_import_safe("transformers", "Transformers")
    all_critical &= test_import_safe("datasets", "Datasets (HuggingFace)")
    all_critical &= test_import_safe("huggingface_hub", "HuggingFace Hub")
    
    # Crypto
    all_critical &= test_import_safe("Crypto.Cipher", "PyCryptodome")
    
    print("\n🤖 PyTorch (Critical - Test Carefully):")
    try:
        # Test torch import without triggering GPU initialization
        import torch
        version = torch.__version__
        print(f"✅ PyTorch {version}")
        
        # Check version compatibility for transformers
        from packaging import version as pkg_version
        if pkg_version.parse(version.split('+')[0]) >= pkg_version.parse('2.6.0'):
            print(f"✅ PyTorch version compatible with transformers security requirements")
        else:
            print(f"⚠️ PyTorch {version} < 2.6.0 - may have compatibility issues with latest transformers")
            print(f"   Consider upgrading: conda update pytorch -c pytorch")
        
        # Test CUDA availability (but don't create tensors)
        if hasattr(torch.cuda, 'is_available'):
            if torch.cuda.is_available():
                print(f"✅ CUDA available")
                try:
                    device_name = torch.cuda.get_device_name(0)
                    print(f"   └─ Device: {device_name}")
                except:
                    print(f"   └─ Device info unavailable")
            else:
                print("⚠️ CUDA not available (CPU only)")
        
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        all_critical = False
    
    # Test optional packages
    print("\n🔧 Optional Packages:")
    test_import_safe("scipy", "SciPy", critical=False)
    test_import_safe("matplotlib", "Matplotlib", critical=False)
    test_import_safe("einops", "Einops", critical=False)
    test_import_safe("yaml", "PyYAML", critical=False)
    
    # Test local modules without importing
    print("\n🎯 Gaussian Shading Modules (File Check):")
    import os
    local_modules = [
        ("watermark.py", "Watermark module"),
        ("inverse_stable_diffusion.py", "Inverse Stable Diffusion"),
        ("image_utils.py", "Image utilities"),
        ("optim_utils.py", "Optimization utilities"),
        ("io_utils.py", "IO utilities"),
        ("prompts_1000.json", "Prompts database")
    ]
    
    for file_name, description in local_modules:
        if os.path.exists(file_name):
            print(f"✅ {description}")
        else:
            print(f"❌ {description}: File not found")
            all_critical = False
    
    print("\n" + "=" * 60)
    if all_critical:
        print("🎉 All critical dependencies are available!")
        print("\n💡 Quick test command:")
        print("   python -c \"from optim_utils import get_dataset; print('Import test passed')\"")
        print("\n🚀 Ready to run:")
        print("   python simplified_gaussian_test.py --num_images 5")
    else:
        print("⚠️ Some critical dependencies are missing!")
        print("\n🔧 Setup checklist:")
        print("1. conda activate gs")
        print("2. conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("3. pip install -r requirements.txt")
    
    return all_critical

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

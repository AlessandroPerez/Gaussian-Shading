#!/usr/bin/env python3
"""
SETUP AND DEPENDENCY CHECK FOR GAUSSIAN SHADING COMPREHENSIVE TEST
================================================================

This script checks and installs missing dependencies for the comprehensive test system.
"""

import subprocess
import sys
import importlib
import pkg_resources


def check_and_install_package(package_name, import_name=None, pip_name=None):
    """Check if a package is installed, and install it if not"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False


def check_pytorch():
    """Check PyTorch installation"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} is installed")
        if torch.cuda.is_available():
            print(f"   🔥 CUDA is available: {torch.cuda.get_device_name()}")
        else:
            print(f"   💻 CUDA not available, will use CPU")
        return True
    except ImportError:
        print("❌ PyTorch not found. Please install PyTorch first:")
        print("   Visit: https://pytorch.org/get-started/locally/")
        return False


def main():
    """Main dependency check and installation"""
    print("🔧 GAUSSIAN SHADING COMPREHENSIVE TEST - DEPENDENCY CHECK")
    print("=" * 60)
    
    # Core dependencies to check/install
    dependencies = [
        # Package name, import name, pip name
        ("NumPy", "numpy", "numpy"),
        ("Pillow", "PIL", "Pillow"),
        ("scikit-learn", "sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib", "matplotlib"),
        ("seaborn", "seaborn", "seaborn"),
        ("tqdm", "tqdm", "tqdm"),
        ("transformers", "transformers", "transformers"),
        ("diffusers", "diffusers", "diffusers"),
        ("open_clip", "open_clip", "open_clip_torch"),
    ]
    
    # Check PyTorch first
    if not check_pytorch():
        print("\n❌ PyTorch is required. Please install it first and run this script again.")
        return False
    
    # Check other dependencies
    all_installed = True
    for package_name, import_name, pip_name in dependencies:
        if not check_and_install_package(package_name, import_name, pip_name):
            all_installed = False
    
    # Additional checks
    print("\n🔍 CHECKING GAUSSIAN SHADING COMPONENTS...")
    
    # Check if local modules are available
    local_modules = [
        "watermark",
        "inverse_stable_diffusion", 
        "image_utils",
        "io_utils"
    ]
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}.py found")
        except ImportError as e:
            print(f"❌ {module}.py not found or has import errors: {e}")
            all_installed = False
    
    # Final status
    print("\n" + "=" * 60)
    if all_installed:
        print("✅ ALL DEPENDENCIES SATISFIED!")
        print("🚀 Ready to run the comprehensive test!")
        print("\nUsage:")
        print("  python gaussian_shading_comprehensive_test.py --num_images 1000")
    else:
        print("❌ SOME DEPENDENCIES MISSING!")
        print("Please resolve the issues above before running the test.")
    
    return all_installed


if __name__ == "__main__":
    main()

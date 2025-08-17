#!/usr/bin/env python3
"""
QUICK START - MINIMAL GAUSSIAN SHADING WATERMARK TEST
====================================================

This is a minimal test script that generates a small number of images
and tests basic watermark functionality without complex dependencies.
"""

import os
import sys
import time
from pathlib import Path


def check_basic_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking basic requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version}")
    
    # Check if we're in the right directory
    required_files = ['watermark.py', 'inverse_stable_diffusion.py', 'image_utils.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            print("   Make sure you're in the Gaussian-Shading directory")
            return False
    print("✅ Required files found")
    
    return True


def install_missing_packages():
    """Install missing packages"""
    print("\n📦 Checking and installing packages...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'diffusers',
        'transformers',
        'numpy',
        'Pillow',
        'tqdm',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - will install")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Installing {len(missing_packages)} missing packages...")
        import subprocess
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True


def run_quick_test():
    """Run a quick watermark test"""
    print("\n🚀 Running quick watermark test...")
    
    try:
        # Test basic watermark functionality
        from watermark import Gaussian_Shading
        from image_utils import set_random_seed
        
        print("   Initializing watermark system...")
        watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
        
        print("   Creating watermarked latents...")
        set_random_seed(42)
        latents = watermark.create_watermark_and_return_w()
        
        print("   Testing watermark detection...")
        accuracy = watermark.eval_watermark(latents)
        
        print(f"   ✅ Watermark accuracy: {accuracy:.6f}")
        
        if accuracy > 0.5:
            print("   ✅ Watermark system working correctly!")
            return True
        else:
            print("   ❌ Watermark system not working properly")
            return False
            
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False


def run_mini_test():
    """Run a mini version with actual image generation"""
    print("\n🎨 Running mini image generation test (this may take a few minutes)...")
    
    try:
        import subprocess
        import sys
        
        # Run the simplified test with minimal parameters
        cmd = [
            sys.executable, "simplified_gaussian_test.py",
            "--num_images", "5",  # Just 5 images
            "--image_length", "256",  # Smaller resolution
            "--output_path", "./quick_test_output/",
            "--num_inference_steps", "20"  # Faster generation
        ]
        
        print("   Command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Mini test completed successfully!")
            print("   📁 Check ./quick_test_output/ for results")
            return True
        else:
            print("   ❌ Mini test failed")
            print("   Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"   ❌ Mini test failed: {e}")
        return False


def main():
    """Main quick start function"""
    print("🚀 GAUSSIAN SHADING WATERMARK - QUICK START")
    print("=" * 50)
    print("This script will help you get started with the watermark test system.")
    print("")
    
    # Step 1: Check requirements
    if not check_basic_requirements():
        print("\n❌ Basic requirements not met. Please fix the issues above.")
        return False
    
    # Step 2: Install packages
    if not install_missing_packages():
        print("\n❌ Failed to install required packages.")
        print("💡 Try manually: pip install -r requirements.txt")
        return False
    
    # Step 3: Quick test
    if not run_quick_test():
        print("\n❌ Quick watermark test failed.")
        return False
    
    # Step 4: Ask user for mini test
    print("\n🎯 QUICK TEST PASSED!")
    print("\nWould you like to run a mini image generation test?")
    print("This will generate 5 small images to verify everything works.")
    print("(This may take 2-5 minutes depending on your hardware)")
    
    response = input("\nRun mini test? [y/N]: ").lower().strip()
    
    if response in ['y', 'yes']:
        if run_mini_test():
            print("\n🎉 ALL TESTS PASSED!")
            print("\n🎯 Next steps:")
            print("  1. Run full test: ./run_comprehensive_test.sh")
            print("  2. Or custom test: python simplified_gaussian_test.py --help")
            print("  3. Check results in: ./quick_test_output/")
        else:
            print("\n⚠️  Mini test failed, but basic functionality works.")
            print("   You can still try the full test.")
    else:
        print("\n✅ Basic setup complete!")
        print("\n🎯 To run the full test:")
        print("  ./run_comprehensive_test.sh")
        print("  or")
        print("  python simplified_gaussian_test.py --num_images 1000")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

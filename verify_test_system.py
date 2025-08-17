#!/usr/bin/env python3
"""
VERIFICATION SCRIPT FOR GAUSSIAN SHADING TEST SYSTEM
===================================================

This script verifies that all components are working correctly before running the full test.
"""

import sys
import traceback


def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🔥 CUDA available: {torch.cuda.get_device_name()}")
        else:
            print(f"   💻 CPU only")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow")
    except Exception as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        from diffusers import DPMSolverMultistepScheduler
        print(f"✅ Diffusers")
    except Exception as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    # Test local modules
    try:
        from watermark import Gaussian_Shading, Gaussian_Shading_chacha
        print(f"✅ Watermark module")
    except Exception as e:
        print(f"❌ Watermark module failed: {e}")
        return False
    
    try:
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        print(f"✅ Inverse Stable Diffusion")
    except Exception as e:
        print(f"❌ Inverse Stable Diffusion failed: {e}")
        return False
    
    try:
        from image_utils import set_random_seed, transform_img, image_distortion
        print(f"✅ Image utilities")
    except Exception as e:
        print(f"❌ Image utilities failed: {e}")
        return False
    
    try:
        from optim_utils import get_dataset
        print(f"✅ IO utilities")
    except Exception as e:
        print(f"❌ IO utilities failed: {e}")
        return False
    
    return True


def test_watermark_system():
    """Test watermark system initialization"""
    print("\n💧 Testing watermark system...")
    
    try:
        from watermark import Gaussian_Shading, Gaussian_Shading_chacha
        
        # Test simple watermark
        watermark_simple = Gaussian_Shading(1, 8, 0.000001, 1000000)
        print("✅ Simple Gaussian Shading initialized")
        
        # Test ChaCha20 watermark
        watermark_chacha = Gaussian_Shading_chacha(1, 8, 0.000001, 1000000)
        print("✅ ChaCha20 Gaussian Shading initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Watermark system test failed: {e}")
        traceback.print_exc()
        return False


def test_diffusion_pipeline():
    """Test diffusion pipeline initialization"""
    print("\n🎨 Testing diffusion pipeline...")
    
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = 'stabilityai/stable-diffusion-2-1-base'
        
        print(f"   Device: {device}")
        print(f"   Model: {model_path}")
        print("   Loading pipeline (this may take a while)...")
        
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
        
        if device == 'cuda':
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                model_path,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
            )
        else:
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                model_path,
                scheduler=scheduler,
                torch_dtype=torch.float32,
            )
        
        pipe.safety_checker = None
        pipe = pipe.to(device)
        
        print("✅ Diffusion pipeline loaded successfully")
        
        # Clean up to free memory
        del pipe
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusion pipeline test failed: {e}")
        traceback.print_exc()
        return False


def test_image_processing():
    """Test image processing utilities"""
    print("\n🖼️  Testing image processing...")
    
    try:
        import torch
        from PIL import Image
        import numpy as np
        from image_utils import set_random_seed, transform_img, image_distortion
        
        # Create test image
        test_image = Image.new('RGB', (512, 512), color='red')
        print("✅ Test image created")
        
        # Test transform
        transformed = transform_img(test_image, 512)
        print(f"✅ Image transform: {transformed.shape}")
        
        # Test distortion
        class MockArgs:
            def __init__(self):
                self.jpeg_ratio = 75
                self.gaussian_blur_r = None
                self.gaussian_std = None
                self.resize_ratio = None
                self.random_crop_ratio = None
                self.brightness_factor = None
                self.median_blur_k = None
                self.random_drop_ratio = None
                self.sp_prob = None
        
        mock_args = MockArgs()
        distorted = image_distortion(test_image, 42, mock_args)
        print("✅ Image distortion test passed")
        
        # Test random seed
        set_random_seed(42)
        print("✅ Random seed setting")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        traceback.print_exc()
        return False


def test_minimal_generation():
    """Test minimal image generation"""
    print("\n🎯 Testing minimal image generation...")
    
    try:
        import torch
        from watermark import Gaussian_Shading
        from image_utils import set_random_seed
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize watermark
        watermark = Gaussian_Shading(1, 8, 0.000001, 1000000)
        
        # Create watermarked latents
        set_random_seed(42)
        init_latents_w = watermark.create_watermark_and_return_w()
        print(f"✅ Watermarked latents created: {init_latents_w.shape}")
        
        # Test watermark evaluation
        accuracy = watermark.eval_watermark(init_latents_w)
        print(f"✅ Watermark evaluation: {accuracy:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal generation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("🧪 GAUSSIAN SHADING TEST SYSTEM VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Watermark System", test_watermark_system),
        ("Image Processing", test_image_processing),
        ("Minimal Generation", test_minimal_generation),
        ("Diffusion Pipeline", test_diffusion_pipeline),  # This one last as it's heavy
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 VERIFICATION SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Ready to run the comprehensive test.")
        print("\nNext steps:")
        print("  1. Run: ./run_comprehensive_test.sh")
        print("  2. Or: python simplified_gaussian_test.py --num_images 100")
    else:
        print("⚠️  Some tests failed. Please fix issues before running the full test.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Install PyTorch: https://pytorch.org/get-started/locally/")
        print("  • Check CUDA installation if using GPU")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

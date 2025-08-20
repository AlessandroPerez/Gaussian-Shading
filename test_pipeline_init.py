#!/usr/bin/env python3
"""
Quick test script to verify pipeline initialization works correctly.
This isolates the pipeline loading from the full test suite.
"""

import torch
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline

def test_pipeline_init():
    """Test if the pipeline can be initialized without errors"""
    print("🧪 Testing Pipeline Initialization")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        print("Loading scheduler...")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            'runwayml/stable-diffusion-v1-5', subfolder='scheduler'
        )
        print("✅ Scheduler loaded")
        
        print("Loading pipeline...")
        if device == 'cuda':
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                scheduler=scheduler,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                scheduler=scheduler,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
        
        print("✅ Pipeline loaded")
        
        print("Moving to device...")
        pipe = pipe.to(device)
        print("✅ Pipeline moved to device")
        
        print("\n🎉 Pipeline initialization successful!")
        print(f"Pipeline type: {type(pipe)}")
        print(f"Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_init()
    exit(0 if success else 1)

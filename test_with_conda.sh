#!/bin/bash
# Test wrapper script that ensures we're in the correct conda environment

echo "🔧 Activating gs conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs

echo "🔍 Checking environment..."
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

echo "🧪 Testing basic imports..."
python -c "
try:
    from watermark import Gaussian_Shading
    print('✅ Watermark import successful')
except Exception as e:
    print('❌ Watermark import failed:', e)
    exit(1)

try:
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    print('✅ InversableStableDiffusionPipeline import successful')
except Exception as e:
    print('❌ InversableStableDiffusionPipeline import failed:', e)
    exit(1)

try:
    from diffusers import DPMSolverMultistepScheduler
    print('✅ Diffusers import successful')
except Exception as e:
    print('❌ Diffusers import failed:', e)
    exit(1)

print('✅ All imports successful!')
"

if [ $? -eq 0 ]; then
    echo "🚀 Running simplified test with 5 images..."
    python simplified_gaussian_test.py --num_images 5 --image_length 256 --output_path ./test_output --num_inference_steps 20
else
    echo "❌ Import test failed. Cannot proceed with full test."
    exit 1
fi

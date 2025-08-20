#!/bin/bash
# fix_environment.sh - Comprehensive fix for Gaussian Shading environment issues

echo "🔧 Fixing Gaussian Shading Environment Issues"
echo "============================================="

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate gs

# Check if environment exists
if [ $? -ne 0 ]; then
    echo "❌ Error: conda environment 'gs' not found"
    echo "Please create it first with:"
    echo "conda create -n gs python=3.10 -y"
    echo "conda activate gs"
    exit 1
fi

echo "✅ Environment activated"

# Install/update compatible package versions
echo ""
echo "📦 Installing compatible package versions..."
pip install "diffusers>=0.21.0,<0.30.0" "transformers>=4.30.0,<4.40.0" accelerate --upgrade

# Check GPU availability
echo ""
echo "🖥️ Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️ CUDA not available - will run on CPU')
"

# Test pipeline initialization
echo ""
echo "🧪 Testing pipeline initialization..."
python test_pipeline_init.py

if [ $? -eq 0 ]; then
    echo "✅ Pipeline initialization successful!"
    echo ""
    echo "🚀 Ready to run tests:"
    echo "   python simplified_gaussian_test.py --num_images 5"
    echo ""
    echo "For A100 GPU (no memory flags needed):"
    echo "   python simplified_gaussian_test.py --num_images 1000"
    echo ""
    echo "For older/limited GPUs:"
    echo "   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python simplified_gaussian_test.py --num_images 1000"
    echo ""
    echo "For CPU only:"
    echo "   python simplified_gaussian_test.py --cpu_only --num_images 100"
else
    echo "❌ Pipeline initialization failed"
    echo "Try running in CPU mode:"
    echo "   python simplified_gaussian_test.py --cpu_only --num_images 1"
fi

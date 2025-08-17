#!/bin/bash
"""
RUN COMPREHENSIVE GAUSSIAN SHADING WATERMARK TEST
==================================================

This script runs the comprehensive watermarking test system.
"""

echo "🏆 GAUSSIAN SHADING WATERMARK COMPREHENSIVE TEST"
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python first."
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch; print('✅ PyTorch available')" 2>/dev/null || echo "❌ PyTorch not found"
python -c "import diffusers; print('✅ Diffusers available')" 2>/dev/null || echo "❌ Diffusers not found"
python -c "import PIL; print('✅ Pillow available')" 2>/dev/null || echo "❌ Pillow not found"

echo ""
echo "🚀 Starting comprehensive watermark test..."
echo "This will generate 1000+ realistic images and test watermark robustness."
echo ""

# Default parameters - can be modified
NUM_IMAGES=${1:-1000}
OUTPUT_DIR=${2:-"./comprehensive_test_results"}
WATERMARK_TYPE=${3:-"--chacha"}  # Use ChaCha20 encryption by default

echo "📋 Configuration:"
echo "   • Number of images: $NUM_IMAGES"
echo "   • Output directory: $OUTPUT_DIR" 
echo "   • Watermark type: $(echo $WATERMARK_TYPE | sed 's/--//')"
echo ""

# Run the simplified test (more stable)
echo "🔄 Running simplified test (recommended)..."
python simplified_gaussian_test.py \
    --num_images $NUM_IMAGES \
    --output_path "$OUTPUT_DIR/simple_test/" \
    $WATERMARK_TYPE \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR/simple_test/"
    echo "📊 Check the results directory for:"
    echo "   • Generated images (watermarked and clean)"
    echo "   • Attack robustness results"
    echo "   • Statistical analysis (JSON and text reports)"
    echo ""
    echo "📈 Key files:"
    echo "   • test_results.json - Complete numerical results"
    echo "   • report.txt - Human-readable summary"
    echo "   • watermarked_images/ - All watermarked images"
    echo "   • clean_images/ - All clean images"
    echo "   • attacked_images/ - Images after various attacks"
else
    echo ""
    echo "❌ Test failed. Check the error messages above."
    echo "💡 Common issues:"
    echo "   • Missing dependencies (run: pip install -r requirements.txt)"
    echo "   • Insufficient GPU memory (try --cpu_only flag)"
    echo "   • Model download issues (check internet connection)"
fi

echo ""
echo "🔧 For custom configurations, run:"
echo "   python simplified_gaussian_test.py --help"

# Comprehensive Gaussian Shading Watermark Test System

This repository contains a comprehensive testing system for the Gaussian Shading watermarking method, designed to evaluate watermark robustness with realistic images and various attacks.

## 🎯 Overview

The testing system provides:
- **1000+ realistic images** generated using Stable Diffusion
- **Balanced dataset**: 50% watermarked, 50% clean images
- **Multiple attack types** and intensities
- **CPU/GPU compatibility** with automatic fallback
- **Comprehensive statistics** and detailed reporting
- **All images saved** for manual inspection

## 🚀 Quick Start

### Prerequisites

1. **Install PyTorch** (with CUDA if you have a GPU):
   ```bash
   # For CUDA 11.8 (check your CUDA version)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Additional packages for the test system**:
   ```bash
   pip install scikit-learn matplotlib seaborn
   ```

### Run the Test

**Option 1: Quick test with default settings**
```bash
./run_comprehensive_test.sh
```

**Option 2: Custom configuration**
```bash
# Generate 500 images, save to custom directory, use simple watermark
./run_comprehensive_test.sh 500 "./my_test_results" ""

# Force CPU usage
python simplified_gaussian_test.py --num_images 1000 --cpu_only
```

**Option 3: Advanced comprehensive test**
```bash
python gaussian_shading_comprehensive_test.py --num_images 1000 --chacha
```

## 📊 Test Configuration

### Watermark Types
- **Simple**: Fast XOR-based watermarking (use no flags)
- **ChaCha20**: Cryptographically secure watermarking (use `--chacha`)

### Attack Types Tested
1. **JPEG Compression**: Quality 85, 70, 50, 25
2. **Gaussian Blur**: Radius 1, 2, 3 pixels
3. **Additive Noise**: Standard deviation 0.02, 0.05, 0.08
4. **Scaling**: 90%, 80%, 70% of original size
5. **Random Cropping**: 90%, 80%, 70% of original area
6. **Brightness**: 120%, 80% of original brightness

### Default Parameters
- **Images**: 1000 total (balanced 50/50 watermarked/clean)
- **Resolution**: 512x512 pixels
- **Model**: Stable Diffusion 2.1 Base
- **Guidance Scale**: 7.5
- **Inference Steps**: 50

## 📁 Output Structure

```
output_directory/
├── watermarked_images/     # All watermarked images
│   ├── watermarked_00000.png
│   ├── watermarked_00001.png
│   └── ...
├── clean_images/           # All clean (non-watermarked) images
│   ├── clean_00000.png
│   ├── clean_00001.png
│   └── ...
├── attacked_images/        # Images after attacks (organized by attack type)
│   ├── jpeg_low/
│   ├── blur_strong/
│   └── ...
└── results/
    ├── test_results.json   # Complete numerical results
    └── report.txt          # Human-readable summary
```

## 📈 Metrics and Analysis

### Key Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Accuracy**: (True positives + True negatives) / Total samples

### Interpretation
- **F1 > 0.9**: Excellent robustness
- **F1 > 0.8**: Good robustness
- **F1 > 0.7**: Moderate robustness
- **F1 < 0.7**: Needs improvement

## 🔧 Advanced Usage

### Custom Attack Configuration

Modify `_define_test_configs()` in the test script to add custom attacks:

```python
"custom_attack": {
    "type": "jpeg", 
    "params": {"jpeg_ratio": 30}
}
```

### Memory Optimization

For limited GPU memory:
```bash
python simplified_gaussian_test.py --num_images 100 --image_length 256
```

For CPU-only execution:
```bash
python simplified_gaussian_test.py --cpu_only
```

### Using Different Models

```bash
python simplified_gaussian_test.py --model_path "runwayml/stable-diffusion-v1-5"
```

## 📋 Command Line Options

```bash
python simplified_gaussian_test.py --help
```

Key options:
- `--num_images`: Number of images to generate (default: 1000)
- `--image_length`: Image resolution (default: 512)
- `--chacha`: Use ChaCha20 encryption (more secure)
- `--cpu_only`: Force CPU usage
- `--output_path`: Output directory
- `--channel_copy`: Watermark channel factor (default: 1)
- `--hw_copy`: Watermark spatial factor (default: 8)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```bash
   python simplified_gaussian_test.py --cpu_only
   # or reduce image size
   python simplified_gaussian_test.py --image_length 256
   ```

2. **Model download fails**:
   - Check internet connection
   - Try a different model: `--model_path "runwayml/stable-diffusion-v1-5"`

3. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install scikit-learn matplotlib seaborn
   ```

4. **Permission errors**:
   ```bash
   sudo chmod +x run_comprehensive_test.sh
   ```

### Performance Tips

- **GPU**: Use CUDA for faster generation (~10-20x speedup)
- **Batch size**: Larger `--num_images` is more efficient
- **Resolution**: Lower `--image_length` reduces memory usage
- **Attacks**: Modify test configs to focus on specific attacks

## 📊 Expected Results

### Typical Performance (ChaCha20 watermark)
- **Clean images**: F1 ≈ 0.95-0.99
- **JPEG (quality 50)**: F1 ≈ 0.85-0.95
- **Gaussian blur**: F1 ≈ 0.80-0.90
- **Strong attacks**: F1 ≈ 0.70-0.85

### Runtime Estimates
- **GPU (RTX 3080)**: ~2-3 hours for 1000 images
- **CPU**: ~8-12 hours for 1000 images
- **Memory**: ~8-16GB GPU memory, ~16-32GB RAM

## 🔬 Research Applications

This test system is suitable for:
- **Watermark robustness evaluation**
- **Attack resistance analysis**
- **Comparative studies** between watermarking methods
- **Performance benchmarking** across different hardware
- **Academic research** and publication

## 📚 References

Based on the paper:
```
@article{yang2024gaussian,
  title={Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models}, 
  author={Yang, Zijin and Zeng, Kai and Chen, Kejiang and Fang, Han and Zhang, Weiming and Yu, Nenghai},
  journal={arXiv preprint arXiv:2404.04956},
  year={2024},
}
```

## 🤝 Contributing

To extend the test system:
1. Add new attack methods in `watermark_attacks.py`
2. Modify test configurations in the main script
3. Update metrics calculation if needed
4. Submit a pull request with your improvements

## 📄 License

This project follows the same license as the original Gaussian Shading repository.

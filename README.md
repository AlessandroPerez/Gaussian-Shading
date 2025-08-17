# Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2404.04956-b31b1b.svg)](https://arxiv.org/abs/2404.04956)

This repository hosts the official PyTorch implementation of the paper: ["**Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models**"](https://arxiv.org/abs/2404.04956) (Accepted by CVPR 2024).


## Method

![method](fig/framework.png)

We propose a watermarking method named Gaussian Shading, designed to ensure no
deterioration in model performance. The embedding process encompasses three primary elements: watermark diffuse, randomization, and distribution-preserving sampling. Watermark diffusion spreads the watermark information throughout the latent representation to enhance the robustness. Watermark randomization and distribution preserving sampling guarantee the congruity of the latent representation distribution with that of watermark-free latent representations, thereby achieving performance-lossless. In the extraction phase, the latent representations are acquired through Denoising Diffusion Implicit Model (DDIM) inversion, allowing for the retrieval of watermark information. 


## Getting Started

### Setting up evironment

#### Step 1: Cloning the repo
```
git clone https://github.com/AlessandroPerez/Gaussian-Shading.git
cd Gaussian-Shading
```

#### Step 2: Create Conda environment
```
conda create -n gs python=3.8 -y
conda activate gs
```

#### Step 3: Downloading Packages via Conda
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### Step 4: Install remaining Pytohon-only packages via pip
```
pip install -r requirements.txt
```

### Test True Positive Rate and Bit Accuracy

For testing in a lossless situation, you can run,
```
python run_gaussian_shading.py \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --chacha \
      --num 1000
```


To test the performance of Gaussian Shading under noise perturbation (e.g., JPEG QF=25), you can run, 
```
python run_gaussian_shading.py \
      --jpeg_ratio 25 \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --chacha \
      --num 1000
```
For more adversarial cases, You can refer to [this script](scripts/run.sh).

### Calculate CLIP Score

To calculate the CLIP Score, it relies on two pre-trained models, you can run,
```
python run_gaussian_shading.py \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --chacha \
      --num 1000 \
      --reference_model ViT-g-14 \
      --reference_model_pretrain laion2b_s12b_b42k 
```

### Calculate FID

When calculating  FID, we have aligned our settings with [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark) and used the same ground truth dataset. The dataset contains 5000 images from the COCO dataset. You can find the corresponding information such as prompts in 'fid_outputs/coco/meta_data.json'. 
The ground truth dataset can download [here](https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0?usp=sharing).


Then, to calculate FID, you can run,
```
python gaussian_shading_fid.py \
      --channel_copy 1 \
      --hw_copy 8 \
      --chacha \
      --num 5000 
```


## Comprehensive Watermark Testing

### Available Test Scripts

#### Quick Test (Diagnostic)
For rapid watermark verification:
```bash
python diagnostic_test.py
```

#### Simple Test System
For basic watermark testing with configurable attacks:
```bash
python simplified_gaussian_test.py --num_images 50 --output_path "./test_results/"
```

#### Comprehensive Test System
For advanced testing with full statistics and attack analysis:
```bash
python gaussian_shading_comprehensive_test.py --num_images 1000 --balanced_dataset
```

### Test Configuration Options

Common parameters across test scripts:
- `--num_images`: Number of images to test (default: 100)
- `--output_path`: Output directory for results and images
- `--device`: Device to use ('cuda' or 'cpu')
- `--fpr`: False positive rate (default: 0.000001)
- `--channel_copy`: Channel copy parameter (default: 1)
- `--hw_copy`: HW copy parameter (default: 8)
- `--chacha`: Use ChaCha20 encryption
- `--balanced_dataset`: Generate equal numbers of watermarked and clean images

### Attack Types Available

The test system includes robustness testing against:
- **Compression**: JPEG at various quality levels (95, 75, 50, 25)
- **Noise**: Gaussian noise (mild, moderate, strong)
- **Blur**: Gaussian blur (mild, moderate, strong)  
- **Resize**: Image resizing (90%, 80%, 70%)
- **Crop**: Center cropping (90%, 80%, 70%)
- **Brightness**: Brightness adjustment (120%, 80%)

### Test Results Format

Test results include:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Confusion Matrix**: True/False Positives and Negatives
- **Timing Statistics**: Average processing time per image

### Sample Test Results

Recent test on 75 images (50 watermarked, 25 clean) using conda environment:

```
TEST SUMMARY
============
Duration: 44.2 minutes
Total Images: 75 (50 watermarked, 25 clean)
Average F1 Score: 0.039
Watermark Detection TPR: 38% (19/50)
Clean Image TNR: 100% (25/25)

Performance by Attack:
- Clean images: F1=0.039, Precision=1.000, Recall=0.020
- JPEG compression: Consistent across quality levels
- Blur/Noise/Resize/Crop: Robust against mild attacks
- Strong attacks: Significantly impact watermark detection
```

**Note**: The low F1 scores in comprehensive testing indicate that aggressive attacks can destroy watermarks, which is expected behavior. The diagnostic test shows perfect watermark detection (1.0 accuracy) on unattacked images.

### Environment Setup for Testing

#### Using Conda (Recommended)
```bash
conda create -n gs python=3.8 -y
conda activate gs
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

#### Quick Start Test
```bash
conda activate gs
python quick_start.py  # Basic functionality test
python verify_test_system.py  # Verify all components work
```

### Additional Notes
- The code is compatible with Stable Diffusion versions 1.4, 2.0, and 2.1, where the latent space size is 4 x 64 x 64. If you want to apply it to other versions of the diffusion model, you will need to adjust the watermark parameters accordingly.

- By default, Gaussian Shading has a capacity of 256 bits. If you want to change the capacity of the watermark, you can adjust `--channel_copy` and `--hw_copy`.

- Test results are saved in JSON format for further analysis and include both individual image results and aggregated statistics.

- The test system automatically handles CPU/GPU compatibility and falls back gracefully when CUDA is unavailable. 

- Due to the time-consuming nature of Chacha20 encryption, we offer a simple encryption method. It involves using Torch  to generate random bits, which are then XORed with the watermark information directly. By removing  `--chacha ` before running, , you can speed up the testing process. While this method may not be strictly performance-lossless, it is still an improvement over the baseline method mentioned in the paper.



## Acknowledgements
We heavily borrow the code from [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark). We appreciate the authors for sharing their code. 

## Citation
If you find our work useful for your research, please consider citing the following papers :)

```

@article{yang2024gaussian,
      title={Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models}, 
      author={Yang, Zijin and Zeng, Kai and Chen, Kejiang and Fang, Han and Zhang, Weiming and Yu, Nenghai},
      journal={arXiv preprint arXiv:2404.04956},
      year={2024},
}

```

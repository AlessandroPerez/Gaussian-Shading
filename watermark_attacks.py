#!/usr/bin/env python3
"""
WATERMARK ATTACK SYSTEM
======================

Attack system for testing watermark robustness with various image processing operations.
Adapted for the Gaussian Shading watermarking system.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import io


class WatermarkAttacks:
    """Attack system for testing watermark robustness"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def gaussian_blur(self, image: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur attack"""
        try:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # Create Gaussian kernel
            kernel = self._gaussian_kernel_2d(kernel_size, sigma).to(image.device)
            kernel = kernel.repeat(image.shape[1], 1, 1, 1)
            
            # Apply convolution
            blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
            return torch.clamp(blurred, 0, 1)
        except Exception as e:
            print(f"Error in gaussian_blur: {e}")
            return image
    
    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Generate 2D Gaussian kernel"""
        kernel_1d = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.unsqueeze(0).unsqueeze(0)
    
    def jpeg_compression(self, image: torch.Tensor, quality: int = 75) -> torch.Tensor:
        """Apply JPEG compression attack"""
        try:
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Convert to PIL Image
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Apply JPEG compression
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            
            # Convert back to tensor
            compressed_np = np.array(compressed_image).astype(np.float32) / 255.0
            compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1).unsqueeze(0)
            
            return compressed_tensor.to(image.device)
        except Exception as e:
            print(f"Error in jpeg_compression: {e}")
            return image.unsqueeze(0) if image.dim() == 3 else image
    
    def additive_white_gaussian_noise(self, image: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
        """Apply additive white Gaussian noise"""
        try:
            noise = torch.randn_like(image) * noise_std
            noisy_image = image + noise
            return torch.clamp(noisy_image, 0, 1)
        except Exception as e:
            print(f"Error in additive_white_gaussian_noise: {e}")
            return image
    
    def rotation(self, image: torch.Tensor, angle_degrees: float = 10) -> torch.Tensor:
        """Apply rotation attack"""
        try:
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Convert to PIL, rotate, and convert back
            image_pil = TF.to_pil_image(image.cpu())
            rotated_pil = TF.rotate(image_pil, angle_degrees, fill=128)  # Fill with gray
            rotated_tensor = TF.to_tensor(rotated_pil).unsqueeze(0)
            
            return rotated_tensor.to(image.device)
        except Exception as e:
            print(f"Error in rotation: {e}")
            return image.unsqueeze(0) if image.dim() == 3 else image
    
    def scaling(self, image: torch.Tensor, scale_factor: float = 0.8) -> torch.Tensor:
        """Apply scaling attack"""
        try:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            _, _, h, w = image.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Scale down and then back up
            scaled_down = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
            scaled_back = F.interpolate(scaled_down, size=(h, w), mode='bilinear', align_corners=False)
            
            return torch.clamp(scaled_back, 0, 1)
        except Exception as e:
            print(f"Error in scaling: {e}")
            return image
    
    def center_crop(self, image: torch.Tensor, crop_ratio: float = 0.8) -> torch.Tensor:
        """Apply center cropping attack"""
        try:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            _, _, h, w = image.shape
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            
            # Calculate cropping coordinates
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            
            # Crop and resize back
            cropped = image[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
            resized = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
            
            return torch.clamp(resized, 0, 1)
        except Exception as e:
            print(f"Error in center_crop: {e}")
            return image
    
    def sharpening(self, image: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Apply sharpening attack"""
        try:
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Convert to PIL and apply sharpening
            image_pil = TF.to_pil_image(image.cpu())
            enhancer = ImageEnhance.Sharpness(image_pil)
            sharpened_pil = enhancer.enhance(1.0 + strength)
            sharpened_tensor = TF.to_tensor(sharpened_pil).unsqueeze(0)
            
            return sharpened_tensor.to(image.device)
        except Exception as e:
            print(f"Error in sharpening: {e}")
            return image.unsqueeze(0) if image.dim() == 3 else image
    
    def brightness_adjustment(self, image: torch.Tensor, brightness_factor: float = 1.2) -> torch.Tensor:
        """Apply brightness adjustment"""
        try:
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Convert to PIL and adjust brightness
            image_pil = TF.to_pil_image(image.cpu())
            enhancer = ImageEnhance.Brightness(image_pil)
            bright_pil = enhancer.enhance(brightness_factor)
            bright_tensor = TF.to_tensor(bright_pil).unsqueeze(0)
            
            return bright_tensor.to(image.device)
        except Exception as e:
            print(f"Error in brightness_adjustment: {e}")
            return image.unsqueeze(0) if image.dim() == 3 else image
    
    def apply_single_attack(self, image: torch.Tensor, attack_config: dict) -> torch.Tensor:
        """Apply a single attack based on configuration"""
        try:
            attack_type = attack_config["type"]
            params = attack_config["params"]
            
            if attack_type == "blur":
                return self.gaussian_blur(image, **params)
            elif attack_type == "jpeg":
                return self.jpeg_compression(image, **params)
            elif attack_type == "noise":
                return self.additive_white_gaussian_noise(image, **params)
            elif attack_type == "rotation":
                return self.rotation(image, **params)
            elif attack_type == "scaling":
                return self.scaling(image, **params)
            elif attack_type == "cropping":
                return self.center_crop(image, **params)
            elif attack_type == "sharpening":
                return self.sharpening(image, **params)
            elif attack_type == "brightness":
                return self.brightness_adjustment(image, **params)
            else:
                return image
        except Exception as e:
            print(f"Error in apply_single_attack: {e}")
            return image

    def apply_attack_combination(self, image: torch.Tensor, attack_types: list, attack_params: dict) -> tuple:
        """Apply combination of attacks"""
        try:
            attacked_image = image.clone()
            attack_info = []
            
            for attack_type in attack_types:
                if attack_type == "blur" and "blur" in attack_params:
                    attacked_image = self.gaussian_blur(attacked_image, **attack_params["blur"])
                    attack_info.append(f"blur_{attack_params['blur']}")
                elif attack_type == "jpeg" and "jpeg" in attack_params:
                    attacked_image = self.jpeg_compression(attacked_image, **attack_params["jpeg"])
                    attack_info.append(f"jpeg_{attack_params['jpeg']}")
                elif attack_type == "noise" and "noise" in attack_params:
                    attacked_image = self.additive_white_gaussian_noise(attacked_image, **attack_params["noise"])
                    attack_info.append(f"noise_{attack_params['noise']}")
                elif attack_type == "rotation" and "rotation" in attack_params:
                    attacked_image = self.rotation(attacked_image, **attack_params["rotation"])
                    attack_info.append(f"rotation_{attack_params['rotation']}")
                elif attack_type == "scaling" and "scaling" in attack_params:
                    attacked_image = self.scaling(attacked_image, **attack_params["scaling"])
                    attack_info.append(f"scaling_{attack_params['scaling']}")
                elif attack_type == "cropping" and "cropping" in attack_params:
                    attacked_image = self.center_crop(attacked_image, **attack_params["cropping"])
                    attack_info.append(f"cropping_{attack_params['cropping']}")
                elif attack_type == "sharpening" and "sharpening" in attack_params:
                    attacked_image = self.sharpening(attacked_image, **attack_params["sharpening"])
                    attack_info.append(f"sharpening_{attack_params['sharpening']}")
            
            return attacked_image, attack_info
            
        except Exception as e:
            print(f"Error in apply_attack_combination: {e}")
            return image, []


# Predefined attack combinations for different intensity levels
ATTACK_PRESETS = {
    "mild": {
        "attack_types": ["blur", "jpeg"],
        "attack_params": {
            "blur": {"kernel_size": 3, "sigma": 0.5},
            "jpeg": {"quality": 85}
        }
    },
    "moderate": {
        "attack_types": ["blur", "jpeg", "noise"],
        "attack_params": {
            "blur": {"kernel_size": 5, "sigma": 1.0},
            "jpeg": {"quality": 70},
            "noise": {"noise_std": 0.03}
        }
    },
    "strong": {
        "attack_types": ["blur", "jpeg", "noise", "scaling"],
        "attack_params": {
            "blur": {"kernel_size": 7, "sigma": 1.5},
            "jpeg": {"quality": 50},
            "noise": {"noise_std": 0.05},
            "scaling": {"scale_factor": 0.8}
        }
    },
    "extreme": {
        "attack_types": ["blur", "jpeg", "noise", "scaling", "rotation"],
        "attack_params": {
            "blur": {"kernel_size": 9, "sigma": 2.0},
            "jpeg": {"quality": 25},
            "noise": {"noise_std": 0.08},
            "scaling": {"scale_factor": 0.7},
            "rotation": {"angle_degrees": 15}
        }
    }
}


def test_attack_system():
    """Test the attack system with a dummy image"""
    print("Testing Watermark Attack System...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attack_system = WatermarkAttacks(device=device)
    
    # Create dummy image
    dummy_image = torch.rand(1, 3, 256, 256).to(device)
    
    # Test individual attacks
    attacks_to_test = [
        ("gaussian_blur", {"kernel_size": 5, "sigma": 1.0}),
        ("jpeg_compression", {"quality": 70}),
        ("additive_white_gaussian_noise", {"noise_std": 0.05}),
        ("rotation", {"angle_degrees": 10}),
        ("scaling", {"scale_factor": 0.8}),
        ("center_crop", {"crop_ratio": 0.8}),
    ]
    
    for attack_name, params in attacks_to_test:
        try:
            attack_func = getattr(attack_system, attack_name)
            result = attack_func(dummy_image, **params)
            print(f"✅ {attack_name}: Success (shape: {result.shape})")
        except Exception as e:
            print(f"❌ {attack_name}: Failed - {e}")
    
    # Test combination attacks
    for preset_name, preset_config in ATTACK_PRESETS.items():
        try:
            result, info = attack_system.apply_attack_combination(
                dummy_image,
                preset_config["attack_types"],
                preset_config["attack_params"]
            )
            print(f"✅ {preset_name} combination: Success")
        except Exception as e:
            print(f"❌ {preset_name} combination: Failed - {e}")
    
    print("Attack system testing completed!")


if __name__ == "__main__":
    test_attack_system()

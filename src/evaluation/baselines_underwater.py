"""
Baseline methods for underwater image enhancement (without RL).

Provides 6 reference baselines for comparison with DQN agent.
"""

import torch
from torch import Tensor
import cv2
import numpy as np
from typing import Dict, Tuple


def _tensor_to_numpy(img: Tensor) -> np.ndarray:
    """Convert from torch tensor [0,1] to numpy [0,255]."""
    if img.dim() == 3:
        img = img.permute(1, 2, 0).contiguous()
    img_np = (img.cpu().numpy() * 255).astype(np.uint8)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


def _numpy_to_tensor(img_np: np.ndarray) -> Tensor:
    """Convert from numpy [0,255] to torch tensor [0,1]."""
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img_np.astype(np.float32) / 255.0)
    if img.dim() == 3:
        img = img.permute(2, 0, 1).contiguous()
    return img


class Baseline:
    """Base class for baselines."""
    
    def enhance(self, image: Tensor) -> Tensor:
        """Enhance image. Subclasses override."""
        raise NotImplementedError
    
    def __call__(self, image: Tensor) -> Tensor:
        return self.enhance(image)


class InputOnlyBaseline(Baseline):
    """Baseline 1: Use input image directly (no enhancement)."""
    
    def enhance(self, image: Tensor) -> Tensor:
        """Return image unchanged."""
        return image.clone()


class WhiteBalanceOnlyBaseline(Baseline):
    """Baseline 2: Only white balance (Grayworld)."""
    
    def enhance(self, image: Tensor) -> Tensor:
        """Apply white balance in LAB space."""
        img_np = _tensor_to_numpy(image)
        
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        mean_a = img_lab[:, :, 1].mean()
        mean_b = img_lab[:, :, 2].mean()
        
        a_shift = 128 - mean_a
        b_shift = 128 - mean_b
        
        img_lab[:, :, 1] = np.clip(img_lab[:, :, 1] + a_shift, 0, 255)
        img_lab[:, :, 2] = np.clip(img_lab[:, :, 2] + b_shift, 0, 255)
        
        img_bgr = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return _numpy_to_tensor(img_bgr)


class CLAHEOnlyBaseline(Baseline):
    """Baseline 3: Only CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    
    def enhance(self, image: Tensor) -> Tensor:
        """Apply CLAHE."""
        img_np = _tensor_to_numpy(image)
        
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l_channel = img_lab[:, :, 0]
        
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe_obj.apply(l_channel)
        
        img_lab[:, :, 0] = l_clahe
        img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        return _numpy_to_tensor(img_bgr)


class WhiteBalanceAndCLAHEBaseline(Baseline):
    """Baseline 4: White balance + CLAHE."""
    
    def __init__(self):
        self.wb_baseline = WhiteBalanceOnlyBaseline()
        self.clahe_baseline = CLAHEOnlyBaseline()
    
    def enhance(self, image: Tensor) -> Tensor:
        """Apply white balance then CLAHE."""
        # First white balance
        img_wb = self.wb_baseline.enhance(image)
        
        # Then CLAHE
        img_clahe = self.clahe_baseline.enhance(img_wb)
        
        return img_clahe


class DCPBaselineSimple(Baseline):
    """Baseline 5: Simple DCP (Dark Channel Prior) dehazing."""
    
    def enhance(self, image: Tensor) -> Tensor:
        """Apply DCP dehazing."""
        img_np = _tensor_to_numpy(image)
        
        # Compute dark channel
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            dark_channel = np.min(img_np, axis=2)
        else:
            dark_channel = img_np
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel_filtered = cv2.morphologyEx(dark_channel, cv2.MORPH_CLOSE, kernel)
        
        # Transmission map
        atmospheric_light = max(float(np.percentile(dark_channel_filtered, 95)), 1.0)
        transmission = 1 - 0.95 * (dark_channel_filtered.astype(np.float32) / atmospheric_light)
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Dehaze
        result = np.zeros_like(img_np, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = (img_np[:, :, c].astype(np.float32) - atmospheric_light) / transmission + atmospheric_light
        
        result = np.clip(result, 0, 255)
        return _numpy_to_tensor(result.astype(np.uint8))


class AggressiveBlendBaseline(Baseline):
    """Baseline 6: Aggressive blend of multiple techniques."""
    
    def enhance(self, image: Tensor) -> Tensor:
        """
        Apply aggressive combination:
        1. White balance
        2. CLAHE
        3. Red channel boost
        4. Slight sharpening
        """
        img_np = _tensor_to_numpy(image)
        
        # Step 1: White balance
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_a = img_lab[:, :, 1].mean()
        mean_b = img_lab[:, :, 2].mean()
        img_lab[:, :, 1] += (128 - mean_a)
        img_lab[:, :, 2] += (128 - mean_b)
        img_lab = np.clip(img_lab, 0, 255)
        
        # Step 2: CLAHE on L channel
        l_channel = img_lab[:, :, 0].astype(np.uint8)
        clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe_obj.apply(l_channel)
        img_lab[:, :, 0] = l_clahe
        
        # Convert back to RGB
        img_bgr = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Step 3: Red channel boost
        img_bgr_float = img_bgr.astype(np.float32)
        img_bgr_float[:, :, 2] = np.clip(img_bgr_float[:, :, 2] * 1.2, 0, 255)  # Red channel in BGR
        
        # Step 4: Sharpening
        blurred = cv2.GaussianBlur(img_bgr_float.astype(np.uint8), (5, 5), 1.0)
        sharpened = cv2.addWeighted(
            img_bgr_float.astype(np.uint8), 1.3, blurred, -0.3, 0
        )
        sharpened = np.clip(sharpened, 0, 255)
        
        return _numpy_to_tensor(sharpened)


# Factory function
def get_baseline(name: str) -> Baseline:
    """
    Get baseline by name.
    
    Available baselines:
        - "input_only": No enhancement
        - "white_balance": Grayworld only
        - "clahe": CLAHE only
        - "white_balance_clahe": WB + CLAHE
        - "dcp": DCP dehazing
        - "aggressive": Aggressive blend of techniques
    """
    baselines = {
        "input_only": InputOnlyBaseline,
        "white_balance": WhiteBalanceOnlyBaseline,
        "clahe": CLAHEOnlyBaseline,
        "white_balance_clahe": WhiteBalanceAndCLAHEBaseline,
        "dcp": DCPBaselineSimple,
        "aggressive": AggressiveBlendBaseline,
    }
    
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")
    
    return baselines[name]()


BASELINE_NAMES = list(get_baseline.__doc__.split("Available baselines:")[1].split("- ")[1:])
BASELINE_IDS = ["input_only", "white_balance", "clahe", "white_balance_clahe", "dcp", "aggressive"]


if __name__ == "__main__":
    # Quick test
    print("Testing baselines...")
    
    dummy_image = torch.rand(3, 128, 128)
    
    for baseline_id in BASELINE_IDS:
        try:
            baseline = get_baseline(baseline_id)
            result = baseline(dummy_image)
            assert result.shape == dummy_image.shape
            assert result.min() >= 0 and result.max() <= 1
            print(f"✓ {baseline_id}: OK")
        except Exception as e:
            print(f"✗ {baseline_id}: FAILED - {e}")
    
    print("\n✓ All baselines tested!")

"""
Classical/manual baselines for adaptive image enhancement.

Includes two types of baselines:
1. Policy-based: Fixed sequences of RL actions (brightness, contrast, etc.)
2. Method-based: Classical image enhancement algorithms (DCP, histogram equalization, etc.)
"""

from PIL import Image
import numpy as np
import cv2

from src.actions.filters import ImageAction, apply_action
from src.metrics import compute_psnr, compute_ssim


BASELINE_POLICIES: dict[str, list[int]] = {
    "input_only": [
        int(ImageAction.STOP),
    ],
    "brightness_contrast_sharpen": [
        int(ImageAction.INCREASE_BRIGHTNESS),
        int(ImageAction.INCREASE_CONTRAST),
        int(ImageAction.SHARPEN),
        int(ImageAction.STOP),
    ],
    "brightness_gamma_contrast": [
        int(ImageAction.INCREASE_BRIGHTNESS),
        int(ImageAction.GAMMA_UP),
        int(ImageAction.INCREASE_CONTRAST),
        int(ImageAction.STOP),
    ],
    "denoise_sharpen": [
        int(ImageAction.GAUSSIAN_DENOISE),
        int(ImageAction.SHARPEN),
        int(ImageAction.STOP),
    ],
}


def apply_baseline_policy(
    degraded_image: Image.Image,
    actions: list[int],
) -> Image.Image:
    current_image = degraded_image.copy()

    for action in actions:
        if action == int(ImageAction.STOP):
            break

        current_image = apply_action(current_image, action)

    return current_image


def evaluate_baseline_policy(
    clean_image: Image.Image,
    degraded_image: Image.Image,
    actions: list[int],
) -> dict[str, float]:
    enhanced_image = apply_baseline_policy(degraded_image, actions)

    psnr_degraded = compute_psnr(degraded_image, clean_image)
    psnr_enhanced = compute_psnr(enhanced_image, clean_image)

    ssim_degraded = compute_ssim(degraded_image, clean_image)
    ssim_enhanced = compute_ssim(enhanced_image, clean_image)

    return {
        "psnr_degraded": psnr_degraded,
        "psnr_enhanced": psnr_enhanced,
        "delta_psnr": psnr_enhanced - psnr_degraded,
        "ssim_degraded": ssim_degraded,
        "ssim_enhanced": ssim_enhanced,
        "delta_ssim": ssim_enhanced - ssim_degraded,
        "num_actions": float(len([a for a in actions if a != int(ImageAction.STOP)])),
    }


def evaluate_all_baselines(
    clean_image: Image.Image,
    degraded_image: Image.Image,
) -> dict[str, dict[str, float]]:
    results = {}

    for name, actions in BASELINE_POLICIES.items():
        results[name] = evaluate_baseline_policy(
            clean_image=clean_image,
            degraded_image=degraded_image,
            actions=actions,
        )

    return results


# ============================================================================
# Classical Method-Based Baselines
# ============================================================================

def identity_baseline(image: Image.Image) -> Image.Image:
    """
    Identity baseline: returns image unchanged.
    Useful for measuring improvement from any processing.
    """
    return image.copy()


def histogram_equalization_baseline(image: Image.Image) -> Image.Image:
    """
    Simple histogram equalization baseline.
    Improves contrast globally without domain-specific knowledge.
    Applies to L channel in LAB space for perceptual consistency.
    """
    array = np.asarray(image).astype(np.uint8)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    
    # Apply histogram equalization to L channel
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def dark_channel_prior_baseline(image: Image.Image, window_size: int = 15) -> Image.Image:
    """
    Dark Channel Prior (DCP) for underwater image enhancement.
    
    Based on: "Single Image Haze Removal via Composition Model" (He et al., 2009)
    and applied to underwater restoration.
    
    The dark channel represents the minimum intensity in local patches.
    In hazy/underwater images, the dark channel is bright (caused by airlight/backscatter).
    By estimating and removing the airlight, we restore contrast.
    
    Args:
        image: Input underwater image
        window_size: Size of local patches for dark channel computation
        
    Returns:
        Enhanced image with restored contrast
    """
    # Convert to float [0, 1]
    img_array = np.asarray(image).astype(np.float32) / 255.0
    
    # Compute dark channel: minimum intensity in each patch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.morphologyEx(
        (img_array.min(axis=2, keepdims=True) * 255).astype(np.uint8),
        cv2.MORPH_ERODE,
        kernel
    ).astype(np.float32) / 255.0
    
    # Estimate transmission map (how much light passes through medium)
    # transmission = 1 - omega * (dark_channel / max_value)
    omega = 0.95
    max_value = dark_channel.max()
    if max_value > 0:
        transmission = 1.0 - omega * (dark_channel / max_value)
    else:
        transmission = np.ones_like(dark_channel)
    
    # Smooth transmission map for consistency
    transmission = cv2.GaussianBlur(transmission, (11, 11), 0)
    transmission = np.clip(transmission, 0.1, 1.0)  # Ensure at least some transmission
    
    # Squeeze transmission to remove channel dimension
    if transmission.ndim == 3:
        transmission = transmission[:, :, 0]
    
    # Estimate airlight (atmospheric light)
    if img_array.shape[0] > 0 and img_array.shape[1] > 0:
        num_pixels = int(0.1 * img_array.shape[0] * img_array.shape[1])
        dark_flat = dark_channel.flatten()
        if len(dark_flat) > num_pixels:
            top_indices = np.argsort(dark_flat)[-num_pixels:]
            # Average color of brightest dark channel pixels
            airlight = img_array.reshape(-1, 3)[top_indices].mean(axis=0)
        else:
            airlight = img_array.mean(axis=(0, 1))
    else:
        airlight = np.array([0.5, 0.5, 0.5])
    
    # Restore image: I = (I - A) / t + A
    result = np.zeros_like(img_array)
    for c in range(3):
        result[:, :, c] = (img_array[:, :, c] - airlight[c]) / np.maximum(transmission, 0.1) + airlight[c]
    
    # Normalize and clip to valid range
    result = np.clip(result, 0.0, 1.0)
    result = (result * 255).astype(np.uint8)
    
    return Image.fromarray(result, mode="RGB")


def evaluate_method_baseline(
    clean_image: Image.Image,
    degraded_image: Image.Image,
    method_name: str,
) -> dict[str, float]:
    """
    Evaluate a classical method baseline.
    
    Args:
        clean_image: Reference high-quality image
        degraded_image: Degraded input image
        method_name: One of "identity", "histogram", "dcp"
        
    Returns:
        Dict with metrics: psnr_degraded, psnr_enhanced, delta_psnr, etc.
    """
    methods = {
        "identity": identity_baseline,
        "histogram": histogram_equalization_baseline,
        "dcp": dark_channel_prior_baseline,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Supported: {list(methods.keys())}")
    
    method = methods[method_name]
    enhanced_image = method(degraded_image)
    
    psnr_degraded = compute_psnr(degraded_image, clean_image)
    psnr_enhanced = compute_psnr(enhanced_image, clean_image)
    
    ssim_degraded = compute_ssim(degraded_image, clean_image)
    ssim_enhanced = compute_ssim(enhanced_image, clean_image)
    
    return {
        "psnr_degraded": psnr_degraded,
        "psnr_enhanced": psnr_enhanced,
        "delta_psnr": psnr_enhanced - psnr_degraded,
        "ssim_degraded": ssim_degraded,
        "ssim_enhanced": ssim_enhanced,
        "delta_ssim": ssim_enhanced - ssim_degraded,
    }


def evaluate_all_method_baselines(
    clean_image: Image.Image,
    degraded_image: Image.Image,
) -> dict[str, dict[str, float]]:
    """
    Evaluate all classical method baselines.
    
    Returns:
        Dict mapping method names to their evaluation results
    """
    results = {}
    for method_name in ["identity", "histogram", "dcp"]:
        try:
            results[method_name] = evaluate_method_baseline(
                clean_image=clean_image,
                degraded_image=degraded_image,
                method_name=method_name,
            )
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
            results[method_name] = {}
    
    return results
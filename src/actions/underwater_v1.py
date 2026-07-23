"""Deterministic underwater image-enhancement operators.

Each function maps a normalized ``CHW`` tensor to another normalized tensor.
The canonical ``underwater_curated_v1`` registry exposes only white balance,
contrast increase, sharpening, and STOP. Additional operators remain available
for controlled ablations and future action-set experiments.
"""

import cv2
import numpy as np
import torch
from torch import Tensor

# ============================================================================
# Helper Functions
# ============================================================================


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


def _clamp_tensor(img: Tensor) -> Tensor:
    """Clamp tensor to [0, 1]."""
    return torch.clamp(img, 0.0, 1.0)


# ============================================================================
# Action 0: White Balance (Grayworld in CIELAB)
# ============================================================================


def white_balance_grayworld(image: Tensor) -> Tensor:
    """
    White balance using Grayworld assumption in CIELAB space.

    Corrects color cast by assuming the average color should be gray.
    Effective for removing blue/green underwater cast.
    """
    # Convert to numpy for OpenCV processing
    img_np = _tensor_to_numpy(image)

    # Convert BGR → LAB
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Grayworld assumption: over a natural scene, the average color should be
    # neutral gray (A=B=128 in CIELAB). Underwater images are skewed toward
    # blue/green because red wavelengths attenuate fastest with depth, so the
    # measured A/B means drift away from 128 in a fairly consistent direction.
    mean_a = img_lab[:, :, 1].mean()
    mean_b = img_lab[:, :, 2].mean()

    # Shift each chrominance channel by exactly the offset needed to recenter
    # its mean at neutral gray — a global, deterministic, one-shot correction.
    a_shift = 128 - mean_a
    b_shift = 128 - mean_b

    img_lab[:, :, 1] = np.clip(img_lab[:, :, 1] + a_shift, 0, 255)
    img_lab[:, :, 2] = np.clip(img_lab[:, :, 2] + b_shift, 0, 255)

    # Convert back
    img_bgr = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return _numpy_to_tensor(img_bgr)


# ============================================================================
# Actions 1-2: Brightness Up/Down
# ============================================================================


def brightness_up(image: Tensor, factor: float = 1.15) -> Tensor:
    """Increase brightness (factor > 1)."""
    return _clamp_tensor(image * factor)


def brightness_down(image: Tensor, factor: float = 0.85) -> Tensor:
    """Decrease brightness (factor < 1)."""
    return _clamp_tensor(image * factor)


# ============================================================================
# Actions 3-4: Contrast Up/Down
# ============================================================================


def contrast_up(image: Tensor, factor: float = 1.15) -> Tensor:
    """Increase contrast."""
    mean = image.mean()
    return _clamp_tensor((image - mean) * factor + mean)


def contrast_down(image: Tensor, factor: float = 0.85) -> Tensor:
    """Decrease contrast."""
    mean = image.mean()
    return _clamp_tensor((image - mean) * factor + mean)


# ============================================================================
# Action 5: Red Channel Boost
# ============================================================================


def red_channel_boost(image: Tensor, multiplier: float = 1.5) -> Tensor:
    """
    Boost red channel specifically.

    Underwater imaging loses red wavelength first.
    This restores red intensity.
    """
    img_boosted = image.clone()
    img_boosted[0] = _clamp_tensor(img_boosted[0] * multiplier)  # Red channel
    return img_boosted


# ============================================================================
# Actions 6-7: Gamma Up/Down
# ============================================================================


def gamma_up(image: Tensor, gamma: float = 0.85) -> Tensor:
    """
    Gamma correction up (brightening, 0 < gamma < 1).

    Applies power-law transformation: I' = I^(1/gamma)
    Brightens mid-tones while preserving black/white points.
    """
    return _clamp_tensor(torch.pow(image, gamma))


def gamma_down(image: Tensor, gamma: float = 1.15) -> Tensor:
    """
    Gamma correction down (darkening, 1 < gamma).

    Applies power-law transformation: I' = I^(1/gamma)
    Darkens mid-tones while preserving extremes.
    """
    return _clamp_tensor(torch.pow(image, gamma))


# ============================================================================
# Action 8: Gaussian Denoise
# ============================================================================


def gaussian_denoise(image: Tensor, radius: float = 0.7) -> Tensor:
    """
    Gaussian blur denoising.

    Reduces sensor noise common in underwater imaging.
    Uses small radius to preserve edges.
    """
    img_np = _tensor_to_numpy(image)

    # Gaussian blur (kernel size must be odd)
    kernel_size = int(2 * radius + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)  # Minimum 3x3

    img_denoised = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), radius)

    return _numpy_to_tensor(img_denoised)


# ============================================================================
# Action 9: Sharpening
# ============================================================================


def sharpen(image: Tensor) -> Tensor:
    """
    Sharpening filter.

    Enhances edges to compensate for underwater blur.
    Uses unsharp masking: sharpened = image + (image - blurred).
    """
    img_np = _tensor_to_numpy(image)

    # Unsharp masking
    blurred = cv2.GaussianBlur(img_np, (5, 5), 1.0)
    sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255)

    return _numpy_to_tensor(sharpened)


# ============================================================================
# Action 10: Emboss
# ============================================================================


def emboss(image: Tensor, strength: float = 0.05) -> Tensor:
    """
    Emboss filter.

    Subtle texture emphasis to reveal details.
    """
    img_np = _tensor_to_numpy(image)

    # Emboss kernel
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32) / 8.0

    embossed = cv2.filter2D(img_np, -1, kernel)

    # Blend with original to avoid over-processing
    result = (1 - strength) * img_np + strength * embossed
    result = np.clip(result, 0, 255)

    return _numpy_to_tensor(result)


# ============================================================================
# Action 11: Histogram Equalization (Global)
# ============================================================================


def histogram_eq(image: Tensor) -> Tensor:
    """
    Global histogram equalization.

    Stretches intensity values across full range.
    Can cause over-enhancement; use CLAHE for better results.
    """
    img_np = _tensor_to_numpy(image)

    # Convert to LAB, equalize L channel only
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l_channel = img_lab[:, :, 0]

    # Equalize
    l_eq = cv2.equalizeHist(l_channel)

    # Replace
    img_lab[:, :, 0] = l_eq
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return _numpy_to_tensor(img_bgr)


# ============================================================================
# Action 12: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================


def clahe(image: Tensor, clip_limit: float = 2.0, grid_size: int = 8) -> Tensor:
    """
    Contrast Limited Adaptive Histogram Equalization.

    Applies histogram equalization locally (not globally).
    Avoids over-enhancement of noisy regions.
    Very effective for underwater images.
    """
    img_np = _tensor_to_numpy(image)

    # Convert to LAB, apply CLAHE to L channel
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l_channel = img_lab[:, :, 0]

    # Create CLAHE object
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_clahe = clahe_obj.apply(l_channel)

    # Replace
    img_lab[:, :, 0] = l_clahe
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return _numpy_to_tensor(img_bgr)


# ============================================================================
# Action 13: Dark Channel Prior (DCP) Dehazing
# ============================================================================


def dark_channel_prior(image: Tensor, window_size: int = 15, weight: float = 0.95) -> Tensor:
    """
    Dark Channel Prior dehazing.

    Physics-based method for haze/backscatter removal.
    Assumes at least one color channel has low intensity in some patches.

    Reference: He et al. "Single Image Haze Removal Using Dark Channel Prior"
    """
    img_np = _tensor_to_numpy(image)

    # Compute dark channel
    dark_channel = np.min(img_np, axis=2) if img_np.ndim == 3 and img_np.shape[2] == 3 else img_np

    # Morphological closing to get transmission estimate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel_filtered = cv2.morphologyEx(dark_channel, cv2.MORPH_CLOSE, kernel)

    # Transmission map (0 = haze, 1 = clear)
    atmospheric_light = float(np.percentile(dark_channel_filtered, 95))
    atmospheric_light = max(atmospheric_light, 1.0)
    transmission = 1 - weight * (dark_channel_filtered.astype(np.float32) / atmospheric_light)
    transmission = np.clip(transmission, 0.1, 1.0)  # Avoid division by zero

    # Dehaze each channel
    result = np.zeros_like(img_np, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = (
            img_np[:, :, c].astype(np.float32) - atmospheric_light
        ) / transmission + atmospheric_light

    result = np.clip(result, 0, 255)
    return _numpy_to_tensor(result.astype(np.uint8))


# ============================================================================
# Action 19: STOP (No-op, terminates episode)
# ============================================================================


def stop(image: Tensor) -> Tensor:
    """
    STOP action: terminate episode without modification.

    Returns image unchanged. Used by agent to end enhancement sequence.
    """
    return image.clone()


# ============================================================================
# Action Registry
# ============================================================================

# Canonical v4.0 action set: the only registry used by the official run
# (dqn_underwater_full_20260510_165955_1494) and by ARCHITECTURE.md. Every
# other function/registry in this module exists for ablations only.
UNDERWATER_CURATED_V1_ACTIONS = {
    0: white_balance_grayworld,
    1: contrast_up,
    2: sharpen,
    3: stop,
}

# Extended curated set: 8 actions that cover more OOD degradation modes.
UNDERWATER_EXTENDED_V1_ACTIONS = {
    0: white_balance_grayworld,
    1: contrast_up,
    2: contrast_down,
    3: clahe,
    4: red_channel_boost,
    5: gamma_up,
    6: sharpen,
    7: stop,
}

CURATED_ACTION_NAMES = {
    0: "white_balance",
    1: "contrast_up",
    2: "sharpen",
    3: "stop",
}

CURATED_ACTION_DESCRIPTIONS = {
    0: "White balance (Grayworld)",
    1: "Increase contrast",
    2: "Sharpen edges",
    3: "STOP (terminate)",
}

EXTENDED_ACTION_NAMES = {
    0: "white_balance",
    1: "contrast_up",
    2: "contrast_down",
    3: "clahe",
    4: "red_channel_boost",
    5: "gamma_up",
    6: "sharpen",
    7: "stop",
}

EXTENDED_ACTION_DESCRIPTIONS = {
    0: "White balance (Grayworld) — corregge cast cromatico",
    1: "Increase contrast — scattering basso",
    2: "Decrease contrast — over-contrast su scene particolari",
    3: "CLAHE — contrasto adattivo locale, robusto OOD",
    4: "Red channel boost — perdita rosso per profondita",
    5: "Gamma up — immagini molto scure",
    6: "Sharpen — blur da scattering",
    7: "STOP (terminate)",
}


def apply_action_curated(image: Tensor, action_id: int) -> Tensor:
    """
    Apply action by ID for the curated underwater action set.

    Args:
        image: Tensor [0, 1]
        action_id: Integer action ID

    Returns:
        Enhanced image tensor [0, 1]
    """
    if action_id not in UNDERWATER_CURATED_V1_ACTIONS:
        raise ValueError(f"Unknown curated action ID: {action_id}")

    action_fn = UNDERWATER_CURATED_V1_ACTIONS[action_id]
    return action_fn(image)


def apply_action_extended(image: Tensor, action_id: int) -> Tensor:
    """Apply action from the extended underwater action set."""
    if action_id not in UNDERWATER_EXTENDED_V1_ACTIONS:
        raise ValueError(f"Unknown extended action ID: {action_id}")

    action_fn = UNDERWATER_EXTENDED_V1_ACTIONS[action_id]
    return action_fn(image)

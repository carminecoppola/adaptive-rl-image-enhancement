"""
Image processing actions for the RL agent.

Each action takes a PIL image and returns a transformed PIL image.
These actions define the discrete action space of the environment.

Standard actions (0-8):
  - Brightness/Contrast/Sharpness adjustments
  - Gamma correction
  - Gaussian denoise
  - STOP

Underwater-specific actions (9-12):
  - Red channel boost (compensate for underwater absorption)
  - LAB color balance (white balance correction)
  - CLAHE (contrast-limited adaptive histogram equalization)
  - Saturation boost (counteract desaturation)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter


class ImageAction(IntEnum):
    # Standard image enhancement actions
    INCREASE_BRIGHTNESS = 0
    DECREASE_BRIGHTNESS = 1
    INCREASE_CONTRAST = 2
    DECREASE_CONTRAST = 3
    GAUSSIAN_DENOISE = 4
    SHARPEN = 5
    GAMMA_UP = 6
    GAMMA_DOWN = 7
    STOP = 8
    
    # Underwater-specific actions
    RED_CHANNEL_BOOST = 9
    LAB_COLOR_BALANCE = 10
    CLAHE = 11
    SATURATION_BOOST = 12


ACTION_NAMES = {
    ImageAction.INCREASE_BRIGHTNESS: "increase_brightness",
    ImageAction.DECREASE_BRIGHTNESS: "decrease_brightness",
    ImageAction.INCREASE_CONTRAST: "increase_contrast",
    ImageAction.DECREASE_CONTRAST: "decrease_contrast",
    ImageAction.GAUSSIAN_DENOISE: "gaussian_denoise",
    ImageAction.SHARPEN: "sharpen",
    ImageAction.GAMMA_UP: "gamma_up",
    ImageAction.GAMMA_DOWN: "gamma_down",
    ImageAction.STOP: "stop",
    ImageAction.RED_CHANNEL_BOOST: "red_channel_boost",
    ImageAction.LAB_COLOR_BALANCE: "lab_color_balance",
    ImageAction.CLAHE: "clahe",
    ImageAction.SATURATION_BOOST: "saturation_boost",
}


def increase_brightness(image: Image.Image, factor: float = 1.15) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def decrease_brightness(image: Image.Image, factor: float = 0.85) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def increase_contrast(image: Image.Image, factor: float = 1.15) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def decrease_contrast(image: Image.Image, factor: float = 0.85) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def gaussian_denoise(image: Image.Image, radius: float = 0.7) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def sharpen(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.SHARPEN)


def gamma_correction(image: Image.Image, gamma: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 255.0
    corrected = np.power(array, gamma)
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected = (corrected * 255).astype(np.uint8)
    return Image.fromarray(corrected)


def gamma_up(image: Image.Image, gamma: float = 0.85) -> Image.Image:
    return gamma_correction(image, gamma=gamma)


def gamma_down(image: Image.Image, gamma: float = 1.15) -> Image.Image:
    return gamma_correction(image, gamma=gamma)


def red_channel_boost(image: Image.Image, factor: float = 1.3) -> Image.Image:
    """
    Boost the red channel to compensate for underwater absorption.
    
    Red light is absorbed most in underwater environments, so amplifying
    the red channel helps restore color balance.
    """
    array = np.asarray(image).astype(np.float32)
    array[:, :, 0] = np.clip(array[:, :, 0] * factor, 0, 255)
    return Image.fromarray(array.astype(np.uint8))


def lab_color_balance(image: Image.Image) -> Image.Image:
    """
    Perform color balance using LAB color space (gray world assumption).
    
    Corrects color cast (dominant blue/green tint) by neutralizing
    the a* and b* channels in LAB space toward zero.
    """
    array = np.asarray(image).astype(np.uint8)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Neutralize a and b channels toward 128 (neutral)
    lab[:, :, 1] -= (lab[:, :, 1].mean() - 128.0)
    lab[:, :, 2] -= (lab[:, :, 2].mean() - 128.0)
    
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def clahe(image: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization).
    
    Improves local contrast without excessive amplification of noise,
    useful for underwater images with uneven illumination.
    """
    array = np.asarray(image).astype(np.uint8)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
    
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def saturation_boost(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Increase color saturation.
    
    Underwater images suffer from desaturation due to color absorption.
    Boosting saturation helps restore color vibrancy.
    """
    return ImageEnhance.Color(image).enhance(factor)


def apply_action(image: Image.Image, action: int) -> Image.Image:
    """
    Apply one discrete image-processing action.

    STOP returns the image unchanged.
    
    Actions 0-8: standard image enhancement (brightness, contrast, gamma, etc.)
    Actions 9-12: underwater-specific (red boost, color balance, CLAHE, saturation)
    """
    action = ImageAction(action)

    if action == ImageAction.INCREASE_BRIGHTNESS:
        return increase_brightness(image)

    if action == ImageAction.DECREASE_BRIGHTNESS:
        return decrease_brightness(image)

    if action == ImageAction.INCREASE_CONTRAST:
        return increase_contrast(image)

    if action == ImageAction.DECREASE_CONTRAST:
        return decrease_contrast(image)

    if action == ImageAction.GAUSSIAN_DENOISE:
        return gaussian_denoise(image)

    if action == ImageAction.SHARPEN:
        return sharpen(image)

    if action == ImageAction.GAMMA_UP:
        return gamma_up(image)

    if action == ImageAction.GAMMA_DOWN:
        return gamma_down(image)

    if action == ImageAction.RED_CHANNEL_BOOST:
        return red_channel_boost(image)

    if action == ImageAction.LAB_COLOR_BALANCE:
        return lab_color_balance(image)

    if action == ImageAction.CLAHE:
        return clahe(image)

    if action == ImageAction.SATURATION_BOOST:
        return saturation_boost(image)

    if action == ImageAction.STOP:
        return image

    raise ValueError(f"Unsupported action: {action}")


def get_action_name(action: int) -> str:
    return ACTION_NAMES[ImageAction(action)]
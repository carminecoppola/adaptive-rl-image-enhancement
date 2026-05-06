"""
Image processing actions for the RL agent.

Each action takes a PIL image and returns a transformed PIL image.
These actions define the discrete action space of the environment.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class ImageAction(IntEnum):
    INCREASE_BRIGHTNESS = 0
    DECREASE_BRIGHTNESS = 1
    INCREASE_CONTRAST = 2
    DECREASE_CONTRAST = 3
    GAUSSIAN_DENOISE = 4
    SHARPEN = 5
    GAMMA_UP = 6
    GAMMA_DOWN = 7
    STOP = 8


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


def apply_action(image: Image.Image, action: int) -> Image.Image:
    """
    Apply one discrete image-processing action.

    STOP returns the image unchanged.
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

    if action == ImageAction.STOP:
        return image

    raise ValueError(f"Unsupported action: {action}")


def get_action_name(action: int) -> str:
    return ACTION_NAMES[ImageAction(action)]
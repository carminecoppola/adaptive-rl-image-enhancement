"""
PSNR metric.
"""

import math
from typing import Union

import numpy as np
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray]


def image_to_float_array(image: ImageLike) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        return array / 255.0

    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32, copy=False)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected ndarray shape (H, W, 3), got {arr.shape}")
        # Support both [0, 255] and [0, 1] ranges.
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    raise TypeError(f"Unsupported image type: {type(image)}")


def compute_psnr(image: ImageLike, target: ImageLike, max_value: float = 1.0) -> float:
    image_array = image_to_float_array(image)
    target_array = image_to_float_array(target)

    mse = np.mean((image_array - target_array) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * math.log10(max_value / math.sqrt(mse))

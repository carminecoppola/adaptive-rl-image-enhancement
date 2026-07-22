"""
SSIM metric.
"""

from typing import Union

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


ImageLike = Union[Image.Image, np.ndarray]


def image_to_float_array(image: ImageLike) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        return array / 255.0

    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32, copy=False)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected ndarray shape (H, W, 3), got {arr.shape}")
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    raise TypeError(f"Unsupported image type: {type(image)}")


def compute_ssim(image: ImageLike, target: ImageLike) -> float:
    image_array = image_to_float_array(image)
    target_array = image_to_float_array(target)

    return float(
        ssim(
            target_array,
            image_array,
            channel_axis=-1,
            data_range=1.0,
        )
    )

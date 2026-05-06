"""
SSIM metric.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def image_to_float_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return array / 255.0


def compute_ssim(image: Image.Image, target: Image.Image) -> float:
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
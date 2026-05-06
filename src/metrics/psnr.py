"""
PSNR metric.
"""

import math
import numpy as np
from PIL import Image


def image_to_float_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return array / 255.0


def compute_psnr(image: Image.Image, target: Image.Image, max_value: float = 1.0) -> float:
    image_array = image_to_float_array(image)
    target_array = image_to_float_array(target)

    mse = np.mean((image_array - target_array) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * math.log10(max_value / math.sqrt(mse))
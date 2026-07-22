from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compute_uciqe(image: Image.Image) -> float:
    """Approximate UCIQE no-reference underwater quality metric."""
    rgb = _pil_to_rgb_array(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0] / 255.0
    a_channel = lab[:, :, 1] - 128.0
    b_channel = lab[:, :, 2] - 128.0

    chroma = np.sqrt(a_channel**2 + b_channel**2)
    chroma_std = float(np.std(chroma))
    luminance_contrast = float(np.percentile(l_channel, 99) - np.percentile(l_channel, 1))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    saturation_mean = float(np.mean(saturation))

    return 0.4680 * chroma_std + 0.2745 * luminance_contrast + 0.2576 * saturation_mean


def compute_uiqm_proxy(image: Image.Image) -> float:
    """
    Lightweight proxy for UIQM-style underwater quality.

    Higher is better. This is a documented proxy, not a strict UIQM reproduction.
    """
    rgb = _pil_to_rgb_array(image).astype(np.float32) / 255.0

    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    colorfulness = float(
        np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(np.std(gray))

    return 0.4 * colorfulness + 0.35 * sharpness + 0.25 * contrast

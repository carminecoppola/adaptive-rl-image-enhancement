"""Deterministic interfaces for optional synthetic image degradation."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def degrade_image(
    image: Image.Image,
    degradation_type: str,
    noise_std: float = 0.1,
) -> Image.Image:
    """Apply one supported synthetic degradation to an RGB image.

    The canonical underwater workflow passes paired degraded images and uses
    ``degradation_type='none'``. Other modes are retained for controlled
    compatibility experiments.
    """
    rgb = image.convert("RGB")
    degradation = degradation_type.strip().lower()

    if degradation == "none":
        return rgb.copy()
    if degradation in {"gaussian_blur", "blur"}:
        return rgb.filter(ImageFilter.GaussianBlur(radius=1.0))
    if degradation in {"brightness", "darken"}:
        return ImageEnhance.Brightness(rgb).enhance(0.7)
    if degradation in {"contrast", "low_contrast"}:
        return ImageEnhance.Contrast(rgb).enhance(0.65)
    if degradation == "gaussian_noise":
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        noise = np.random.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        degraded = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((degraded * 255.0).astype(np.uint8), mode="RGB")

    raise ValueError(f"Unsupported degradation type: {degradation_type!r}")


__all__ = ["degrade_image"]

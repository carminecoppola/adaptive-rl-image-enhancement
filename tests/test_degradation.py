import numpy as np
import pytest
from PIL import Image

from src.data.degradation import degrade_image


@pytest.fixture
def test_image() -> Image.Image:
    array = np.ones((64, 64, 3), dtype=np.uint8) * 180
    return Image.fromarray(array, mode="RGB")


@pytest.mark.parametrize(
    "degradation_type",
    ["gaussian_noise", "low_brightness", "low_contrast", "blur", "combined"],
)
def test_supported_degradations_return_image(test_image: Image.Image, degradation_type: str) -> None:
    degraded = degrade_image(test_image, degradation_type=degradation_type)
    assert isinstance(degraded, Image.Image)
    assert degraded.mode == "RGB"
    assert degraded.size == test_image.size


def test_low_brightness_reduces_mean_intensity(test_image: Image.Image) -> None:
    degraded = degrade_image(test_image, degradation_type="low_brightness", brightness_factor=0.5)
    assert np.asarray(degraded).mean() < np.asarray(test_image).mean()


def test_low_contrast_returns_same_shape(test_image: Image.Image) -> None:
    degraded = degrade_image(test_image, degradation_type="low_contrast", contrast_factor=0.5)
    assert degraded.size == test_image.size


def test_unsupported_degradation_raises(test_image: Image.Image) -> None:
    with pytest.raises(ValueError):
        degrade_image(test_image, degradation_type="unsupported")

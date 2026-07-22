"""
Test suite for classical baseline methods.
"""

import numpy as np
import pytest
from PIL import Image

from src.evaluation.baselines import (
    dark_channel_prior_baseline,
    evaluate_all_method_baselines,
    evaluate_method_baseline,
    histogram_equalization_baseline,
    identity_baseline,
)


@pytest.fixture
def test_image() -> Image.Image:
    """Create a test RGB image."""
    array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def clean_image() -> Image.Image:
    """Create a clean reference image."""
    array = np.ones((64, 64, 3), dtype=np.uint8) * 200
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def degraded_image() -> Image.Image:
    """Create a degraded version (dark and blue-tinted)."""
    array = np.ones((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = 80  # Red: low
    array[:, :, 1] = 100  # Green: low
    array[:, :, 2] = 150  # Blue-heavy tint
    return Image.fromarray(array, mode="RGB")


def test_identity_baseline_preserves_image(test_image):
    """Test that identity baseline returns the image unchanged."""
    result = identity_baseline(test_image)

    assert result.size == test_image.size
    assert result.tobytes() == test_image.tobytes()


def test_histogram_equalization_baseline(test_image):
    """Test that histogram equalization returns a valid image."""
    result = histogram_equalization_baseline(test_image)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == test_image.size


def test_dark_channel_prior_baseline(test_image):
    """Test that DCP baseline returns a valid image."""
    result = dark_channel_prior_baseline(test_image)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == test_image.size

    # Check values are in valid range
    array = np.asarray(result)
    assert array.min() >= 0
    assert array.max() <= 255


def test_dcp_improves_degraded_image(clean_image, degraded_image):
    """Test that DCP can improve a degraded image."""
    enhanced = dark_channel_prior_baseline(degraded_image)

    # DCP should produce a different image
    assert enhanced.tobytes() != degraded_image.tobytes()

    # Output should be valid
    array = np.asarray(enhanced)
    assert array.dtype == np.uint8
    assert array.shape == degraded_image.size[::-1] + (3,)


def test_evaluate_method_baseline(clean_image, degraded_image):
    """Test evaluation of a single baseline method."""
    result = evaluate_method_baseline(
        clean_image=clean_image,
        degraded_image=degraded_image,
        method_name="histogram",
    )

    assert "psnr_degraded" in result
    assert "psnr_enhanced" in result
    assert "delta_psnr" in result
    assert "ssim_degraded" in result
    assert "ssim_enhanced" in result
    assert "delta_ssim" in result


def test_evaluate_all_method_baselines(clean_image, degraded_image):
    """Test evaluation of all baseline methods."""
    results = evaluate_all_method_baselines(clean_image, degraded_image)

    # Should have 3 methods
    assert len(results) == 3
    assert "identity" in results
    assert "histogram" in results
    assert "dcp" in results

    # Each should have metrics
    for _method_name, metrics in results.items():
        if metrics:  # Some might be empty if there are errors
            assert "psnr_degraded" in metrics
            assert "psnr_enhanced" in metrics


def test_dcp_with_small_window_size(test_image):
    """Test DCP with different window sizes."""
    result_small = dark_channel_prior_baseline(test_image, window_size=5)
    result_large = dark_channel_prior_baseline(test_image, window_size=25)

    assert result_small.size == test_image.size
    assert result_large.size == test_image.size

    # Results should be different due to different window sizes
    assert result_small.tobytes() != result_large.tobytes()


def test_baseline_methods_handle_edge_cases():
    """Test baseline methods with edge case images."""
    # All white image
    white_img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    result = dark_channel_prior_baseline(white_img)
    assert isinstance(result, Image.Image)

    # All black image
    black_img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    result = dark_channel_prior_baseline(black_img)
    assert isinstance(result, Image.Image)

    # Gradient image
    gradient_array = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        gradient_array[i, :, :] = int(255 * i / 31)
    gradient_img = Image.fromarray(gradient_array, mode="RGB")
    result = dark_channel_prior_baseline(gradient_img)
    assert isinstance(result, Image.Image)

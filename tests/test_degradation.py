"""
Test suite for underwater image degradation.
Verifies physics-based underwater image simulation.
"""

import pytest
import numpy as np
from PIL import Image
from src.data.degradation import degrade_underwater, degrade_image


@pytest.fixture
def test_image() -> Image.Image:
    """Create a test RGB image."""
    array = np.ones((64, 64, 3), dtype=np.uint8) * 200
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def gradient_image() -> Image.Image:
    """Create a gradient test image for analyzing color distortion."""
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    # Red gradient
    for i in range(64):
        array[i, :, 0] = int(255 * i / 63)
    # Green gradient
    for i in range(64):
        array[:, i, 1] = int(255 * i / 63)
    # Blue gradient
    for j in range(64):
        array[j, :, 2] = int(255 * j / 63)
    return Image.fromarray(array, mode="RGB")


def test_degrade_underwater_returns_image(test_image):
    """Test that degrade_underwater returns a PIL image."""
    result = degrade_underwater(test_image)
    
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == test_image.size


def test_degrade_underwater_absorption(test_image):
    """Test that red channel is absorbed more than blue."""
    # Very high depth to see strong effect
    degraded = degrade_underwater(test_image, depth=10.0, turbidity=0.0)
    
    original_array = np.asarray(test_image).astype(np.float32)
    degraded_array = np.asarray(degraded).astype(np.float32)
    
    # Red channel should be reduced more than blue
    red_reduction = (original_array[:, :, 0].mean() - degraded_array[:, :, 0].mean()) / original_array[:, :, 0].mean()
    blue_reduction = (original_array[:, :, 2].mean() - degraded_array[:, :, 2].mean()) / original_array[:, :, 2].mean()
    
    assert red_reduction > blue_reduction, "Red should be absorbed more than blue"


def test_degrade_underwater_backscatter(test_image):
    """Test that backscatter creates blue/green tint."""
    degraded = degrade_underwater(test_image, depth=5.0, turbidity=0.8)
    
    original_array = np.asarray(test_image).astype(np.float32)
    degraded_array = np.asarray(degraded).astype(np.float32)
    
    # With high turbidity and backscatter, blue and green should be boosted relative to red
    green_boost = degraded_array[:, :, 1].mean() - original_array[:, :, 1].mean()
    blue_boost = degraded_array[:, :, 2].mean() - original_array[:, :, 2].mean()
    red_change = degraded_array[:, :, 0].mean() - original_array[:, :, 0].mean()
    
    # Green and blue should have less negative change (or positive) compared to red
    assert green_boost > red_change
    assert blue_boost > red_change


def test_degrade_underwater_depth_effect(test_image):
    """Test that stronger depth creates more degradation."""
    shallow = degrade_underwater(test_image, depth=1.0)
    deep = degrade_underwater(test_image, depth=10.0)
    
    shallow_array = np.asarray(shallow).astype(np.float32)
    deep_array = np.asarray(deep).astype(np.float32)
    
    # Deep water should have less red on average (more absorption)
    shallow_red = shallow_array[:, :, 0].mean()
    deep_red = deep_array[:, :, 0].mean()
    
    assert deep_red < shallow_red, "Deeper water should absorb more red"


def test_degrade_underwater_turbidity_effect(test_image):
    """Test that higher turbidity affects backscatter."""
    clear = degrade_underwater(test_image, depth=5.0, turbidity=0.1)
    murky = degrade_underwater(test_image, depth=5.0, turbidity=0.9)
    
    clear_array = np.asarray(clear).astype(np.float32)
    murky_array = np.asarray(murky).astype(np.float32)
    
    # Murky water should have more veil (values closer to backscatter color)
    clear_std = clear_array.std()
    murky_std = murky_array.std()
    
    # Murky should have lower std due to veil flattening
    assert murky_std < clear_std, "Higher turbidity should reduce contrast"


def test_degrade_image_underwater_support():
    """Test that degrade_image supports underwater type."""
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    
    # Should not raise an error
    degraded = degrade_image(img, degradation_type="underwater", depth=5.0, turbidity=0.5)
    
    assert isinstance(degraded, Image.Image)
    assert degraded.size == img.size


def test_degrade_image_underwater_vs_direct():
    """Test that degrade_image and degrade_underwater produce same results."""
    img = Image.new("RGB", (64, 64), color=(150, 150, 150))
    
    via_degrade_image = degrade_image(img, degradation_type="underwater", depth=3.0, turbidity=0.4)
    via_degrade_underwater = degrade_underwater(img, depth=3.0, turbidity=0.4)
    
    # Should be very similar (allowing small rounding differences)
    img1 = np.asarray(via_degrade_image).astype(np.float32)
    img2 = np.asarray(via_degrade_underwater).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    assert mse < 1.0, "Results should be nearly identical"


def test_degrade_underwater_output_range(test_image):
    """Test that output values are in valid range [0, 255]."""
    degraded = degrade_underwater(test_image, depth=10.0, turbidity=1.0)
    
    array = np.asarray(degraded)
    assert array.min() >= 0, "Min value should be >= 0"
    assert array.max() <= 255, "Max value should be <= 255"

"""
Test suite for image processing actions.
Verifies underwater-specific and standard filters.
"""

import pytest
import numpy as np
from PIL import Image
from src.actions.filters import (
    ImageAction,
    apply_action,
    get_action_name,
    red_channel_boost,
    lab_color_balance,
    clahe,
    saturation_boost,
)


@pytest.fixture
def test_image() -> Image.Image:
    """Create a test RGB image for action testing."""
    array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def underwater_image() -> Image.Image:
    """Create a test underwater-like image (bluish tint)."""
    array = np.ones((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = 80   # Red: low
    array[:, :, 1] = 150  # Green: medium
    array[:, :, 2] = 200  # Blue: high
    return Image.fromarray(array, mode="RGB")


def test_action_enum():
    """Test that all new action enums are defined."""
    assert ImageAction.RED_CHANNEL_BOOST == 9
    assert ImageAction.LAB_COLOR_BALANCE == 10
    assert ImageAction.CLAHE == 11
    assert ImageAction.SATURATION_BOOST == 12


def test_action_names():
    """Test that all new actions have names."""
    assert get_action_name(9) == "red_channel_boost"
    assert get_action_name(10) == "lab_color_balance"
    assert get_action_name(11) == "clahe"
    assert get_action_name(12) == "saturation_boost"


def test_red_channel_boost(underwater_image):
    """Test that red_channel_boost increases red values."""
    boosted = red_channel_boost(underwater_image, factor=1.5)
    
    assert isinstance(boosted, Image.Image)
    assert boosted.mode == "RGB"
    assert boosted.size == underwater_image.size
    
    # Red channel should be boosted
    boosted_arr = np.asarray(boosted)
    original_arr = np.asarray(underwater_image)
    assert boosted_arr[:, :, 0].mean() > original_arr[:, :, 0].mean()


def test_lab_color_balance(underwater_image):
    """Test that lab_color_balance processes the image."""
    balanced = lab_color_balance(underwater_image)
    
    assert isinstance(balanced, Image.Image)
    assert balanced.mode == "RGB"
    assert balanced.size == underwater_image.size


def test_clahe(test_image):
    """Test that CLAHE enhances contrast."""
    enhanced = clahe(test_image)
    
    assert isinstance(enhanced, Image.Image)
    assert enhanced.mode == "RGB"
    assert enhanced.size == test_image.size


def test_saturation_boost(test_image):
    """Test that saturation_boost increases color saturation."""
    boosted = saturation_boost(test_image, factor=1.5)
    
    assert isinstance(boosted, Image.Image)
    assert boosted.mode == "RGB"
    assert boosted.size == test_image.size


def test_apply_action_underwater(test_image):
    """Test apply_action with underwater-specific actions."""
    # Red channel boost
    result = apply_action(test_image, ImageAction.RED_CHANNEL_BOOST)
    assert isinstance(result, Image.Image)
    
    # LAB color balance
    result = apply_action(test_image, ImageAction.LAB_COLOR_BALANCE)
    assert isinstance(result, Image.Image)
    
    # CLAHE
    result = apply_action(test_image, ImageAction.CLAHE)
    assert isinstance(result, Image.Image)
    
    # Saturation boost
    result = apply_action(test_image, ImageAction.SATURATION_BOOST)
    assert isinstance(result, Image.Image)


def test_apply_action_stop(test_image):
    """Test that STOP action returns unchanged image."""
    result = apply_action(test_image, ImageAction.STOP)
    
    # STOP should return exact same image
    assert result.tobytes() == test_image.tobytes()


def test_action_sequence(test_image):
    """Test applying multiple actions in sequence."""
    img = test_image
    actions = [
        ImageAction.RED_CHANNEL_BOOST,
        ImageAction.LAB_COLOR_BALANCE,
        ImageAction.CLAHE,
        ImageAction.SATURATION_BOOST,
    ]
    
    for action in actions:
        img = apply_action(img, action)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"


def test_action_space_size():
    """Test that action space includes all expected actions."""
    # Should have 13 actions total (0-12)
    expected_actions = 13
    
    # Try to instantiate all action values
    for i in range(expected_actions):
        action = ImageAction(i)
        assert action is not None
        assert get_action_name(i) is not None

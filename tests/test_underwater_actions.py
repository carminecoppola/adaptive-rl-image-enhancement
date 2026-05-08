"""
Tests for underwater_v1 actions.
"""

import pytest
import torch
from torch import Tensor

from src.actions.underwater_v1 import (
    UNDERWATER_V1_ACTIONS,
    UNDERWATER_CURATED_V1_ACTIONS,
    ACTION_NAMES,
    CURATED_ACTION_NAMES,
    white_balance_grayworld,
    brightness_up,
    brightness_down,
    contrast_up,
    contrast_down,
    red_channel_boost,
    gamma_up,
    gamma_down,
    gaussian_denoise,
    sharpen,
    emboss,
    histogram_eq,
    clahe,
    dark_channel_prior,
    stop,
    apply_action,
    apply_action_curated,
)


@pytest.fixture
def dummy_image():
    """Create dummy underwater-like image (blue-ish)."""
    img = torch.rand(3, 128, 128)
    # Add blue cast
    img[2] += 0.3  # Boost blue
    img[0] -= 0.1  # Reduce red
    return torch.clamp(img, 0, 1)


def test_all_actions_exist():
    """Test all expected actions are registered."""
    expected_count = 15  # 0-14, with 15-19 reserved
    assert len(UNDERWATER_V1_ACTIONS) >= expected_count


def test_action_deterministic(dummy_image):
    """Test actions are deterministic."""
    result1 = white_balance_grayworld(dummy_image.clone())
    result2 = white_balance_grayworld(dummy_image.clone())
    assert torch.allclose(result1, result2, atol=1e-5)


def test_action_output_shape(dummy_image):
    """Test output shape matches input."""
    for action_id, action_fn in UNDERWATER_V1_ACTIONS.items():
        result = action_fn(dummy_image)
        assert result.shape == dummy_image.shape, f"Action {action_id} changed shape"


def test_action_output_range(dummy_image):
    """Test output is in [0, 1] range."""
    for action_id, action_fn in UNDERWATER_V1_ACTIONS.items():
        result = action_fn(dummy_image)
        assert result.min() >= 0, f"Action {action_id} produced values < 0"
        assert result.max() <= 1, f"Action {action_id} produced values > 1"


def test_action_non_destructive(dummy_image):
    """Test original image is not modified."""
    original = dummy_image.clone()
    _ = white_balance_grayworld(dummy_image)
    assert torch.allclose(dummy_image, original)


def test_white_balance_effect(dummy_image):
    """Test white balance changes color distribution."""
    result = white_balance_grayworld(dummy_image)
    # Output should be different from input
    assert not torch.allclose(result, dummy_image)
    # But shapes match
    assert result.shape == dummy_image.shape


def test_brightness_up_effect(dummy_image):
    """Test brightness up increases image."""
    result = brightness_up(dummy_image, factor=1.2)
    assert result.mean() > dummy_image.mean()


def test_brightness_down_effect(dummy_image):
    """Test brightness down decreases image."""
    result = brightness_down(dummy_image, factor=0.8)
    assert result.mean() < dummy_image.mean()


def test_contrast_up_effect(dummy_image):
    """Test contrast up increases spread around the mean."""
    result = contrast_up(dummy_image, factor=1.2)
    assert result.std() > dummy_image.std()


def test_contrast_down_effect(dummy_image):
    """Test contrast down decreases spread around the mean."""
    result = contrast_down(dummy_image, factor=0.8)
    assert result.std() < dummy_image.std()


def test_red_channel_boost_effect(dummy_image):
    """Test red channel boost affects red channel."""
    result = red_channel_boost(dummy_image, multiplier=1.5)
    # Red channel should be boosted
    assert result[0].mean() > dummy_image[0].mean()
    # Other channels unchanged
    assert torch.allclose(result[1], dummy_image[1], atol=1e-5)
    assert torch.allclose(result[2], dummy_image[2], atol=1e-5)


def test_gamma_up_brightens(dummy_image):
    """Test gamma up brightens image."""
    result = gamma_up(dummy_image, gamma=0.85)
    assert result.mean() > dummy_image.mean()


def test_gamma_down_darkens(dummy_image):
    """Test gamma down darkens image."""
    result = gamma_down(dummy_image, gamma=1.15)
    assert result.mean() < dummy_image.mean()


def test_denoise_smooths(dummy_image):
    """Test denoise smooths image (reduces variance)."""
    result = gaussian_denoise(dummy_image, radius=1.0)
    # Denoised should have lower variance
    assert result.std() < dummy_image.std()


def test_sharpen_modifies(dummy_image):
    """Test sharpening modifies image."""
    result = sharpen(dummy_image)
    assert not torch.allclose(result, dummy_image)


def test_emboss_modifies(dummy_image):
    """Test embossing modifies image."""
    result = emboss(dummy_image, strength=0.1)
    assert not torch.allclose(result, dummy_image)


def test_histogram_eq_modifies(dummy_image):
    """Test histogram equalization modifies image."""
    result = histogram_eq(dummy_image)
    assert not torch.allclose(result, dummy_image)


def test_clahe_modifies(dummy_image):
    """Test CLAHE modifies image."""
    result = clahe(dummy_image, clip_limit=2.0, grid_size=8)
    assert not torch.allclose(result, dummy_image)


def test_dcp_modifies(dummy_image):
    """Test DCP dehazing modifies image."""
    result = dark_channel_prior(dummy_image, window_size=15, weight=0.95)
    assert not torch.allclose(result, dummy_image)


def test_stop_preserves(dummy_image):
    """Test STOP action preserves image."""
    result = stop(dummy_image)
    assert torch.allclose(result, dummy_image)


def test_apply_action_by_id(dummy_image):
    """Test apply_action function."""
    for action_id in UNDERWATER_V1_ACTIONS.keys():
        result = apply_action(dummy_image, action_id)
        assert result.shape == dummy_image.shape
        assert result.min() >= 0 and result.max() <= 1


def test_apply_action_invalid_id(dummy_image):
    """Test apply_action raises on invalid ID."""
    with pytest.raises(ValueError):
        apply_action(dummy_image, 999)


def test_action_names_complete():
    """Test ACTION_NAMES covers all actions."""
    for action_id in UNDERWATER_V1_ACTIONS.keys():
        assert action_id in ACTION_NAMES


def test_curated_action_set_complete():
    """Test curated action metadata matches action registry."""
    assert len(UNDERWATER_CURATED_V1_ACTIONS) == 4
    for action_id in UNDERWATER_CURATED_V1_ACTIONS.keys():
        assert action_id in CURATED_ACTION_NAMES


def test_apply_curated_action_by_id(dummy_image):
    """Test apply_action_curated function."""
    for action_id in UNDERWATER_CURATED_V1_ACTIONS.keys():
        result = apply_action_curated(dummy_image, action_id)
        assert result.shape == dummy_image.shape
        assert result.min() >= 0 and result.max() <= 1


def test_sequence_of_actions(dummy_image):
    """Test applying multiple actions in sequence."""
    result = dummy_image.clone()
    actions_sequence = [0, 3, 9, 14]  # white balance → contrast up → sharpen → stop
    
    for action_id in actions_sequence:
        result = apply_action(result, action_id)
        assert result.shape == dummy_image.shape
        assert result.min() >= 0 and result.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

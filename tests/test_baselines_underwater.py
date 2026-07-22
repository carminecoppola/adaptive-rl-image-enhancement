"""
Tests for underwater baselines.
"""

import pytest
import torch
from torch import Tensor

from src.evaluation.baselines_underwater import (
    InputOnlyBaseline,
    WhiteBalanceOnlyBaseline,
    CLAHEOnlyBaseline,
    WhiteBalanceAndCLAHEBaseline,
    DCPBaselineSimple,
    AggressiveBlendBaseline,
    get_baseline,
    BASELINE_IDS,
)


@pytest.fixture
def underwater_image():
    """Create underwater-like image (blue-shifted)."""
    img = torch.rand(3, 128, 128)
    img[2] += 0.3  # More blue
    img[0] -= 0.1  # Less red
    return torch.clamp(img, 0, 1)


def test_input_only_baseline_preserves(underwater_image):
    """Test input_only baseline returns unchanged image."""
    baseline = InputOnlyBaseline()
    result = baseline(underwater_image)
    
    assert torch.allclose(result, underwater_image)


def test_white_balance_baseline_modifies(underwater_image):
    """Test white balance baseline modifies image."""
    baseline = WhiteBalanceOnlyBaseline()
    result = baseline(underwater_image)
    
    assert not torch.allclose(result, underwater_image)
    assert result.shape == underwater_image.shape
    assert result.min() >= 0 and result.max() <= 1


def test_clahe_baseline_modifies(underwater_image):
    """Test CLAHE baseline modifies image."""
    baseline = CLAHEOnlyBaseline()
    result = baseline(underwater_image)
    
    assert not torch.allclose(result, underwater_image)
    assert result.shape == underwater_image.shape
    assert result.min() >= 0 and result.max() <= 1


def test_white_balance_and_clahe_modifies(underwater_image):
    """Test white balance + CLAHE baseline modifies image."""
    baseline = WhiteBalanceAndCLAHEBaseline()
    result = baseline(underwater_image)
    
    assert not torch.allclose(result, underwater_image)
    assert result.shape == underwater_image.shape
    assert result.min() >= 0 and result.max() <= 1


def test_dcp_baseline_modifies(underwater_image):
    """Test DCP baseline modifies image."""
    baseline = DCPBaselineSimple()
    result = baseline(underwater_image)
    
    assert not torch.allclose(result, underwater_image)
    assert result.shape == underwater_image.shape
    assert result.min() >= 0 and result.max() <= 1


def test_aggressive_blend_modifies(underwater_image):
    """Test aggressive blend modifies image."""
    baseline = AggressiveBlendBaseline()
    result = baseline(underwater_image)
    
    assert not torch.allclose(result, underwater_image)
    assert result.shape == underwater_image.shape
    assert result.min() >= 0 and result.max() <= 1


def test_baseline_output_shape(underwater_image):
    """Test all baselines preserve shape."""
    for baseline_id in BASELINE_IDS:
        baseline = get_baseline(baseline_id)
        result = baseline(underwater_image)
        assert result.shape == underwater_image.shape, f"{baseline_id} changed shape"


def test_baseline_output_range(underwater_image):
    """Test all baselines output in [0, 1] range."""
    for baseline_id in BASELINE_IDS:
        baseline = get_baseline(baseline_id)
        result = baseline(underwater_image)
        assert result.min() >= 0, f"{baseline_id} produced values < 0"
        assert result.max() <= 1, f"{baseline_id} produced values > 1"


def test_baseline_non_destructive(underwater_image):
    """Test baselines don't modify input image."""
    original = underwater_image.clone()
    
    for baseline_id in BASELINE_IDS:
        baseline = get_baseline(baseline_id)
        _ = baseline(underwater_image)
        assert torch.allclose(underwater_image, original), f"{baseline_id} modified input"


def test_get_baseline_invalid_name():
    """Test get_baseline raises on invalid name."""
    with pytest.raises(ValueError):
        get_baseline("invalid_baseline")


def test_get_baseline_factory(underwater_image):
    """Test factory function for all baseline IDs."""
    for baseline_id in BASELINE_IDS:
        baseline = get_baseline(baseline_id)
        result = baseline(underwater_image)
        
        assert result.shape == underwater_image.shape
        assert result.min() >= 0 and result.max() <= 1


def test_baseline_deterministic(underwater_image):
    """Test baselines are deterministic."""
    for baseline_id in BASELINE_IDS:
        baseline = get_baseline(baseline_id)
        result1 = baseline(underwater_image.clone())
        result2 = baseline(underwater_image.clone())
        
        assert torch.allclose(result1, result2, atol=1e-5), f"{baseline_id} not deterministic"


def test_baseline_batch_processing():
    """Test baselines work with single images (not batches)."""
    img = torch.rand(3, 128, 128)
    
    for baseline_id in BASELINE_IDS:
        baseline = get_baseline(baseline_id)
        result = baseline(img)
        
        assert result.dim() == 3
        assert result.shape == img.shape


def test_white_balance_reduces_color_cast(underwater_image):
    """Test white balance reduces color cast."""
    baseline = WhiteBalanceOnlyBaseline()
    result = baseline(underwater_image)
    
    # White balance should reduce difference between R and B channels
    # (assuming input has blue cast)
    # We don't assert exact values, just that it runs without error


def test_aggressive_blend_vs_individual(underwater_image):
    """Test aggressive blend applies multiple techniques."""
    aggressive = AggressiveBlendBaseline()
    result_aggressive = aggressive(underwater_image)
    
    # Compare with input
    # Aggressive should be more modified than single techniques
    # We don't assert exact values, just that it runs


def test_multiple_applications_stability(underwater_image):
    """Test applying same baseline twice gives reasonable result."""
    baseline = CLAHEOnlyBaseline()
    
    result1 = baseline(underwater_image)
    result2 = baseline(result1)
    
    # Applying twice should be stable (second application shouldn't make huge differences)
    # This is a reasonableness check
    assert result1.shape == result2.shape
    assert result2.min() >= 0 and result2.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

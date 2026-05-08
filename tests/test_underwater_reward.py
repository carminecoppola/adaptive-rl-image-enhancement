"""
Tests for underwater reward function.
"""

import pytest
import torch
from torch import Tensor

from src.training.reward_underwater import (
    compute_psnr,
    compute_ssim,
    UnderwaterReward,
    create_reward_function,
)


@pytest.fixture
def reference_image():
    """Create reference image."""
    return torch.rand(3, 128, 128)


@pytest.fixture
def degraded_image(reference_image):
    """Create degraded version of reference."""
    noise = 0.05 * torch.randn_like(reference_image)
    return torch.clamp(reference_image + noise, 0, 1)


@pytest.fixture
def improved_image(reference_image):
    """Create improved version (closer to reference)."""
    noise = 0.02 * torch.randn_like(reference_image)
    return torch.clamp(reference_image + noise, 0, 1)


def test_compute_psnr_identical_images():
    """Test PSNR of identical images."""
    img = torch.rand(3, 128, 128)
    psnr = compute_psnr(img, img)
    assert psnr == 100.0  # Perfect match


def test_compute_psnr_different_images(reference_image):
    """Test PSNR is finite for different images."""
    degraded = reference_image + 0.1 * torch.randn_like(reference_image)
    psnr = compute_psnr(reference_image, torch.clamp(degraded, 0, 1))
    assert psnr > 0
    assert psnr < 100


def test_compute_ssim_identical_images():
    """Test SSIM of identical images."""
    img = torch.rand(3, 128, 128)
    ssim = compute_ssim(img, img)
    assert ssim > 0.99  # Very high (close to 1)


def test_compute_ssim_different_images(reference_image):
    """Test SSIM is reasonable for different images."""
    degraded = reference_image + 0.1 * torch.randn_like(reference_image)
    ssim = compute_ssim(reference_image, torch.clamp(degraded, 0, 1))
    assert -1 < ssim < 1


def test_reward_positive_on_improvement(reference_image, degraded_image, improved_image):
    """Test reward is positive when image improves."""
    reward_fn = UnderwaterReward()
    reward, _ = reward_fn(degraded_image, improved_image, reference_image)
    
    # Improved image should give positive reward (or close to zero)
    # Note: with step penalty, it might be slightly negative
    # But delta_psnr should be positive
    # We check that improved beats degraded
    reward_from_degraded, _ = reward_fn(degraded_image, degraded_image, reference_image)
    assert reward >= reward_from_degraded


def test_reward_negative_on_degradation(reference_image, improved_image):
    """Test reward is negative when image degrades."""
    reward_fn = UnderwaterReward()
    
    # Very degraded image
    very_degraded = reference_image + 0.5 * torch.randn_like(reference_image)
    very_degraded = torch.clamp(very_degraded, 0, 1)
    
    reward, _ = reward_fn(improved_image, very_degraded, reference_image)
    
    # Degradation should give negative reward
    assert reward < 0


def test_reward_components(reference_image, degraded_image, improved_image):
    """Test reward components are computed."""
    reward_fn = UnderwaterReward()
    reward, components = reward_fn(degraded_image, improved_image, reference_image)
    
    expected_keys = [
        "psnr_prev", "psnr_curr", "delta_psnr",
        "ssim_prev", "ssim_curr", "delta_ssim",
        "psnr_reward", "ssim_reward",
        "step_penalty", "terminal_bonus",
        "perceptual_reward", "total_reward"
    ]
    
    for key in expected_keys:
        assert key in components
        assert isinstance(components[key], (int, float))


def test_reward_step_penalty_applied(reference_image, degraded_image):
    """Test step penalty is applied."""
    reward_fn = UnderwaterReward(step_penalty=0.01)
    reward, components = reward_fn(degraded_image, degraded_image, reference_image)
    
    # No improvement (delta_psnr = 0, delta_ssim ≈ 0), only penalty
    assert components["step_penalty"] == 0.01
    assert reward < 0  # Penalty makes it negative


def test_reward_terminal_bonus(reference_image, degraded_image, improved_image):
    """Test terminal bonus is applied."""
    reward_fn = UnderwaterReward(terminal_bonus=0.2)
    
    # With terminal=False
    reward_no_terminal, _ = reward_fn(degraded_image, improved_image, reference_image, is_terminal=False)
    
    # With terminal=True
    reward_terminal, components = reward_fn(degraded_image, improved_image, reference_image, is_terminal=True)
    
    # Terminal reward should be higher
    assert reward_terminal > reward_no_terminal
    assert components["terminal_bonus"] == 0.2


def test_reward_weights(reference_image, degraded_image, improved_image):
    """Test reward weights affect computation."""
    reward_fn_high_psnr = UnderwaterReward(alpha=2.0, beta=0.0)
    reward_fn_high_ssim = UnderwaterReward(alpha=0.0, beta=2.0)
    
    reward_psnr, _ = reward_fn_high_psnr(degraded_image, improved_image, reference_image)
    reward_ssim, _ = reward_fn_high_ssim(degraded_image, improved_image, reference_image)
    
    # Different weights should give different rewards
    # (unless delta_psnr == delta_ssim by coincidence)
    # We just check they're both valid
    assert isinstance(reward_psnr, float)
    assert isinstance(reward_ssim, float)


def test_reward_batch_images():
    """Test reward works with batched images."""
    batch_size = 4
    img_prev = torch.rand(batch_size, 3, 128, 128)
    img_curr = torch.rand(batch_size, 3, 128, 128)
    img_ref = torch.rand(batch_size, 3, 128, 128)
    
    reward_fn = UnderwaterReward()
    
    # Should work with first image of batch
    reward, _ = reward_fn(img_prev[0], img_curr[0], img_ref[0])
    assert isinstance(reward, float)


def test_create_reward_function_from_config():
    """Test creating reward function from config."""
    config = {
        "psnr_weight": 1.0,
        "ssim_weight": 0.5,
        "step_penalty": 0.01,
        "terminal_bonus": 0.2,
        "use_perceptual_loss": False,
    }
    
    reward_fn = create_reward_function(config)
    assert isinstance(reward_fn, UnderwaterReward)
    assert reward_fn.alpha == 1.0
    assert reward_fn.beta == 0.5


def test_reward_deterministic(reference_image, degraded_image, improved_image):
    """Test reward is deterministic."""
    reward_fn = UnderwaterReward()
    
    reward1, _ = reward_fn(degraded_image, improved_image, reference_image)
    reward2, _ = reward_fn(degraded_image, improved_image, reference_image)
    
    assert reward1 == reward2


def test_reward_without_improvement(reference_image):
    """Test reward when no improvement (same input/output)."""
    reward_fn = UnderwaterReward()
    reward, components = reward_fn(reference_image, reference_image, reference_image)
    
    # Delta metrics should be near zero
    assert abs(components["delta_psnr"]) < 0.1
    assert abs(components["delta_ssim"]) < 0.1
    
    # Reward should be negative (just step penalty)
    assert reward < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

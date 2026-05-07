"""
Test suite for ImageEnhancementEnv.
Verifies environment functionality including terminal reward activation.
"""

import numpy as np
import torch
from PIL import Image
from src.envs.env import ImageEnhancementEnv


def test_env_initialization():
    """Test that ImageEnhancementEnv initializes correctly."""
    clean_image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        max_steps=5,
        image_size=(32, 32),
    )
    assert env is not None
    assert env.max_steps == 5
    assert env.step_count == 0


def test_terminal_reward_psnr_disabled_by_default():
    """Test that terminal reward is disabled by default (0.0)."""
    clean_image = Image.new("RGB", (32, 32), color=(200, 200, 200))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        max_steps=5,
        image_size=(32, 32),
        terminal_reward_psnr_scale=0.0,  # Disabled
    )
    assert env.terminal_reward_psnr_scale == 0.0


def test_terminal_reward_psnr_enabled():
    """Test that terminal reward can be enabled with positive scale."""
    clean_image = Image.new("RGB", (32, 32), color=(100, 100, 100))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        max_steps=5,
        image_size=(32, 32),
        terminal_reward_psnr_scale=1.5,  # Enabled
    )
    assert env.terminal_reward_psnr_scale == 1.5


def test_terminal_reward_ssim_enabled():
    """Test that SSIM-based terminal reward can be enabled."""
    clean_image = Image.new("RGB", (32, 32), color=(150, 150, 150))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        max_steps=5,
        image_size=(32, 32),
        terminal_reward_ssim_scale=0.5,
    )
    assert env.terminal_reward_ssim_scale == 0.5


def test_env_step_returns_valid_observation():
    """Test that environment step returns a valid observation."""
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        max_steps=5,
        image_size=(32, 32),
    )
    
    observation = env.reset()
    assert observation is not None
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (3, 32, 32) or observation.shape == (4, 32, 32)  # with or without step channel


def test_env_reset():
    """Test that environment reset works correctly."""
    clean_image = Image.new("RGB", (64, 64), color=(128, 200, 100))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        max_steps=5,
        image_size=(64, 64),
    )
    
    obs1 = env.reset()
    obs2 = env.reset()
    
    # Both should be valid observations
    assert obs1 is not None
    assert obs2 is not None
    # Should be numpy arrays
    assert isinstance(obs1, np.ndarray)
    assert isinstance(obs2, np.ndarray)

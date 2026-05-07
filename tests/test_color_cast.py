"""
Test suite for color cast metrics.
Verifies color cast detection and scoring.
"""

import pytest
import numpy as np
from PIL import Image
from src.metrics.color_cast import compute_color_cast_score, get_dominant_color_channel


@pytest.fixture
def neutral_image() -> Image.Image:
    """Create a neutral gray image with balanced colors."""
    array = np.ones((64, 64, 3), dtype=np.uint8) * 128
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def blue_cast_image() -> Image.Image:
    """Create an image with strong blue cast (underwater-like)."""
    array = np.ones((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = 80   # Red: low
    array[:, :, 1] = 100  # Green: low
    array[:, :, 2] = 200  # Blue: high
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def red_cast_image() -> Image.Image:
    """Create an image with strong red cast."""
    array = np.ones((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = 200  # Red: high
    array[:, :, 1] = 100  # Green: low
    array[:, :, 2] = 100  # Blue: low
    return Image.fromarray(array, mode="RGB")


def test_compute_color_cast_score_neutral(neutral_image):
    """Test that neutral image has low color cast score."""
    score = compute_color_cast_score(neutral_image)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Neutral image should have very low score (close to 0)
    assert score < 0.1


def test_compute_color_cast_score_blue_cast(blue_cast_image):
    """Test that blue-cast image has higher color cast score."""
    neutral_score = compute_color_cast_score(
        Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 128, mode="RGB")
    )
    blue_score = compute_color_cast_score(blue_cast_image)
    
    # Blue cast should have higher score than neutral
    assert blue_score > neutral_score


def test_compute_color_cast_score_range():
    """Test that color cast score stays within [0, 1]."""
    for _ in range(10):
        array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(array, mode="RGB")
        score = compute_color_cast_score(img)
        
        assert 0.0 <= score <= 1.0


def test_get_dominant_color_channel_neutral(neutral_image):
    """Test that neutral image has no dominant channel."""
    dominant = get_dominant_color_channel(neutral_image)
    assert dominant == "neutral"


def test_get_dominant_color_channel_blue(blue_cast_image):
    """Test that blue-cast image is identified as blue."""
    dominant = get_dominant_color_channel(blue_cast_image)
    assert dominant == "blue"


def test_get_dominant_color_channel_red(red_cast_image):
    """Test that red-cast image is identified as red."""
    dominant = get_dominant_color_channel(red_cast_image)
    assert dominant == "red"


def test_compute_color_cast_score_consistency():
    """Test that score is consistent for same image."""
    array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(array, mode="RGB")
    
    score1 = compute_color_cast_score(img)
    score2 = compute_color_cast_score(img)
    
    assert score1 == score2

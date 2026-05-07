"""
Color cast detection and scoring for underwater images.

Computes a color cast metric that measures the dominance of specific
color channels (especially blue/green in underwater environments).
Lower scores indicate better white balance.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import cv2


def compute_color_cast_score(image: Image.Image) -> float:
    """
    Compute a color cast score measuring color imbalance.
    
    Computes mean values in LAB color space for a* and b* channels.
    - a* > 128: image is greenish (undersea blue cast shifted to green in LAB)
    - a* < 128: image is reddish
    - b* > 128: image is yellowish
    - b* < 128: image is bluish (typical underwater)
    
    Returns a score in [0, 1] where 0 = neutral balance, 1 = extreme cast.
    """
    array = np.asarray(image).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Extract a* and b* channels
    a_channel = lab[:, :, 1]  # Green-red axis: 0=green, 128=neutral, 255=red
    b_channel = lab[:, :, 2]  # Yellow-blue axis: 0=yellow, 128=neutral, 255=blue
    
    # Compute deviation from neutral (128)
    a_deviation = np.abs(a_channel.mean() - 128.0) / 128.0
    b_deviation = np.abs(b_channel.mean() - 128.0) / 128.0
    
    # Combined score: average of both deviations, clipped to [0, 1]
    score = np.clip((a_deviation + b_deviation) / 2.0, 0.0, 1.0)
    
    return float(score)


def get_dominant_color_channel(image: Image.Image) -> str:
    """
    Identify which color channel is dominant (R, G, or B).
    
    Returns one of: "red", "green", "blue", "neutral"
    """
    array = np.asarray(image).astype(np.float32)
    
    r_mean = array[:, :, 0].mean()
    g_mean = array[:, :, 1].mean()
    b_mean = array[:, :, 2].mean()
    
    channels = {"red": r_mean, "green": g_mean, "blue": b_mean}
    max_channel = max(channels, key=channels.get)
    
    # Check if dominant channel is significantly higher than others
    avg = (r_mean + g_mean + b_mean) / 3.0
    dominant_value = channels[max_channel]
    
    if dominant_value > avg * 1.1:  # 10% threshold
        return max_channel
    return "neutral"

"""
Gymnasium environment for adaptive image enhancement.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from src.actions.filters import ImageAction, apply_action, get_action_name
from src.metrics import compute_psnr, compute_ssim
from src.metrics.color_cast import compute_color_cast_score


class ImageEnhancementEnv(gym.Env):
    """
    RL environment where an agent improves a degraded image through
    sequential image-processing actions.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        clean_image: Image.Image,
        degraded_image: Image.Image,
        max_steps: int = 10,
        image_size: tuple[int, int] = (32, 32),
        reward_metric: str = "psnr",
        step_penalty: float = 0.01,
        repeated_action_penalty: float = 0.0,
        no_improvement_penalty: float = 0.0,
        stop_bonus_scale: float = 0.0,
        stop_no_improvement_penalty: float = 0.0,
        early_stop_min_improvement: float = 0.0,
        truncate_without_stop_penalty: float = 0.0,
        stop_action_bonus: float = 0.0,
        terminal_reward_psnr_scale: float = 0.0,
        terminal_reward_ssim_scale: float = 0.0,
        color_cast_weight: float = 0.0,
        color_cast_improvement_scale: float = 0.5,
        include_step_channel: bool = True,
    ) -> None:
        super().__init__()

        clean_rgb = clean_image.convert("RGB")
        degraded_rgb = degraded_image.convert("RGB")
        self.clean_image = (
            clean_rgb.copy()
            if clean_rgb.size == image_size
            else clean_rgb.resize(image_size, Image.Resampling.BICUBIC)
        )
        self.initial_degraded_image = (
            degraded_rgb.copy()
            if degraded_rgb.size == image_size
            else degraded_rgb.resize(image_size, Image.Resampling.BICUBIC)
        )

        self.max_steps = max_steps
        self.image_size = image_size
        self.reward_metric = reward_metric
        self.step_penalty = step_penalty
        self.repeated_action_penalty = repeated_action_penalty
        self.no_improvement_penalty = no_improvement_penalty
        self.stop_bonus_scale = stop_bonus_scale
        self.stop_no_improvement_penalty = stop_no_improvement_penalty
        self.early_stop_min_improvement = early_stop_min_improvement
        self.truncate_without_stop_penalty = truncate_without_stop_penalty
        self.stop_action_bonus = stop_action_bonus
        self.terminal_reward_psnr_scale = terminal_reward_psnr_scale
        self.terminal_reward_ssim_scale = terminal_reward_ssim_scale
        self.color_cast_weight = color_cast_weight
        self.color_cast_improvement_scale = color_cast_improvement_scale
        self.include_step_channel = include_step_channel

        self.num_actions = len(ImageAction)
        self.stop_action = int(ImageAction.STOP)

        height, width = image_size[1], image_size[0]

        channels = 4 if self.include_step_channel else 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(height, width, channels),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.num_actions)

        self.current_image: Image.Image | None = None
        self.current_step = 0
        self.previous_quality = 0.0
        self.initial_quality = 0.0
        self.previous_action: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.current_image = self.initial_degraded_image.copy()
        self.current_step = 0
        self.previous_quality = self._compute_quality(self.current_image)
        self.initial_quality = self.previous_quality
        self.initial_color_cast = compute_color_cast_score(self.current_image) if self.color_cast_weight > 0 else 0.0
        self.previous_color_cast = self.initial_color_cast
        self.previous_action = None

        observation = self._image_to_observation(self.current_image)

        info = {
            "step": self.current_step,
            "quality": self.previous_quality,
            "psnr": compute_psnr(self.current_image, self.clean_image),
            "ssim": compute_ssim(self.current_image, self.clean_image),
        }

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_image is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.current_step += 1

        terminated = action == self.stop_action

        previous_quality = self.previous_quality

        if not terminated:
            self.current_image = apply_action(self.current_image, action)

        current_quality = self._compute_quality(self.current_image)

        delta_quality = current_quality - previous_quality
        step_penalty_applied = self.step_penalty if not terminated else 0.0
        repeated_penalty_applied = 0.0
        no_improvement_penalty_applied = 0.0
        stop_bonus_applied = 0.0
        stop_no_improvement_penalty_applied = 0.0
        truncate_without_stop_penalty_applied = 0.0
        stop_action_bonus_applied = 0.0
        terminal_psnr_reward_applied = 0.0
        terminal_ssim_reward_applied = 0.0
        color_cast_reward_applied = 0.0

        if not terminated and self.previous_action is not None and action == self.previous_action:
            repeated_penalty_applied = self.repeated_action_penalty

        if not terminated and delta_quality <= 0.0:
            no_improvement_penalty_applied = self.no_improvement_penalty

        if terminated:
            improvement_vs_initial = current_quality - self.initial_quality
            if improvement_vs_initial >= self.early_stop_min_improvement:
                stop_bonus_applied = self.stop_bonus_scale * improvement_vs_initial
            else:
                stop_no_improvement_penalty_applied = self.stop_no_improvement_penalty
            stop_action_bonus_applied = self.stop_action_bonus
            if self.terminal_reward_psnr_scale > 0.0:
                psnr_now = compute_psnr(self.current_image, self.clean_image)
                psnr_initial = compute_psnr(self.initial_degraded_image, self.clean_image)
                terminal_psnr_reward_applied = self.terminal_reward_psnr_scale * (psnr_now - psnr_initial)
            if self.terminal_reward_ssim_scale > 0.0:
                ssim_now = compute_ssim(self.current_image, self.clean_image)
                ssim_initial = compute_ssim(self.initial_degraded_image, self.clean_image)
                terminal_ssim_reward_applied = self.terminal_reward_ssim_scale * (ssim_now - ssim_initial)

        if self.color_cast_weight > 0.0:
            current_color_cast = compute_color_cast_score(self.current_image)
            delta_color_cast = self.previous_color_cast - current_color_cast  # Negative is good (lower cast)
            color_cast_reward_applied = self.color_cast_weight * self.color_cast_improvement_scale * delta_color_cast
            self.previous_color_cast = current_color_cast

        reward = (
            delta_quality
            - step_penalty_applied
            - repeated_penalty_applied
            - no_improvement_penalty_applied
            + stop_bonus_applied
            + stop_action_bonus_applied
            + terminal_psnr_reward_applied
            + terminal_ssim_reward_applied
            + color_cast_reward_applied
            - stop_no_improvement_penalty_applied
        )

        self.previous_quality = current_quality
        self.previous_action = action

        truncated = self.current_step >= self.max_steps
        if truncated and not terminated:
            truncate_without_stop_penalty_applied = self.truncate_without_stop_penalty

        reward -= truncate_without_stop_penalty_applied

        observation = self._image_to_observation(self.current_image)

        info = {
            "step": self.current_step,
            "action": action,
            "action_name": get_action_name(action),
            "quality": current_quality,
            "previous_quality": previous_quality,
            "delta_quality": float(delta_quality),
            "step_penalty_applied": float(step_penalty_applied),
            "repeated_penalty_applied": float(repeated_penalty_applied),
            "no_improvement_penalty_applied": float(no_improvement_penalty_applied),
            "stop_bonus_applied": float(stop_bonus_applied),
            "stop_action_bonus_applied": float(stop_action_bonus_applied),
            "terminal_psnr_reward_applied": float(terminal_psnr_reward_applied),
            "terminal_ssim_reward_applied": float(terminal_ssim_reward_applied),
            "color_cast_reward_applied": float(color_cast_reward_applied),
            "stop_no_improvement_penalty_applied": float(stop_no_improvement_penalty_applied),
            "truncate_without_stop_penalty_applied": float(truncate_without_stop_penalty_applied),
            "early_stop_min_improvement": float(self.early_stop_min_improvement),
            "reward": float(reward),
            "psnr": compute_psnr(self.current_image, self.clean_image),
            "ssim": compute_ssim(self.current_image, self.clean_image),
            "color_cast_score": compute_color_cast_score(self.current_image) if self.color_cast_weight > 0 else 0.0,
            "terminated": terminated,
            "truncated": truncated,
        }

        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.current_image is None:
            raise RuntimeError("Environment must be reset before calling render().")

        return np.asarray(self.current_image.convert("RGB"), dtype=np.uint8)

    def _compute_quality(self, image: Image.Image) -> float:
        if self.reward_metric == "psnr":
            return compute_psnr(image, self.clean_image)

        if self.reward_metric == "ssim":
            return compute_ssim(image, self.clean_image)

        if self.reward_metric == "combined":
            psnr = compute_psnr(image, self.clean_image)
            ssim = compute_ssim(image, self.clean_image)
            return psnr + 10.0 * ssim

        raise ValueError(f"Unsupported reward metric: {self.reward_metric}")

    def _image_to_observation(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        rgb = array / 255.0
        if not self.include_step_channel:
            return rgb

        step_ratio = min(1.0, self.current_step / max(1, self.max_steps))
        step_plane = np.full(
            (rgb.shape[0], rgb.shape[1], 1),
            fill_value=step_ratio,
            dtype=np.float32,
        )
        return np.concatenate([rgb, step_plane], axis=2)

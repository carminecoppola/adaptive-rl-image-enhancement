"""
Gymnasium environment for adaptive image enhancement.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from src.actions import apply_action_to_pil, get_action_name, get_num_actions, get_stop_action_id
from src.metrics import compute_psnr, compute_ssim


class ImageEnhancementEnv(gym.Env):
    """
    Sequential underwater image-enhancement task.

    The policy observes only the current image and optional context channels;
    the clean/reference image is kept inside the environment exclusively for
    reward and evaluation metrics. This separation prevents target leakage.
    Every action is an interpretable, deterministic image-processing operator,
    and the episode ends with STOP or the configured step limit.
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
        psnr_weight: float = 1.0,
        ssim_weight: float = 10.0,
        include_step_channel: bool = True,
        include_lab_stats: bool = False,
        action_set_name: str = "general",
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
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.include_step_channel = include_step_channel
        self.include_lab_stats = include_lab_stats
        self.action_set_name = action_set_name

        self.num_actions = get_num_actions(action_set_name)
        self.stop_action = get_stop_action_id(action_set_name)

        height, width = image_size[1], image_size[0]

        step_ch = 1 if self.include_step_channel else 0
        lab_ch = 1 if self.include_lab_stats else 0
        channels = 3 + step_ch + lab_ch
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

        # Start every episode from the untouched degraded input. Copies are
        # important because PIL operations may otherwise leak state between
        # episodes that reuse the same dataset sample.
        self.current_image = self.initial_degraded_image.copy()
        self.current_step = 0
        self.previous_quality = self._compute_quality(self.current_image)
        self.initial_quality = self.previous_quality
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

        # STOP is a real policy decision: it preserves the current image and
        # triggers terminal quality terms instead of applying another filter.
        terminated = action == self.stop_action

        previous_quality = self.previous_quality

        if not terminated:
            self.current_image = apply_action_to_pil(
                self.current_image, action, self.action_set_name
            )

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
                terminal_psnr_reward_applied = self.terminal_reward_psnr_scale * (
                    psnr_now - psnr_initial
                )
            if self.terminal_reward_ssim_scale > 0.0:
                ssim_now = compute_ssim(self.current_image, self.clean_image)
                ssim_initial = compute_ssim(self.initial_degraded_image, self.clean_image)
                terminal_ssim_reward_applied = self.terminal_reward_ssim_scale * (
                    ssim_now - ssim_initial
                )

        # Keep the transition-quality signal separate from behavioral shaping.
        # This decomposition is also written to ``info`` so reward failures can
        # be diagnosed after a run instead of inferred from one aggregate value.
        reward = (
            delta_quality
            - step_penalty_applied
            - repeated_penalty_applied
            - no_improvement_penalty_applied
            + stop_bonus_applied
            + stop_action_bonus_applied
            + terminal_psnr_reward_applied
            + terminal_ssim_reward_applied
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
            "action_name": get_action_name(self.action_set_name, action),
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
            "stop_no_improvement_penalty_applied": float(stop_no_improvement_penalty_applied),
            "truncate_without_stop_penalty_applied": float(truncate_without_stop_penalty_applied),
            "early_stop_min_improvement": float(self.early_stop_min_improvement),
            "reward": float(reward),
            "psnr": compute_psnr(self.current_image, self.clean_image),
            "ssim": compute_ssim(self.current_image, self.clean_image),
            "terminated": terminated,
            "truncated": truncated,
        }

        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.current_image is None:
            raise RuntimeError("Environment must be reset before calling render().")

        return np.asarray(self.current_image.convert("RGB"), dtype=np.uint8)

    def _compute_quality(self, image: Image.Image) -> float:
        """Return the reference-based quality objective used by the reward.

        In the official underwater configuration, SSIM receives a larger
        numeric weight because its natural scale is much smaller than PSNR's.
        Checkpoint selection still uses mean delta PSNR as the primary metric.
        """
        if self.reward_metric == "psnr":
            return compute_psnr(image, self.clean_image)

        if self.reward_metric == "ssim":
            return compute_ssim(image, self.clean_image)

        if self.reward_metric == "combined":
            psnr = compute_psnr(image, self.clean_image)
            ssim = compute_ssim(image, self.clean_image)
            return self.psnr_weight * psnr + self.ssim_weight * ssim

        raise ValueError(f"Unsupported reward metric: {self.reward_metric}")

    def _image_to_observation(self, image: Image.Image) -> np.ndarray:
        """Build an ``H x W x C`` policy observation without the reference.

        The step plane tells the CNN how much of the five-action budget has
        already been consumed. LAB statistics are optional and disabled in the
        final v4.0 configuration after the corresponding ablation worsened OOD
        behavior.
        """
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        rgb = array / 255.0
        parts = [rgb]

        if self.include_step_channel:
            step_ratio = min(1.0, self.current_step / max(1, self.max_steps))
            step_plane = np.full(
                (rgb.shape[0], rgb.shape[1], 1),
                fill_value=step_ratio,
                dtype=np.float32,
            )
            parts.append(step_plane)

        if self.include_lab_stats:
            import cv2

            arr_uint8 = (rgb * 255).astype(np.uint8)
            lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

            l_mean = lab[:, :, 0].mean() / 255.0
            l_std = lab[:, :, 0].std() / 255.0
            a_mean = (lab[:, :, 1].mean() - 128.0) / 128.0
            a_std = lab[:, :, 1].std() / 128.0
            b_mean = (lab[:, :, 2].mean() - 128.0) / 128.0
            b_std = lab[:, :, 2].std() / 128.0

            h, w = rgb.shape[0], rgb.shape[1]
            lab_value = float(
                np.clip(np.mean([l_mean, l_std, a_mean, a_std, b_mean, b_std]), 0.0, 1.0)
            )
            lab_plane = np.full((h, w, 1), fill_value=lab_value, dtype=np.float32)
            parts.append(lab_plane)

        return np.concatenate(parts, axis=2)

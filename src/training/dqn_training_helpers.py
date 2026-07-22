from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.actions import get_action_name, get_stop_action_id
from src.agents import DQNAgent
from src.data.degradation import degrade_image
from src.envs.env import ImageEnhancementEnv
from src.metrics import compute_psnr, compute_ssim
from src.training.dqn_types import EvalStats


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_action_entropy(actions: list[int], num_actions: int) -> float:
    if not actions:
        return 0.0
    counts = np.zeros(num_actions, dtype=np.float64)
    for action in actions:
        counts[action] += 1.0
    probs = counts / counts.sum()
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum())


def compute_action_repeat_ratio(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return float(repeats / (len(actions) - 1))


def choose_degradation_type(default_type: str, candidate_types: list[str], key: int) -> str:
    if default_type == "none":
        return "none"
    if default_type != "mixed":
        return default_type
    if not candidate_types:
        return "gaussian_noise"
    return candidate_types[key % len(candidate_types)]


def extract_clean_and_degraded_images(sample: tuple[Any, Any]) -> tuple[Image.Image, Image.Image | None]:
    """
    Normalize dataset samples across synthetic and paired datasets.

    Returns:
        clean_image: Reference image used for reward/evaluation.
        degraded_image: Optional degraded input image when dataset is already paired.
    """
    image, metadata = sample
    degraded_image = image.convert("RGB") if isinstance(image, Image.Image) else image

    if isinstance(metadata, dict) and "reference_pil" in metadata:
        reference_pil = metadata["reference_pil"]
        if not isinstance(reference_pil, Image.Image):
            raise TypeError("Expected metadata['reference_pil'] to be a PIL image.")
        return reference_pil.convert("RGB"), degraded_image

    if not isinstance(degraded_image, Image.Image):
        raise TypeError("Expected dataset sample image to be a PIL image.")
    return degraded_image, None


def build_env_for_image(
    clean_image: Image.Image,
    max_steps: int,
    image_size: tuple[int, int],
    reward_metric: str,
    step_penalty: float,
    repeated_action_penalty: float,
    no_improvement_penalty: float,
    stop_bonus_scale: float,
    stop_no_improvement_penalty: float,
    early_stop_min_improvement: float,
    truncate_without_stop_penalty: float,
    stop_action_bonus: float,
    terminal_reward_psnr_scale: float,
    terminal_reward_ssim_scale: float,
    include_step_channel: bool,
    include_lab_stats: bool,
    action_set_name: str,
    degradation_type: str,
    noise_std: float,
    degraded_image: Image.Image | None = None,
    psnr_weight: float = 1.0,
    ssim_weight: float = 10.0,
) -> ImageEnhancementEnv:
    """
    Build environment for a single image.
    
    Args:
        clean_image: Reference high-quality image.
        degraded_image: Optional precomputed degraded image.
    """
    if degraded_image is None:
        if degradation_type == "none":
            degraded_image = clean_image.copy()
        else:
            degraded_image = degrade_image(
                clean_image,
                degradation_type=degradation_type,
                noise_std=noise_std,
            )
    return ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        max_steps=max_steps,
        image_size=image_size,
        reward_metric=reward_metric,
        step_penalty=step_penalty,
        repeated_action_penalty=repeated_action_penalty,
        no_improvement_penalty=no_improvement_penalty,
        stop_bonus_scale=stop_bonus_scale,
        stop_no_improvement_penalty=stop_no_improvement_penalty,
        early_stop_min_improvement=early_stop_min_improvement,
        truncate_without_stop_penalty=truncate_without_stop_penalty,
        stop_action_bonus=stop_action_bonus,
        terminal_reward_psnr_scale=terminal_reward_psnr_scale,
        terminal_reward_ssim_scale=terminal_reward_ssim_scale,
        psnr_weight=psnr_weight,
        ssim_weight=ssim_weight,
        include_step_channel=include_step_channel,
        include_lab_stats=include_lab_stats,
        action_set_name=action_set_name,
    )


def evaluate_on_indices(
    agent: DQNAgent,
    dataset: Any,
    eval_indices: list[int],
    eval_step_seed: int,
    max_steps: int,
    image_size: tuple[int, int],
    reward_metric: str,
    step_penalty: float,
    repeated_action_penalty: float,
    no_improvement_penalty: float,
    stop_bonus_scale: float,
    stop_no_improvement_penalty: float,
    early_stop_min_improvement: float,
    truncate_without_stop_penalty: float,
    stop_action_bonus: float,
    terminal_reward_psnr_scale: float,
    terminal_reward_ssim_scale: float,
    include_step_channel: bool,
    include_lab_stats: bool,
    action_set_name: str,
    default_degradation_type: str,
    candidate_degradation_types: list[str],
    noise_std: float,
    psnr_weight: float = 1.0,
    ssim_weight: float = 10.0,
) -> EvalStats:
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards: list[float] = []
    action_counter: Counter = Counter()
    episode_lengths: list[int] = []
    stop_count = 0
    delta_psnr_values: list[float] = []
    delta_ssim_values: list[float] = []
    stop_action_id = get_stop_action_id(action_set_name)

    for offset, idx in enumerate(eval_indices):
        degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=idx + eval_step_seed,
        )
        clean_image, degraded_image = extract_clean_and_degraded_images(dataset[idx])
        env = build_env_for_image(
            clean_image=clean_image,
            max_steps=max_steps,
            image_size=image_size,
            reward_metric=reward_metric,
            step_penalty=step_penalty,
            repeated_action_penalty=repeated_action_penalty,
            no_improvement_penalty=no_improvement_penalty,
            stop_bonus_scale=stop_bonus_scale,
            stop_no_improvement_penalty=stop_no_improvement_penalty,
            early_stop_min_improvement=early_stop_min_improvement,
            truncate_without_stop_penalty=truncate_without_stop_penalty,
            stop_action_bonus=stop_action_bonus,
            terminal_reward_psnr_scale=terminal_reward_psnr_scale,
            terminal_reward_ssim_scale=terminal_reward_ssim_scale,
            psnr_weight=psnr_weight,
            ssim_weight=ssim_weight,
            include_step_channel=include_step_channel,
            include_lab_stats=include_lab_stats,
            action_set_name=action_set_name,
            degradation_type=degradation_type,
            noise_std=noise_std,
            degraded_image=degraded_image,
        )
        state, _ = env.reset(seed=eval_step_seed + offset)
        clean_eval = env.clean_image
        degraded_eval = env.initial_degraded_image.copy()

        episode_reward = 0.0
        step_actions: list[int] = []
        for _ in range(max_steps):
            action = agent.select_action(state)
            step_actions.append(action)
            action_counter[get_action_name(action_set_name, action)] += 1
            if action == stop_action_id:
                stop_count += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break

        rewards.append(episode_reward)
        episode_lengths.append(len(step_actions))
        final_image = env.current_image
        if final_image is None:
            raise RuntimeError("Environment returned None current_image after rollout.")
        final_eval = final_image.copy()
        delta_psnr_values.append(compute_psnr(final_eval, clean_eval) - compute_psnr(degraded_eval, clean_eval))
        delta_ssim_values.append(compute_ssim(final_eval, clean_eval) - compute_ssim(degraded_eval, clean_eval))

    agent.epsilon = old_epsilon

    total_actions = sum(action_counter.values())
    dominant_action_share = 0.0
    if total_actions > 0:
        dominant_action_share = max(action_counter.values()) / total_actions

    return {
        "mean_eval_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_eval_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_delta_psnr": float(np.mean(delta_psnr_values)) if delta_psnr_values else 0.0,
        "std_delta_psnr": float(np.std(delta_psnr_values)) if delta_psnr_values else 0.0,
        "mean_delta_ssim": float(np.mean(delta_ssim_values)) if delta_ssim_values else 0.0,
        "std_delta_ssim": float(np.std(delta_ssim_values)) if delta_ssim_values else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "stop_rate": float(stop_count / total_actions) if total_actions else 0.0,
        "dominant_action_share": float(dominant_action_share),
    }

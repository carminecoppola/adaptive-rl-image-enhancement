import numpy as np
from PIL import Image

from src.envs.env import ImageEnhancementEnv
from src.metrics import compute_psnr, compute_ssim
from src.training.dqn_training_helpers import build_env_for_image


def make_env() -> ImageEnhancementEnv:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    return ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        max_steps=3,
        image_size=(32, 32),
        include_step_channel=True,
        action_set_name="underwater_curated_v1",
    )


def make_env_with_lab() -> ImageEnhancementEnv:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 140, 180))
    return ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        max_steps=3,
        image_size=(32, 32),
        include_step_channel=True,
        include_lab_stats=True,
        action_set_name="underwater_curated_v1",
    )


def test_env_initialization() -> None:
    env = make_env()
    assert env.max_steps == 3
    assert env.action_space.n == 4


def test_reset_returns_observation_and_info() -> None:
    env = make_env()
    observation, info = env.reset()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (32, 32, 4)
    assert "psnr" in info
    assert "ssim" in info


def test_reset_with_lab_stats_returns_5_channel_observation() -> None:
    env = make_env_with_lab()
    observation, _ = env.reset()
    assert observation.shape == (32, 32, 5)
    assert env.observation_space.shape == (32, 32, 5)


def test_observation_values_stay_in_box_range_with_lab_stats() -> None:
    env = make_env_with_lab()
    observation, _ = env.reset()
    assert np.all(observation >= 0.0)
    assert np.all(observation <= 1.0)


def test_env_without_lab_stats_keeps_4_channel_shape() -> None:
    env = make_env()
    observation, _ = env.reset()
    assert env.observation_space.shape == (32, 32, 4)
    assert observation.shape == (32, 32, 4)


def test_combined_quality_uses_configurable_reward_weights() -> None:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 140, 180))
    psnr_only_env = ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        reward_metric="combined",
        psnr_weight=1.0,
        ssim_weight=0.0,
        include_step_channel=True,
        action_set_name="underwater_curated_v1",
    )
    ssim_only_env = ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        reward_metric="combined",
        psnr_weight=0.0,
        ssim_weight=1.0,
        include_step_channel=True,
        action_set_name="underwater_curated_v1",
    )

    psnr_only_quality = psnr_only_env._compute_quality(psnr_only_env.initial_degraded_image)
    ssim_only_quality = ssim_only_env._compute_quality(ssim_only_env.initial_degraded_image)

    assert psnr_only_quality != ssim_only_quality


def test_combined_quality_matches_legacy_formula_with_default_weights() -> None:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 140, 180))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        reward_metric="combined",
        psnr_weight=1.0,
        ssim_weight=10.0,
        include_step_channel=True,
        action_set_name="underwater_curated_v1",
    )

    quality = env._compute_quality(env.initial_degraded_image)
    psnr = compute_psnr(env.initial_degraded_image, env.clean_image)
    ssim = compute_ssim(env.initial_degraded_image, env.clean_image)
    expected = env.psnr_weight * psnr + env.ssim_weight * ssim
    legacy = psnr + 10.0 * ssim

    assert np.isclose(quality, expected)
    assert np.isclose(quality, legacy)


def test_build_env_for_image_propagates_reward_weights() -> None:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 140, 180))
    env = build_env_for_image(
        clean_image=clean_image,
        max_steps=3,
        image_size=(32, 32),
        reward_metric="combined",
        step_penalty=0.01,
        repeated_action_penalty=0.0,
        no_improvement_penalty=0.0,
        stop_bonus_scale=0.0,
        stop_no_improvement_penalty=0.0,
        early_stop_min_improvement=0.0,
        truncate_without_stop_penalty=0.0,
        stop_action_bonus=0.0,
        terminal_reward_psnr_scale=0.0,
        terminal_reward_ssim_scale=0.0,
        include_step_channel=True,
        include_lab_stats=False,
        action_set_name="underwater_curated_v1",
        degradation_type="none",
        noise_std=0.0,
        degraded_image=degraded_image,
        psnr_weight=1.0,
        ssim_weight=10.0,
    )

    assert env.psnr_weight == 1.0
    assert env.ssim_weight == 10.0


def test_step_returns_valid_transition() -> None:
    env = make_env()
    observation, _ = env.reset()
    next_observation, reward, terminated, truncated, info = env.step(0)
    assert observation.shape == next_observation.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["action_name"] == "white_balance"


def test_stop_action_terminates() -> None:
    env = make_env()
    env.reset()
    _, _, terminated, _, info = env.step(env.stop_action)
    assert terminated is True
    assert info["terminated"] is True


def test_underwater_action_set_initialization() -> None:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        max_steps=5,
        image_size=(32, 32),
        include_step_channel=True,
        action_set_name="underwater_v1",
    )
    assert env.action_space.n == 15
    assert env.stop_action == 14


def test_curated_underwater_action_set_initialization() -> None:
    clean_image = Image.new("RGB", (32, 32), color=(180, 180, 180))
    degraded_image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    env = ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        max_steps=3,
        image_size=(32, 32),
        include_step_channel=True,
        action_set_name="underwater_curated_v1",
    )
    assert env.action_space.n == 4
    assert env.stop_action == 3

import numpy as np
from PIL import Image

from src.envs.env import ImageEnhancementEnv


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

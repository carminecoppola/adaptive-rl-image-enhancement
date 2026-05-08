import numpy as np
import pytest
from PIL import Image

from src.actions.filters import ImageAction, apply_action, get_action_name


@pytest.fixture
def test_image() -> Image.Image:
    array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_action_enum() -> None:
    assert len(ImageAction) == 9
    assert ImageAction.STOP == 8


def test_action_names() -> None:
    expected = {
        0: "increase_brightness",
        1: "decrease_brightness",
        2: "increase_contrast",
        3: "decrease_contrast",
        4: "gaussian_denoise",
        5: "sharpen",
        6: "gamma_up",
        7: "gamma_down",
        8: "stop",
    }
    for action, name in expected.items():
        assert get_action_name(action) == name


def test_apply_action_stop(test_image: Image.Image) -> None:
    result = apply_action(test_image, ImageAction.STOP)
    assert result.tobytes() == test_image.tobytes()


@pytest.mark.parametrize(
    "action",
    [
        ImageAction.INCREASE_BRIGHTNESS,
        ImageAction.DECREASE_BRIGHTNESS,
        ImageAction.INCREASE_CONTRAST,
        ImageAction.DECREASE_CONTRAST,
        ImageAction.GAUSSIAN_DENOISE,
        ImageAction.SHARPEN,
        ImageAction.GAMMA_UP,
        ImageAction.GAMMA_DOWN,
    ],
)
def test_standard_actions_return_rgb_images(test_image: Image.Image, action: ImageAction) -> None:
    result = apply_action(test_image, action)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == test_image.size


def test_invalid_action_raises(test_image: Image.Image) -> None:
    with pytest.raises(ValueError):
        apply_action(test_image, 999)

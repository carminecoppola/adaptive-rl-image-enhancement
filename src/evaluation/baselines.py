"""
Classical/manual baselines for adaptive image enhancement.
"""

from PIL import Image

from src.actions.filters import ImageAction, apply_action
from src.metrics import compute_psnr, compute_ssim


BASELINE_POLICIES: dict[str, list[int]] = {
    "input_only": [
        int(ImageAction.STOP),
    ],
    "brightness_contrast_sharpen": [
        int(ImageAction.INCREASE_BRIGHTNESS),
        int(ImageAction.INCREASE_CONTRAST),
        int(ImageAction.SHARPEN),
        int(ImageAction.STOP),
    ],
    "brightness_gamma_contrast": [
        int(ImageAction.INCREASE_BRIGHTNESS),
        int(ImageAction.GAMMA_UP),
        int(ImageAction.INCREASE_CONTRAST),
        int(ImageAction.STOP),
    ],
    "denoise_sharpen": [
        int(ImageAction.GAUSSIAN_DENOISE),
        int(ImageAction.SHARPEN),
        int(ImageAction.STOP),
    ],
}


def apply_baseline_policy(
    degraded_image: Image.Image,
    actions: list[int],
) -> Image.Image:
    current_image = degraded_image.copy()

    for action in actions:
        if action == int(ImageAction.STOP):
            break

        current_image = apply_action(current_image, action)

    return current_image


def evaluate_baseline_policy(
    clean_image: Image.Image,
    degraded_image: Image.Image,
    actions: list[int],
) -> dict[str, float]:
    enhanced_image = apply_baseline_policy(degraded_image, actions)

    psnr_degraded = compute_psnr(degraded_image, clean_image)
    psnr_enhanced = compute_psnr(enhanced_image, clean_image)

    ssim_degraded = compute_ssim(degraded_image, clean_image)
    ssim_enhanced = compute_ssim(enhanced_image, clean_image)

    return {
        "psnr_degraded": psnr_degraded,
        "psnr_enhanced": psnr_enhanced,
        "delta_psnr": psnr_enhanced - psnr_degraded,
        "ssim_degraded": ssim_degraded,
        "ssim_enhanced": ssim_enhanced,
        "delta_ssim": ssim_enhanced - ssim_degraded,
        "num_actions": float(len([a for a in actions if a != int(ImageAction.STOP)])),
    }


def evaluate_all_baselines(
    clean_image: Image.Image,
    degraded_image: Image.Image,
) -> dict[str, dict[str, float]]:
    results = {}

    for name, actions in BASELINE_POLICIES.items():
        results[name] = evaluate_baseline_policy(
            clean_image=clean_image,
            degraded_image=degraded_image,
            actions=actions,
        )

    return results
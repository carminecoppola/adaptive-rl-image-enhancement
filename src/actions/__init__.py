"""
Action registry system for RL image enhancement.

Supports multiple action sets:
- "general": Generic actions (brightness, contrast, sharpen, etc.)
- "underwater_v1": Underwater-specific 15-action set.
- "underwater_curated_v1": Small underwater-focused action set tuned for stable RL.
- "underwater_extended_v1": OOD-focused underwater action set with broader coverage.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.actions import filters
from src.actions import underwater_v1


# Registry of action sets by name
ACTION_SETS = {
    "general": {
        "actions": filters,  # Use filters module for backward compatibility
        "action_names": filters.ACTION_NAMES,
        "num_actions": 9,  # 0-8 (STOP is 8)
        "apply_tensor_action": filters.apply_action,
    },
    "underwater_v1": {
        "actions": underwater_v1,
        "action_names": underwater_v1.ACTION_NAMES,
        "action_descriptions": underwater_v1.ACTION_DESCRIPTIONS,
        "num_actions": 15,  # 0-14 (STOP is 14)
        "apply_tensor_action": underwater_v1.apply_action,
    },
    "underwater_curated_v1": {
        "actions": underwater_v1,
        "action_names": underwater_v1.CURATED_ACTION_NAMES,
        "action_descriptions": underwater_v1.CURATED_ACTION_DESCRIPTIONS,
        "num_actions": 4,  # 0-3 (STOP is 3)
        "apply_tensor_action": underwater_v1.apply_action_curated,
    },
    "underwater_extended_v1": {
        "actions": underwater_v1,
        "action_names": underwater_v1.EXTENDED_ACTION_NAMES,
        "action_descriptions": underwater_v1.EXTENDED_ACTION_DESCRIPTIONS,
        "num_actions": 8,  # 0-7 (STOP is 7)
        "apply_tensor_action": underwater_v1.apply_action_extended,
    },
}


def get_action_set(name: str):
    """
    Get action set by name.
    
    Args:
        name: Action set name ("general" or "underwater_v1")
    
    Returns:
        Action set module/dict
    """
    if name not in ACTION_SETS:
        raise ValueError(f"Unknown action set: {name}. Available: {list(ACTION_SETS.keys())}")
    return ACTION_SETS[name]["actions"]


def get_num_actions(name: str) -> int:
    """Get number of actions in a set."""
    if name not in ACTION_SETS:
        raise ValueError(f"Unknown action set: {name}")
    return ACTION_SETS[name]["num_actions"]


def get_action_names(name: str) -> dict:
    """Get action name mapping for a set."""
    if name not in ACTION_SETS:
        raise ValueError(f"Unknown action set: {name}")
    return ACTION_SETS[name]["action_names"]


def get_action_descriptions(name: str) -> dict:
    """Get action description mapping for a set when available."""
    if name not in ACTION_SETS:
        raise ValueError(f"Unknown action set: {name}")
    return ACTION_SETS[name].get("action_descriptions", {})


def get_stop_action_id(name: str) -> int:
    """Get the STOP action id for the selected action set."""
    names = get_action_names(name)
    for action_id, action_name in names.items():
        if str(action_name).lower() == "stop":
            return int(action_id)
    raise ValueError(f"Action set {name} does not define a STOP action.")


def get_action_name(name: str, action: int) -> str:
    """Resolve an action id to its display name for a specific action set."""
    names = get_action_names(name)
    if action not in names:
        raise ValueError(f"Unknown action id {action} for action set {name}")
    return str(names[action])


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def apply_action_to_pil(image: Image.Image, action: int, action_set_name: str) -> Image.Image:
    """Apply an action from the selected action set to a PIL image."""
    if action_set_name == "general":
        return filters.apply_action(image, action)

    if action_set_name not in ACTION_SETS:
        raise ValueError(f"Unsupported action set: {action_set_name}")

    tensor_image = _pil_to_tensor(image)
    apply_tensor_action = ACTION_SETS[action_set_name].get("apply_tensor_action")
    if apply_tensor_action is None:
        raise ValueError(f"Action set {action_set_name} does not define a tensor action handler.")
    enhanced = apply_tensor_action(tensor_image, action)
    return _tensor_to_pil(enhanced)


__all__ = [
    "ACTION_SETS",
    "get_action_set",
    "get_num_actions",
    "get_action_names",
    "get_action_descriptions",
    "get_stop_action_id",
    "get_action_name",
    "apply_action_to_pil",
    "filters",
    "underwater_v1",
]

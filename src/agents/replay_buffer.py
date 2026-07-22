"""Fixed-capacity replay memory for off-policy value learning."""

from __future__ import annotations

import random
from collections import deque
from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

State: TypeAlias = NDArray[np.floating] | Tensor
Transition: TypeAlias = tuple[State, int, float, State, bool]


class ReplayBuffer:
    """Store recent transitions and sample uncorrelated training batches."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer of {len(self.buffer)}"
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        next_states_tensor = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=device,
        )

        # Environment observations are NHWC; convolutional networks expect NCHW.
        states_tensor = states_tensor.permute(0, 3, 1, 2)
        next_states_tensor = next_states_tensor.permute(0, 3, 1, 2)

        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def __len__(self) -> int:
        return len(self.buffer)

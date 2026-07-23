"""Convolutional Q-network used by the value-based image-enhancement agent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class QNetwork(nn.Module):
    """Estimate one action value for each discrete enhancement operation.

    The network accepts image observations in ``NCHW`` format. The optional
    dueling heads are retained for experiments, while the canonical v4.0
    configuration uses the single Q-head.
    """

    MINIMUM_INPUT_SIZE = 128

    def __init__(
        self,
        num_actions: int,
        in_channels: int = 3,
        use_dueling_dqn: bool = False,
    ) -> None:
        super().__init__()
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self.in_channels = in_channels
        self.use_dueling_dqn = use_dueling_dqn

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(
                1,
                in_channels,
                self.MINIMUM_INPUT_SIZE,
                self.MINIMUM_INPUT_SIZE,
            )
            feature_dim = self.features(sample).shape[1]

        if self.use_dueling_dqn:
            self.value_head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )

    def forward(self, observations: Tensor) -> Tensor:
        """Return Q-values for a batch of image observations."""
        observations = observations.float()
        # Tolerate both [0, 1] normalized inputs and raw [0, 255] uint-range
        # tensors so the same network works from the env, from tests with
        # synthetic images, and from ad-hoc inference scripts.
        if observations.numel() > 0 and observations.detach().max().item() > 1.0:
            observations = observations / 255.0

        if (
            observations.shape[-2] < self.MINIMUM_INPUT_SIZE
            or observations.shape[-1] < self.MINIMUM_INPUT_SIZE
        ):
            # The conv stack's receptive field/stride assumes at least
            # 128x128; smaller images (e.g. the 32x32 ones used in unit
            # tests) are upsampled first instead of failing or needing a
            # second architecture.
            observations = F.interpolate(
                observations,
                size=(self.MINIMUM_INPUT_SIZE, self.MINIMUM_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        features = self.features(observations)
        if not self.use_dueling_dqn:
            # Canonical v4.0 path: one Q-value per action, directly.
            return self.q_head(features)

        # Dueling architecture (experimental, unused in v4.0): split value
        # into a state-value term and a per-action advantage term. Subtracting
        # the mean advantage anchors the decomposition (otherwise value and
        # advantage could drift by an arbitrary constant and still sum to the
        # same Q-value, making the split ill-defined).
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# Backward-compatible import for external notebooks and old scripts. New code
# should use ``QNetwork`` because this module represents a network, not the
# complete DQN algorithm.
DQN = QNetwork

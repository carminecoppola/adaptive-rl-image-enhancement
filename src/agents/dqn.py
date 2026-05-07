import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, num_actions: int, in_channels: int = 3, use_dueling_dqn: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.use_dueling_dqn = use_dueling_dqn

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            feature_dim = self.features(dummy).shape[1]

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
                nn.Linear(512, num_actions)
            )

    def forward(self, x):
        x = x / 255.0 if x.max() > 1.0 else x
        # Keep legacy checkpoint compatibility: existing conv stack was designed
        # around 128x128 inputs, so smaller observations are upsampled on-the-fly.
        if x.shape[-2] < 128 or x.shape[-1] < 128:
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.features(x)
        if self.use_dueling_dqn:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_head(x)

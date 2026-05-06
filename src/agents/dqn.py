import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_actions: int, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

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

        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0 if x.max() > 1.0 else x
        x = self.features(x)
        return self.q_head(x)

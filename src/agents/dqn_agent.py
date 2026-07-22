"""Value-based agent supporting standard DQN and Double DQN targets."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.q_network import QNetwork


class DQNAgent:
    """Train and evaluate a discrete-action image-enhancement policy.

    ``use_double_dqn=True`` separates next-action selection (policy network)
    from evaluation (target network). Set it to ``False`` to recover the
    standard DQN target while preserving the same training interface.
    """
    def __init__(
        self,
        num_actions: int,
        in_channels: int = 3,
        epsilon: float = 1.0,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 32,
        use_double_dqn: bool = True,
        use_dueling_dqn: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.use_dueling_dqn = use_dueling_dqn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels

        self.policy_net = QNetwork(
            num_actions,
            in_channels=in_channels,
            use_dueling_dqn=use_dueling_dqn,
        ).to(self.device)
        self.target_net = QNetwork(
            num_actions,
            in_channels=in_channels,
            use_dueling_dqn=use_dueling_dqn,
        ).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray) -> int:
        """Choose an action with the epsilon-greedy exploration strategy.

        The environment exposes observations as ``H x W x C`` NumPy arrays,
        while PyTorch convolutional layers expect ``N x C x H x W`` tensors.
        Evaluation scripts set ``epsilon`` to zero, making this method fully
        greedy and deterministic for a fixed model and input state.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.ndim == 3:
            state_tensor = state_tensor.unsqueeze(0)

        state_tensor = state_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(self, replay_buffer: Any) -> float | None:
        """Perform one replay-based Q-learning update.

        Double DQN deliberately uses the policy network to *select* the next
        action and the target network to *evaluate* it. Keeping those roles
        separate reduces the optimistic value estimates produced by standard
        DQN. Non-finite values are rejected here so one corrupted transition
        cannot destabilize the rest of a long HPC run.
        """
        if len(replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            self.batch_size,
            device=self.device,
        )
        rewards = torch.nan_to_num(
            rewards,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        current_q_values = current_q_values.squeeze(1)
        if not torch.isfinite(current_q_values).all():
            return None

        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: action selection from policy_net, action evaluation from target_net.
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)
            target_q_values = torch.nan_to_num(
                target_q_values,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0,
            ).clamp(-20.0, 20.0)

        loss = self.loss_fn(current_q_values, target_q_values)
        if not torch.isfinite(loss):
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Synchronize the slowly updated target with the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

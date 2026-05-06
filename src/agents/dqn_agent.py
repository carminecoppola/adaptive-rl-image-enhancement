import random
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.dqn import DQN


class DQNAgent:
    def __init__(
        self,
        num_actions: int,
        in_channels: int = 3,
        epsilon: float = 1.0,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 32,
        device=None,
    ):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels

        self.policy_net = DQN(num_actions, in_channels=in_channels).to(self.device)
        self.target_net = DQN(num_actions, in_channels=in_channels).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.ndim == 3:
            state_tensor = state_tensor.unsqueeze(0)

        state_tensor = state_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            self.batch_size,
            device=self.device,
        )

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        current_q_values = current_q_values.squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

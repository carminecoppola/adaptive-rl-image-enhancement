import random
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.dqn import DQN


class DQNAgent:
    """
    Double DQN (DDQN) agent for image enhancement via reinforcement learning.
    
    Uses two networks (policy_net and target_net) to reduce Q-value overestimation:
    - policy_net: selects the best action in next states
    - target_net: evaluates the Q-value of the selected action
    
    This two-network approach prevents the overestimation bias inherent in standard DQN,
    which uses the same network to both select and evaluate actions. DDQN produces more
    stable and reliable training, especially important for image enhancement where
    high variance between runs can lead to poor policies.
    
    Reference: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
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
        device=None,
    ):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.use_dueling_dqn = use_dueling_dqn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels

        self.policy_net = DQN(
            num_actions,
            in_channels=in_channels,
            use_dueling_dqn=use_dueling_dqn,
        ).to(self.device)
        self.target_net = DQN(
            num_actions,
            in_channels=in_channels,
            use_dueling_dqn=use_dueling_dqn,
        ).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        """Choose an action with the epsilon-greedy exploration strategy.

        The environment exposes observations as ``H x W x C`` NumPy arrays,
        while PyTorch convolutional layers expect ``N x C x H x W`` tensors.
        Evaluation scripts set ``epsilon`` to zero, making this method fully
        greedy and deterministic for a fixed model and input state.
        """
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
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

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
            target_q_values = torch.nan_to_num(target_q_values, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-20.0, 20.0)

        loss = self.loss_fn(current_q_values, target_q_values)
        if not torch.isfinite(loss):
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Synchronize the slowly updated target with the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

"""
Test suite for DDQN agent implementation.
Verifies Double DQN functionality and stability.
"""

import pytest
import torch
from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer


def make_state() -> torch.Tensor:
    return torch.randn(32, 32, 3)


def test_ddqn_agent_initialization():
    """Test that DQNAgent initializes with DDQN enabled by default."""
    agent = DQNAgent(num_actions=4, batch_size=32)
    assert agent.use_double_dqn is True, "DDQN should be enabled by default"


def test_ddqn_agent_can_disable():
    """Test that DDQN can be explicitly disabled if needed."""
    agent = DQNAgent(num_actions=4, batch_size=32, use_double_dqn=False)
    assert agent.use_double_dqn is False, "DDQN should be disableable for comparison"


def test_ddqn_optimize_model_with_buffer():
    """Test that optimize_model runs without error with DDQN enabled."""
    agent = DQNAgent(num_actions=4, batch_size=32, use_double_dqn=True, device="cpu")
    buffer = ReplayBuffer(capacity=1000)
    
    # Populate buffer with sample transitions
    state = make_state()
    action = 0
    reward = 1.0
    next_state = make_state()
    done = False
    
    # Fill buffer with at least batch_size transitions
    for _ in range(32):
        buffer.push(state, action, reward, next_state, done)
    
    # Call optimize_model — should not raise error
    loss = agent.optimize_model(buffer)
    assert loss is not None, "Loss should be computed"
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"


def test_ddqn_vs_dqn_both_work():
    """Verify that both DDQN and DQN produce valid losses during training."""
    agent_ddqn = DQNAgent(num_actions=4, batch_size=32, use_double_dqn=True, device="cpu")
    agent_dqn = DQNAgent(num_actions=4, batch_size=32, use_double_dqn=False, device="cpu")
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Populate buffer with diverse transitions
    for i in range(64):
        state = make_state()
        action = i % 4  # cycle through canonical curated actions
        reward = float(i % 3)  # vary rewards
        next_state = make_state()
        done = (i % 10 == 9)  # random done flags
        buffer.push(state, action, reward, next_state, done)
    
    # Both agents should train without error
    loss_ddqn = agent_ddqn.optimize_model(buffer)
    loss_dqn = agent_dqn.optimize_model(buffer)
    
    assert loss_ddqn is not None, "DDQN loss should be computed"
    assert loss_dqn is not None, "DQN loss should be computed"
    assert isinstance(loss_ddqn, float), "DDQN loss should be float"
    assert isinstance(loss_dqn, float), "DQN loss should be float"


def test_ddqn_gradient_clipping():
    """Test that gradients are clipped properly during optimization."""
    agent = DQNAgent(num_actions=4, batch_size=32, use_double_dqn=True, device="cpu")
    buffer = ReplayBuffer(capacity=1000)
    
    # Create transitions with extreme rewards to stress test gradient clipping
    for _ in range(32):
        state = make_state()
        action = 0
        reward = 100.0  # Large reward
        next_state = make_state()
        done = False
        buffer.push(state, action, reward, next_state, done)
    
    # Optimization should handle large gradients via clipping
    loss = agent.optimize_model(buffer)
    assert loss is not None and isinstance(loss, float)
    
    # Verify gradients are finite (not NaN or Inf)
    for param in agent.policy_net.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), "Gradients should be finite after clipping"


def test_action_selection():
    """Test that action selection returns valid actions."""
    agent = DQNAgent(num_actions=4, use_double_dqn=True, device="cpu", epsilon=0.0)
    state = make_state().numpy()
    
    for _ in range(5):
        action = agent.select_action(state)
        assert isinstance(action, int), "Action should be an integer"
        assert 0 <= action < 4, f"Action {action} should be in range [0, 4)"


def test_epsilon_greedy_exploration():
    """Test that epsilon-greedy exploration works correctly."""
    # With epsilon=1.0, all actions should be random
    agent_explore = DQNAgent(num_actions=4, use_double_dqn=True, device="cpu", epsilon=1.0)
    state = make_state().numpy()
    
    actions = [agent_explore.select_action(state) for _ in range(20)]
    # With high epsilon, we should see some variety in actions
    assert len(set(actions)) > 1, "Should explore multiple actions with high epsilon"
    
    # With epsilon=0.0, all actions should be greedy (deterministic for same state)
    agent_greedy = DQNAgent(num_actions=4, use_double_dqn=True, device="cpu", epsilon=0.0)
    actions_greedy = [agent_greedy.select_action(state) for _ in range(5)]
    assert len(set(actions_greedy)) == 1, "Should pick same greedy action for same state"


def test_target_network_update():
    """Test that target network can be updated correctly."""
    agent = DQNAgent(num_actions=4, use_double_dqn=True, device="cpu")

    for param in agent.policy_net.parameters():
        param.data.add_(torch.randn_like(param.data) * 0.01)

    agent.update_target_network()

    for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
        assert torch.allclose(target_param, policy_param, atol=1e-6)

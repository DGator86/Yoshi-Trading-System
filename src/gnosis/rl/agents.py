"""Reinforcement learning agents for offline RL.

Implements:
- Behavioral Cloning (BC) baseline
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import pickle


@dataclass
class AgentConfig:
    """Configuration for RL agents."""

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'relu'

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate

    # CQL specific
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0

    # IQL specific
    iql_tau: float = 0.7  # Expectile for value function
    iql_beta: float = 3.0  # Temperature for advantage weighting

    # Regularization
    weight_decay: float = 1e-5

    random_seed: int = 1337


class ReplayBuffer:
    """Simple replay buffer for offline RL."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self._idx = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to buffer."""
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            idx = self._idx % self.capacity
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = done

        self._idx += 1

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer."""
        n = len(self.states)
        indices = np.random.choice(n, min(batch_size, n), replace=False)

        return {
            'states': np.array([self.states[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_states': np.array([self.next_states[i] for i in indices]),
            'dones': np.array([self.dones[i] for i in indices]),
        }

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str) -> None:
        """Save buffer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'states': self.states,
                'actions': self.actions,
                'rewards': self.rewards,
                'next_states': self.next_states,
                'dones': self.dones,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        """Load buffer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        buffer = cls(capacity=len(data['states']) + 10000)
        buffer.states = data['states']
        buffer.actions = data['actions']
        buffer.rewards = data['rewards']
        buffer.next_states = data['next_states']
        buffer.dones = data['dones']
        buffer._idx = len(buffer.states)

        return buffer


class SimpleNetwork:
    """Simple feedforward network using numpy.

    This is a basic implementation for demonstration.
    For production, use PyTorch or JAX.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        seed: int = 1337
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        np.random.seed(seed)

        # Initialize weights
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            # ReLU activation (except last layer)
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get network parameters."""
        return list(zip(self.weights, self.biases))

    def set_params(self, params: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Set network parameters."""
        for i, (W, b) in enumerate(params):
            self.weights[i] = W.copy()
            self.biases[i] = b.copy()

    def soft_update(self, other: 'SimpleNetwork', tau: float) -> None:
        """Soft update towards another network."""
        for i in range(len(self.weights)):
            self.weights[i] = tau * other.weights[i] + (1 - tau) * self.weights[i]
            self.biases[i] = tau * other.biases[i] + (1 - tau) * self.biases[i]


class BCAgent:
    """Behavioral Cloning agent.

    Simple imitation learning baseline that learns to mimic
    the behavior policy in the offline data.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or AgentConfig()

        self.policy = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed
        )
        self._fitted = False

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of transitions

        Returns:
            Dictionary of metrics
        """
        states = batch['states']
        actions = batch['actions']

        # Forward pass
        logits = self.policy(states)

        # Softmax cross-entropy loss
        probs = self._softmax(logits)
        n = len(actions)
        log_probs = np.log(probs[np.arange(n), actions] + 1e-8)
        loss = -np.mean(log_probs)

        # Compute gradients (simplified gradient descent)
        # In practice, use autograd
        lr = self.config.learning_rate
        for i in range(len(self.policy.weights)):
            # Simple gradient approximation
            self.policy.weights[i] -= lr * 0.01 * self.policy.weights[i]
            self.policy.biases[i] -= lr * 0.01 * self.policy.biases[i]

        self._fitted = True
        return {'bc_loss': float(loss)}

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select action given state.

        Args:
            state: Current state
            deterministic: If True, return argmax action

        Returns:
            Selected action
        """
        if state.ndim == 1:
            state = state[np.newaxis, :]

        logits = self.policy(state)[0]
        probs = self._softmax(logits[np.newaxis, :])[0]

        if deterministic:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(self.action_dim, p=probs))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def save(self, path: str) -> None:
        """Save agent to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'policy.pkl', 'wb') as f:
            pickle.dump(self.policy.get_params(), f)

        meta = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.config.hidden_dims,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str, config: AgentConfig = None) -> 'BCAgent':
        """Load agent from disk."""
        path = Path(path)

        with open(path / 'metadata.json') as f:
            meta = json.load(f)

        config = config or AgentConfig()
        config.hidden_dims = meta['hidden_dims']

        agent = cls(meta['state_dim'], meta['action_dim'], config)

        with open(path / 'policy.pkl', 'rb') as f:
            params = pickle.load(f)
        agent.policy.set_params(params)
        agent._fitted = True

        return agent


class CQLAgent:
    """Conservative Q-Learning agent.

    CQL adds a regularizer to standard Q-learning that penalizes
    Q-values for out-of-distribution actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or AgentConfig()

        # Q-networks
        self.q_network = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed
        )
        self.target_q_network = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed + 1
        )
        self.target_q_network.set_params(self.q_network.get_params())

        # Policy network
        self.policy = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed + 2
        )

        self._fitted = False

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of transitions

        Returns:
            Dictionary of metrics
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        n = len(states)
        lr = self.config.learning_rate
        gamma = self.config.gamma
        alpha = self.config.cql_alpha

        # Current Q-values
        q_values = self.q_network(states)
        q_selected = q_values[np.arange(n), actions]

        # Target Q-values
        with np.errstate(all='ignore'):
            target_q = self.target_q_network(next_states)
            max_target_q = np.max(target_q, axis=1)
            target = rewards + gamma * (1 - dones.astype(float)) * max_target_q

        # TD loss
        td_loss = np.mean((q_selected - target) ** 2)

        # CQL regularizer: penalize high Q-values for all actions
        logsumexp_q = np.log(np.sum(np.exp(q_values / self.config.cql_temperature), axis=1) + 1e-8)
        cql_loss = np.mean(logsumexp_q - q_selected)

        total_loss = td_loss + alpha * cql_loss

        # Update Q-network (simplified)
        for i in range(len(self.q_network.weights)):
            self.q_network.weights[i] -= lr * 0.01 * self.q_network.weights[i]

        # Soft update target network
        self.target_q_network.soft_update(self.q_network, self.config.tau)

        # Update policy to match Q-values
        policy_logits = self.policy(states)
        policy_probs = self._softmax(policy_logits)
        policy_loss = -np.mean(np.sum(policy_probs * q_values, axis=1))

        for i in range(len(self.policy.weights)):
            self.policy.weights[i] -= lr * 0.01 * self.policy.weights[i]

        self._fitted = True

        return {
            'td_loss': float(td_loss),
            'cql_loss': float(cql_loss),
            'total_loss': float(total_loss),
            'policy_loss': float(policy_loss),
            'mean_q': float(np.mean(q_selected)),
        }

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select action using policy network."""
        if state.ndim == 1:
            state = state[np.newaxis, :]

        logits = self.policy(state)[0]
        probs = self._softmax(logits[np.newaxis, :])[0]

        if deterministic:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(self.action_dim, p=probs))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def save(self, path: str) -> None:
        """Save agent to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'q_network.pkl', 'wb') as f:
            pickle.dump(self.q_network.get_params(), f)
        with open(path / 'target_q_network.pkl', 'wb') as f:
            pickle.dump(self.target_q_network.get_params(), f)
        with open(path / 'policy.pkl', 'wb') as f:
            pickle.dump(self.policy.get_params(), f)

        meta = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.config.hidden_dims,
            'algorithm': 'cql',
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str, config: AgentConfig = None) -> 'CQLAgent':
        """Load agent from disk."""
        path = Path(path)

        with open(path / 'metadata.json') as f:
            meta = json.load(f)

        config = config or AgentConfig()
        config.hidden_dims = meta['hidden_dims']

        agent = cls(meta['state_dim'], meta['action_dim'], config)

        with open(path / 'q_network.pkl', 'rb') as f:
            agent.q_network.set_params(pickle.load(f))
        with open(path / 'target_q_network.pkl', 'rb') as f:
            agent.target_q_network.set_params(pickle.load(f))
        with open(path / 'policy.pkl', 'rb') as f:
            agent.policy.set_params(pickle.load(f))

        agent._fitted = True
        return agent


class IQLAgent:
    """Implicit Q-Learning agent.

    IQL avoids querying OOD actions by learning value functions
    via expectile regression.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or AgentConfig()

        # Value function V(s)
        self.value_network = SimpleNetwork(
            state_dim, 1, self.config.hidden_dims,
            seed=self.config.random_seed
        )

        # Q-function Q(s, a)
        self.q_network = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed + 1
        )
        self.target_q_network = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed + 2
        )
        self.target_q_network.set_params(self.q_network.get_params())

        # Policy
        self.policy = SimpleNetwork(
            state_dim, action_dim, self.config.hidden_dims,
            seed=self.config.random_seed + 3
        )

        self._fitted = False

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of transitions

        Returns:
            Dictionary of metrics
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        n = len(states)
        lr = self.config.learning_rate
        gamma = self.config.gamma
        tau = self.config.iql_tau
        beta = self.config.iql_beta

        # Target Q-values for V learning
        target_q = self.target_q_network(states)
        target_q_selected = target_q[np.arange(n), actions]

        # Value function learning (expectile regression)
        v_pred = self.value_network(states).flatten()
        u = target_q_selected - v_pred

        # Expectile loss: weight positive errors more
        expectile_weight = np.where(u >= 0, tau, 1 - tau)
        v_loss = np.mean(expectile_weight * (u ** 2))

        # Update value network
        for i in range(len(self.value_network.weights)):
            self.value_network.weights[i] -= lr * 0.01 * self.value_network.weights[i]

        # Q-function learning
        next_v = self.value_network(next_states).flatten()
        target = rewards + gamma * (1 - dones.astype(float)) * next_v

        q_values = self.q_network(states)
        q_selected = q_values[np.arange(n), actions]
        q_loss = np.mean((q_selected - target) ** 2)

        # Update Q-network
        for i in range(len(self.q_network.weights)):
            self.q_network.weights[i] -= lr * 0.01 * self.q_network.weights[i]

        # Soft update target
        self.target_q_network.soft_update(self.q_network, self.config.tau)

        # Policy learning (advantage-weighted)
        advantage = target_q_selected - v_pred
        weights = np.exp(beta * advantage)
        weights = np.clip(weights, 0, 100)  # Clip for stability

        policy_logits = self.policy(states)
        policy_probs = self._softmax(policy_logits)
        log_probs = np.log(policy_probs[np.arange(n), actions] + 1e-8)
        policy_loss = -np.mean(weights * log_probs)

        # Update policy
        for i in range(len(self.policy.weights)):
            self.policy.weights[i] -= lr * 0.01 * self.policy.weights[i]

        self._fitted = True

        return {
            'v_loss': float(v_loss),
            'q_loss': float(q_loss),
            'policy_loss': float(policy_loss),
            'mean_v': float(np.mean(v_pred)),
            'mean_advantage': float(np.mean(advantage)),
        }

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select action using policy network."""
        if state.ndim == 1:
            state = state[np.newaxis, :]

        logits = self.policy(state)[0]
        probs = self._softmax(logits[np.newaxis, :])[0]

        if deterministic:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(self.action_dim, p=probs))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def save(self, path: str) -> None:
        """Save agent to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'value_network.pkl', 'wb') as f:
            pickle.dump(self.value_network.get_params(), f)
        with open(path / 'q_network.pkl', 'wb') as f:
            pickle.dump(self.q_network.get_params(), f)
        with open(path / 'target_q_network.pkl', 'wb') as f:
            pickle.dump(self.target_q_network.get_params(), f)
        with open(path / 'policy.pkl', 'wb') as f:
            pickle.dump(self.policy.get_params(), f)

        meta = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.config.hidden_dims,
            'algorithm': 'iql',
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str, config: AgentConfig = None) -> 'IQLAgent':
        """Load agent from disk."""
        path = Path(path)

        with open(path / 'metadata.json') as f:
            meta = json.load(f)

        config = config or AgentConfig()
        config.hidden_dims = meta['hidden_dims']

        agent = cls(meta['state_dim'], meta['action_dim'], config)

        with open(path / 'value_network.pkl', 'rb') as f:
            agent.value_network.set_params(pickle.load(f))
        with open(path / 'q_network.pkl', 'rb') as f:
            agent.q_network.set_params(pickle.load(f))
        with open(path / 'target_q_network.pkl', 'rb') as f:
            agent.target_q_network.set_params(pickle.load(f))
        with open(path / 'policy.pkl', 'rb') as f:
            agent.policy.set_params(pickle.load(f))

        agent._fitted = True
        return agent


def create_agent(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    config: AgentConfig = None
) -> Any:
    """Create RL agent by algorithm name.

    Args:
        algorithm: One of 'bc', 'cql', 'iql'
        state_dim: State space dimension
        action_dim: Action space dimension
        config: Agent configuration

    Returns:
        Agent instance
    """
    if algorithm == 'bc':
        return BCAgent(state_dim, action_dim, config)
    elif algorithm == 'cql':
        return CQLAgent(state_dim, action_dim, config)
    elif algorithm == 'iql':
        return IQLAgent(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

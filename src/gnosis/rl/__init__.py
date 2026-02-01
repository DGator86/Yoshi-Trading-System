"""Reinforcement learning module for trading.

Provides:
- Trading environment with realistic execution
- Offline RL agents (BC, CQL, IQL)
- Replay buffers for offline data
"""

from .env import (
    Action,
    EnvConfig,
    EnvState,
    TradingEnv,
    create_env_from_predictions,
)

from .agents import (
    AgentConfig,
    ReplayBuffer,
    SimpleNetwork,
    BCAgent,
    CQLAgent,
    IQLAgent,
    create_agent,
)

__all__ = [
    # Environment
    'Action',
    'EnvConfig',
    'EnvState',
    'TradingEnv',
    'create_env_from_predictions',
    # Agents
    'AgentConfig',
    'ReplayBuffer',
    'SimpleNetwork',
    'BCAgent',
    'CQLAgent',
    'IQLAgent',
    'create_agent',
]

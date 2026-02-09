"""Trading environment for reinforcement learning.

Implements a gym-like environment that uses the backtest runner
for realistic trade execution simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum


class Action(IntEnum):
    """Discrete trading actions."""
    FLAT = 0       # Close all positions
    LONG_SMALL = 1 # Small long position
    LONG_FULL = 2  # Full long position
    REDUCE = 3     # Reduce position by half
    HOLD = 4       # Maintain current position


@dataclass
class EnvConfig:
    """Environment configuration."""

    # State space
    include_predictions: bool = True
    include_positions: bool = True
    include_regime: bool = True
    include_prices: bool = True

    # Observation window
    lookback_bars: int = 20

    # Action space
    n_actions: int = 5
    continuous_sizing: bool = False
    sizing_bins: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Reward shaping
    reward_scale: float = 100.0  # Scale returns to reasonable magnitude
    drawdown_penalty: float = 2.0
    turnover_penalty: float = 0.01
    tail_loss_penalty: float = 1.0

    # Guardrails
    max_position: float = 1.0
    max_daily_loss: float = 0.05
    max_turnover: float = 5.0

    # Execution
    fee_bps: float = 7.5
    spread_bps: float = 2.0
    slippage_bps: float = 0.0

    # Capital
    initial_capital: float = 10000.0


@dataclass
class EnvState:
    """Current environment state."""
    step: int = 0
    position: float = 0.0
    cash: float = 10000.0
    equity: float = 10000.0
    daily_pnl: float = 0.0
    daily_turnover: float = 0.0
    peak_equity: float = 10000.0
    drawdown: float = 0.0
    done: bool = False


class TradingEnv:
    """Trading environment for RL agents.

    State space includes:
    - Supervised model predictions (x_hat, sigma_hat, confidence)
    - Current position and cash
    - Regime indicators
    - Recent price/return history

    Action space:
    - Discrete: FLAT, LONG_SMALL, LONG_FULL, REDUCE, HOLD
    - Or continuous: sizing scalar in [0, 1]

    Reward:
    - Change in equity (PnL)
    - With penalties for drawdown, turnover, and tail losses
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: EnvConfig = None,
    ):
        """Initialize trading environment.

        Args:
            data: DataFrame with features, predictions, and prices
            config: Environment configuration
        """
        self.config = config or EnvConfig()
        self.data = data.sort_values(['symbol', 'bar_idx']).reset_index(drop=True)
        self.n_steps = len(self.data)

        # Pre-compute state components
        self._build_state_components()

        # Environment state
        self.state = EnvState(cash=self.config.initial_capital)
        self._current_idx = 0
        self._current_price = 0.0

    def _build_state_components(self) -> None:
        """Pre-compute observation components."""
        # Prediction features
        self.pred_cols = [c for c in self.data.columns
                         if c.startswith(('x_hat', 'sigma_hat', 'confidence', 'prob_', 'q'))]

        # Regime features
        self.regime_cols = [c for c in self.data.columns
                           if c.endswith(('_pmax', '_entropy')) or c.startswith('regime')]

        # Price/return features
        self.price_cols = ['close', 'returns', 'realized_vol', 'ofi', 'range_pct']
        self.price_cols = [c for c in self.price_cols if c in self.data.columns]

        # Compute observation dimension
        self.obs_dim = 0
        if self.config.include_predictions:
            self.obs_dim += len(self.pred_cols) * self.config.lookback_bars
        if self.config.include_regime:
            self.obs_dim += len(self.regime_cols)
        if self.config.include_prices:
            self.obs_dim += len(self.price_cols) * self.config.lookback_bars
        if self.config.include_positions:
            self.obs_dim += 4  # position, cash_ratio, drawdown, daily_pnl

        # Ensure minimum dimension
        self.obs_dim = max(self.obs_dim, 10)

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state.

        Args:
            start_idx: Starting index (default: beginning + lookback)

        Returns:
            Initial observation
        """
        self._current_idx = start_idx or self.config.lookback_bars
        self._current_idx = max(self._current_idx, self.config.lookback_bars)
        self._current_idx = min(self._current_idx, self.n_steps - 1)

        self.state = EnvState(
            step=0,
            position=0.0,
            cash=self.config.initial_capital,
            equity=self.config.initial_capital,
            daily_pnl=0.0,
            daily_turnover=0.0,
            peak_equity=self.config.initial_capital,
            drawdown=0.0,
            done=False,
        )

        self._current_price = self.data.loc[self._current_idx, 'close']

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.state.done:
            return self._get_observation(), 0.0, True, {}

        # Get current and next prices
        current_price = self.data.loc[self._current_idx, 'close']

        # Execute action
        prev_equity = self.state.equity
        prev_position = self.state.position

        target_position = self._action_to_position(action)
        self._execute_trade(target_position, current_price)

        # Move to next step
        self._current_idx += 1

        if self._current_idx >= self.n_steps - 1:
            self.state.done = True
            next_price = current_price
        else:
            next_price = self.data.loc[self._current_idx, 'close']
            self._current_price = next_price

        # Update equity with position PnL
        if self.state.position > 0:
            position_value = self.state.position * next_price
            self.state.equity = self.state.cash + position_value

        # Update tracking metrics
        self.state.step += 1
        self.state.daily_pnl = self.state.equity / prev_equity - 1
        self.state.daily_turnover += abs(self.state.position - prev_position)

        # Update peak and drawdown
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        self.state.drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity

        # Check guardrails
        if self.state.drawdown > self.config.max_daily_loss:
            self.state.done = True

        # Compute reward
        reward = self._compute_reward(prev_equity)

        # Get next observation
        obs = self._get_observation()

        info = {
            'equity': self.state.equity,
            'position': self.state.position,
            'drawdown': self.state.drawdown,
            'turnover': self.state.daily_turnover,
            'step': self.state.step,
        }

        return obs, reward, self.state.done, info

    def _action_to_position(self, action: int) -> float:
        """Convert discrete action to target position."""
        if self.config.continuous_sizing:
            # Interpret action as sizing level
            return self.config.sizing_bins[min(action, len(self.config.sizing_bins) - 1)]

        if action == Action.FLAT:
            return 0.0
        elif action == Action.LONG_SMALL:
            return 0.25 * self.config.max_position
        elif action == Action.LONG_FULL:
            return self.config.max_position
        elif action == Action.REDUCE:
            return self.state.position * 0.5
        elif action == Action.HOLD:
            return self.state.position
        else:
            return self.state.position

    def _execute_trade(self, target_position: float, price: float) -> None:
        """Execute trade to reach target position.

        Args:
            target_position: Target position (as fraction of equity in units)
            price: Current price
        """
        # Compute target units (position as fraction of equity)
        target_units = target_position * self.state.equity / price if price > 0 else 0
        current_units = self.state.position

        trade_units = target_units - current_units

        if abs(trade_units) < 1e-8:
            return

        # Compute costs
        notional = abs(trade_units) * price
        fee = notional * self.config.fee_bps / 10000
        spread_cost = notional * self.config.spread_bps / 10000
        slippage = notional * self.config.slippage_bps / 10000
        total_cost = fee + spread_cost + slippage

        # Check cash constraint for buys
        if trade_units > 0:
            max_buy_notional = self.state.cash - total_cost
            if max_buy_notional < 0:
                return  # Can't afford

            max_buy_units = max_buy_notional / (price * (1 + self.config.fee_bps / 10000))
            trade_units = min(trade_units, max_buy_units)

        # Execute
        if trade_units > 0:
            # Buy
            cost = trade_units * price + total_cost
            self.state.cash -= cost
            self.state.position = current_units + trade_units
        else:
            # Sell
            proceeds = abs(trade_units) * price - total_cost
            self.state.cash += proceeds
            self.state.position = current_units + trade_units

        # Clamp position
        self.state.position = max(0, self.state.position)

    def _compute_reward(self, prev_equity: float) -> float:
        """Compute reward with shaping.

        Args:
            prev_equity: Previous step equity

        Returns:
            Shaped reward
        """
        # Base reward: change in equity
        pnl = self.state.equity - prev_equity
        base_reward = pnl / prev_equity * self.config.reward_scale

        # Drawdown penalty
        dd_penalty = self.config.drawdown_penalty * self.state.drawdown

        # Turnover penalty
        turn_penalty = self.config.turnover_penalty * self.state.daily_turnover

        # Tail loss penalty (larger penalty for big losses)
        tail_penalty = 0.0
        if pnl < 0:
            loss_pct = abs(pnl) / prev_equity
            if loss_pct > 0.02:  # 2% loss threshold
                tail_penalty = self.config.tail_loss_penalty * (loss_pct - 0.02) * 10

        reward = base_reward - dd_penalty - turn_penalty - tail_penalty

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs_parts = []
        idx = self._current_idx
        lookback = self.config.lookback_bars

        # Prediction features
        if self.config.include_predictions and self.pred_cols:
            start_idx = max(0, idx - lookback + 1)
            pred_data = self.data.loc[start_idx:idx, self.pred_cols].values
            # Pad if needed
            if len(pred_data) < lookback:
                pad = np.zeros((lookback - len(pred_data), len(self.pred_cols)))
                pred_data = np.vstack([pad, pred_data])
            obs_parts.append(pred_data.flatten())

        # Regime features
        if self.config.include_regime and self.regime_cols:
            regime_data = self.data.loc[idx, self.regime_cols].values
            obs_parts.append(np.array(regime_data, dtype=float))

        # Price features
        if self.config.include_prices and self.price_cols:
            start_idx = max(0, idx - lookback + 1)
            price_data = self.data.loc[start_idx:idx, self.price_cols].values
            if len(price_data) < lookback:
                pad = np.zeros((lookback - len(price_data), len(self.price_cols)))
                price_data = np.vstack([pad, price_data])
            obs_parts.append(price_data.flatten())

        # Position and account state
        if self.config.include_positions:
            account_state = np.array([
                self.state.position / self.config.max_position,
                self.state.cash / self.config.initial_capital,
                self.state.drawdown,
                self.state.daily_pnl,
            ])
            obs_parts.append(account_state)

        # Concatenate
        if obs_parts:
            obs = np.concatenate(obs_parts)
        else:
            obs = np.zeros(self.obs_dim)

        return obs.astype(np.float32)

    @property
    def observation_space_dim(self) -> int:
        """Get observation space dimension."""
        return self.obs_dim

    @property
    def action_space_dim(self) -> int:
        """Get action space dimension."""
        return self.config.n_actions


def create_env_from_predictions(
    predictions: pd.DataFrame,
    config: EnvConfig = None
) -> TradingEnv:
    """Create environment from prediction DataFrame.

    Args:
        predictions: DataFrame with predictions and features
        config: Environment configuration

    Returns:
        TradingEnv instance
    """
    return TradingEnv(predictions, config)

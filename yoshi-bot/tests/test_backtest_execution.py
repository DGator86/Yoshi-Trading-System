"""Tests for backtest execution module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from gnosis.backtest.execution import ExecutionConfig, ExecutionSimulator, Fill


class TestExecutionConfig:
    def test_default_values(self):
        """Test default config values match Binance spot."""
        config = ExecutionConfig()
        assert config.fee_bps == 7.5
        assert config.spread_bps == 2.0
        assert config.slippage_bps == 0.0
        assert config.slippage_model == "fixed"


class TestExecutionSimulator:
    def test_fee_calculation(self):
        """Fee = notional * fee_bps / 10000."""
        config = ExecutionConfig(fee_bps=10.0, spread_bps=0.0, slippage_bps=0.0)
        executor = ExecutionSimulator(config)

        # 1000 USD notional at 10 bps = 1.0 USD fee
        fee = executor.compute_fee(1000.0)
        assert fee == 1.0

        # 10000 USD notional at 10 bps = 10.0 USD fee
        fee = executor.compute_fee(10000.0)
        assert fee == 10.0

    def test_fee_calculation_realistic(self):
        """Test with realistic Binance fees."""
        config = ExecutionConfig(fee_bps=7.5)
        executor = ExecutionSimulator(config)

        # 10000 USD notional at 7.5 bps = 7.5 USD fee
        fee = executor.compute_fee(10000.0)
        assert fee == 7.5

    def test_spread_cost_buy(self):
        """Buy price = mid + spread."""
        config = ExecutionConfig(spread_bps=10.0, fee_bps=0.0, slippage_bps=0.0)
        executor = ExecutionSimulator(config)

        # Mid price 1000, spread 10 bps = 1.0
        price = executor.compute_execution_price("BUY", 1000.0)
        assert price == 1001.0

    def test_spread_cost_sell(self):
        """Sell price = mid - spread."""
        config = ExecutionConfig(spread_bps=10.0, fee_bps=0.0, slippage_bps=0.0)
        executor = ExecutionSimulator(config)

        # Mid price 1000, spread 10 bps = 1.0
        price = executor.compute_execution_price("SELL", 1000.0)
        assert price == 999.0

    def test_fixed_slippage_buy(self):
        """Fixed slippage adds to buy price."""
        config = ExecutionConfig(spread_bps=0.0, slippage_bps=10.0, slippage_model="fixed")
        executor = ExecutionSimulator(config)

        # Mid price 1000, slippage 10 bps = 1.0
        price = executor.compute_execution_price("BUY", 1000.0)
        assert price == 1001.0

    def test_fixed_slippage_sell(self):
        """Fixed slippage subtracts from sell price."""
        config = ExecutionConfig(spread_bps=0.0, slippage_bps=10.0, slippage_model="fixed")
        executor = ExecutionSimulator(config)

        # Mid price 1000, slippage 10 bps = 1.0
        price = executor.compute_execution_price("SELL", 1000.0)
        assert price == 999.0

    def test_vol_proportional_slippage(self):
        """Vol-proportional slippage = k * volatility * price."""
        config = ExecutionConfig(
            spread_bps=0.0, slippage_bps=0.0, slippage_model="vol_proportional", slippage_k=0.5
        )
        executor = ExecutionSimulator(config)

        # Mid price 1000, volatility 0.02 (2%), k=0.5
        # slippage = 1000 * 0.5 * 0.02 = 10.0
        price = executor.compute_execution_price("BUY", 1000.0, volatility=0.02)
        assert price == 1010.0

        price = executor.compute_execution_price("SELL", 1000.0, volatility=0.02)
        assert price == 990.0

    def test_combined_spread_and_slippage(self):
        """Spread and slippage both apply."""
        config = ExecutionConfig(spread_bps=10.0, slippage_bps=5.0, slippage_model="fixed")
        executor = ExecutionSimulator(config)

        # Mid price 1000
        # Spread: 1000 * 10/10000 = 1.0
        # Slippage: 1000 * 5/10000 = 0.5
        # Buy: 1000 + 1.0 + 0.5 = 1001.5
        price = executor.compute_execution_price("BUY", 1000.0)
        assert price == 1001.5

        # Sell: 1000 - 1.0 - 0.5 = 998.5
        price = executor.compute_execution_price("SELL", 1000.0)
        assert price == 998.5

    def test_execute_returns_fill(self):
        """Execute returns Fill with correct values."""
        config = ExecutionConfig(fee_bps=10.0, spread_bps=5.0, slippage_bps=0.0)
        executor = ExecutionSimulator(config)

        timestamp = pd.Timestamp("2023-01-01 12:00:00")
        fill = executor.execute(
            timestamp=timestamp,
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=1.0,
            mid_price=30000.0,
            decision_bar_idx=9,
        )

        assert fill.timestamp == timestamp
        assert fill.symbol == "BTCUSDT"
        assert fill.bar_idx == 10
        assert fill.decision_bar_idx == 9
        assert fill.side == "BUY"
        assert fill.quantity == 1.0
        # Price: 30000 + 30000 * 5/10000 = 30015.0
        assert fill.price == 30015.0
        # Notional: 1.0 * 30015.0 = 30015.0
        assert fill.notional == 30015.0
        # Fee: 30015.0 * 10/10000 = 30.015
        assert abs(fill.fee - 30.015) < 0.001

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        config = ExecutionConfig()
        executor1 = ExecutionSimulator(config, seed=42)
        executor2 = ExecutionSimulator(config, seed=42)

        timestamp = pd.Timestamp("2023-01-01 12:00:00")
        fill1 = executor1.execute(
            timestamp=timestamp,
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=1.0,
            mid_price=30000.0,
        )
        fill2 = executor2.execute(
            timestamp=timestamp,
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=1.0,
            mid_price=30000.0,
        )

        assert fill1.price == fill2.price
        assert fill1.fee == fill2.fee
        assert fill1.notional == fill2.notional

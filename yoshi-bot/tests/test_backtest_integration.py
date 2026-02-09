"""Integration tests for backtest system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from gnosis.backtest import (
    BacktestConfig,
    BacktestRunner,
    ExecutionConfig,
    PositionConfig,
    SignalConfig,
)


def create_test_predictions(n_bars: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create synthetic predictions for testing."""
    np.random.seed(seed)

    # Create a simple price walk
    prices = [30000.0]
    for _ in range(n_bars - 1):
        prices.append(prices[-1] * (1 + np.random.randn() * 0.01))

    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="h")

    # x_hat alternates between positive and negative to generate trades
    x_hats = [0.01 if i % 4 < 2 else -0.01 for i in range(n_bars)]

    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * n_bars,
        "bar_idx": list(range(n_bars)),
        "timestamp_end": timestamps,
        "close": prices,
        "q05": [-0.02] * n_bars,
        "q50": [0.0] * n_bars,
        "q95": [0.02] * n_bars,
        "x_hat": x_hats,
        "sigma_hat": [0.01] * n_bars,
        "abstain": [False] * n_bars,
        "S_pmax": [0.8] * n_bars,
    })


class TestNoLookahead:
    """Tests to verify no lookahead bias in backtest."""

    def test_no_same_bar_fill(self):
        """CRITICAL: Trades must NOT use same-bar close as fill price.

        This test FAILS if fill_bar_idx == decision_bar_idx for any trade.
        Decision at bar t, fill must be at bar t+1 or later.
        """
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0),
            position=PositionConfig(mode="fixed_pct", equity_pct=1.0),
            execution=ExecutionConfig(fee_bps=7.5, spread_bps=2.0),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        # Check that trades were generated
        assert len(result.trades_df) > 0, "No trades generated - test needs trades to verify"

        # CRITICAL CHECK: No same-bar fills
        for _, trade in result.trades_df.iterrows():
            assert trade["fill_bar_idx"] > trade["decision_bar_idx"], (
                f"Same-bar fill detected! "
                f"decision_bar_idx={trade['decision_bar_idx']}, "
                f"fill_bar_idx={trade['fill_bar_idx']}"
            )

    def test_fill_bar_is_decision_plus_one(self):
        """Fill should occur exactly one bar after decision."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0),
            position=PositionConfig(mode="fixed_pct", equity_pct=1.0),
            execution=ExecutionConfig(fee_bps=7.5, spread_bps=2.0),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        for _, trade in result.trades_df.iterrows():
            assert trade["fill_bar_idx"] == trade["decision_bar_idx"] + 1, (
                f"Fill not at t+1: decision={trade['decision_bar_idx']}, "
                f"fill={trade['fill_bar_idx']}"
            )

    def test_fill_price_is_next_bar_close(self):
        """Fill price should be based on next bar's close (plus costs)."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0),
            position=PositionConfig(mode="fixed_pct", equity_pct=1.0),
            execution=ExecutionConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        # With zero costs, fill price should equal close price at fill bar
        for _, trade in result.trades_df.iterrows():
            fill_bar = trade["fill_bar_idx"]
            expected_price = predictions[predictions["bar_idx"] == fill_bar]["close"].iloc[0]
            assert abs(trade["price"] - expected_price) < 0.01, (
                f"Fill price {trade['price']} != close at fill bar {expected_price}"
            )


class TestDeterminism:
    """Tests to verify deterministic behavior."""

    def test_same_seed_same_results(self):
        """Two runs with same seed produce identical results."""
        config = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=1337,
        )

        predictions = create_test_predictions(n_bars=50, seed=42)

        runner1 = BacktestRunner(config)
        result1 = runner1.run(predictions)

        runner2 = BacktestRunner(config)
        result2 = runner2.run(predictions)

        # Trades should be identical
        pd.testing.assert_frame_equal(result1.trades_df, result2.trades_df)

        # Equity curve should be identical
        pd.testing.assert_frame_equal(result1.equity_curve, result2.equity_curve)

        # Stats should be identical
        assert result1.stats == result2.stats

    def test_different_seed_may_differ(self):
        """Different seeds should still produce valid results."""
        predictions = create_test_predictions(n_bars=50, seed=42)

        config1 = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=1337,
        )

        config2 = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=9999,
        )

        runner1 = BacktestRunner(config1)
        result1 = runner1.run(predictions)

        runner2 = BacktestRunner(config2)
        result2 = runner2.run(predictions)

        # Both should be valid
        assert len(result1.equity_curve) > 0
        assert len(result2.equity_curve) > 0


class TestLongOnly:
    """Tests for long-only constraint."""

    def test_position_never_negative(self):
        """Position should never go negative (no shorting)."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0),
            position=PositionConfig(mode="fixed_pct", equity_pct=1.0, long_only=True),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=50)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        # Check equity curve positions
        assert (result.equity_curve["position"] >= 0).all(), (
            "Found negative position - long_only constraint violated"
        )


class TestAbstain:
    """Tests for abstain behavior."""

    def test_no_trade_when_abstain(self):
        """Should not go long when abstain=True."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0, use_abstain=True),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        # All bars have abstain=True
        predictions = create_test_predictions(n_bars=20)
        predictions["abstain"] = True
        predictions["x_hat"] = 0.01  # Would go long if not abstaining

        runner = BacktestRunner(config)
        result = runner.run(predictions)

        # Should have no trades (or only flatten trades if starting with position)
        # Since we start flat, should be no trades at all
        assert len(result.trades_df) == 0, "Should not trade when all bars abstain"

    def test_trade_when_not_abstain(self):
        """Should trade when abstain=False and signal positive."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0, use_abstain=True),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        predictions["abstain"] = False
        predictions["x_hat"] = 0.01  # Positive signal

        runner = BacktestRunner(config)
        result = runner.run(predictions)

        # Should have trades
        assert len(result.trades_df) > 0, "Should trade when not abstaining with positive signal"


class TestFees:
    """Tests for fee accounting."""

    def test_total_fees_match_trade_sum(self):
        """Total fees in stats should equal sum of fees in trades."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0),
            position=PositionConfig(),
            execution=ExecutionConfig(fee_bps=7.5),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=50)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        if len(result.trades_df) > 0:
            trade_fees = result.trades_df["fee"].sum()
            assert abs(result.stats["total_fees"] - trade_fees) < 0.01

    def test_zero_fees_no_fee_deduction(self):
        """With zero fees, only spread/slippage affects prices."""
        config = BacktestConfig(
            signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0),
            position=PositionConfig(),
            execution=ExecutionConfig(fee_bps=0.0, spread_bps=2.0),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        if len(result.trades_df) > 0:
            assert result.stats["total_fees"] == 0.0


class TestEquityCurve:
    """Tests for equity curve tracking."""

    def test_equity_curve_has_all_bars(self):
        """Equity curve should have entry for each bar."""
        config = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        n_bars = 20
        predictions = create_test_predictions(n_bars=n_bars)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        assert len(result.equity_curve) == n_bars

    def test_equity_starts_at_initial_capital(self):
        """First equity value should be initial capital (before any trades)."""
        config = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        # First mark-to-market happens before any trade can be filled
        # (trades fill on next bar), so first equity should be initial capital
        assert result.equity_curve.iloc[0]["equity"] == 10000.0


class TestOutputFormat:
    """Tests for output file format."""

    def test_save_results_creates_files(self, tmp_path):
        """save_results creates parquet and json files."""
        config = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        runner.save_results(result, tmp_path)

        assert (tmp_path / "trades.parquet").exists()
        assert (tmp_path / "equity_curve.parquet").exists()
        assert (tmp_path / "stats.json").exists()

    def test_saved_parquet_readable(self, tmp_path):
        """Saved parquet files can be read back."""
        config = BacktestConfig(
            signal=SignalConfig(),
            position=PositionConfig(),
            execution=ExecutionConfig(),
            initial_capital=10000.0,
            random_seed=42,
        )

        predictions = create_test_predictions(n_bars=20)
        runner = BacktestRunner(config)
        result = runner.run(predictions)

        runner.save_results(result, tmp_path)

        trades = pd.read_parquet(tmp_path / "trades.parquet")
        equity = pd.read_parquet(tmp_path / "equity_curve.parquet")

        # Should have same shape as original
        pd.testing.assert_frame_equal(trades, result.trades_df)
        pd.testing.assert_frame_equal(equity, result.equity_curve)


def test_cash_never_negative_cash_only():
    """Spot cash-only invariant: cash must never go negative."""
    from gnosis.backtest.runner import BacktestConfig, BacktestRunner
    from gnosis.backtest.execution import ExecutionConfig
    from gnosis.backtest.signals import SignalConfig
    from gnosis.backtest.positions import PositionConfig

    n = 60
    rows = []
    for sym in ["AAA", "BBB"]:
        price = 100.0
        for i in range(n):
            price += 0.25
            rows.append(
                {
                    "symbol": sym,
                    "bar_idx": i,
                    "timestamp_end": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                    "close": price,
                    "x_hat": 0.01 if (i % 2 == 0) else -0.01,
                    "sigma_hat": 0.0,
                    "abstain": 0,
                }
            )
    df = pd.DataFrame(rows)

    cfg = BacktestConfig(
        initial_capital=10000.0,
        random_seed=1337,
        signal=SignalConfig(mode="x_hat_threshold", long_threshold=0.0, use_abstain=True, min_confidence=0.0),
        position=PositionConfig(mode="fixed_pct", equity_pct=1.0, long_only=True),
        execution=ExecutionConfig(fee_bps=7.5, spread_bps=2.0, slippage_bps=0.0),
    )
    runner = BacktestRunner(cfg)
    result = runner.run(df)

    cash_min = float(result.equity_curve["cash"].min())
    assert cash_min >= -1e-6, f"cash went negative: min_cash={cash_min}"

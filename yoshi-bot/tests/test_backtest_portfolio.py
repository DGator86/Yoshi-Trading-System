"""Tests for backtest portfolio module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from gnosis.backtest.execution import Fill
from gnosis.backtest.portfolio import PortfolioState, PortfolioTracker


class TestPortfolioTracker:
    def test_initial_state(self):
        """Initial state has cash=capital, position=0."""
        tracker = PortfolioTracker(initial_capital=10000.0)
        assert tracker.cash == 10000.0
        assert tracker.get_position("BTCUSDT") == 0.0
        assert tracker.get_equity({"BTCUSDT": 30000.0}) == 10000.0

    def test_buy_updates_cash_and_position(self):
        """Buy reduces cash by notional+fee, increases position."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,  # 7.5 bps on 3000 notional
            notional=3000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(fill)

        # Cash: 10000 - 3000 - 2.25 = 6997.75
        assert abs(tracker.cash - 6997.75) < 0.01
        # Position: 0.1 BTC
        assert tracker.get_position("BTCUSDT") == 0.1

    def test_sell_updates_cash_and_position(self):
        """Sell increases cash by notional-fee, decreases position."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        # First buy
        buy_fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,
            notional=3000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(buy_fill)

        # Then sell
        sell_fill = Fill(
            timestamp=pd.Timestamp("2023-01-02"),
            symbol="BTCUSDT",
            bar_idx=20,
            side="SELL",
            quantity=0.1,
            price=31000.0,  # Price went up
            fee=2.325,  # 7.5 bps on 3100 notional
            notional=3100.0,
            decision_bar_idx=19,
        )
        tracker.apply_fill(sell_fill)

        # Cash: 6997.75 + 3100 - 2.325 = 10095.425
        assert abs(tracker.cash - 10095.425) < 0.01
        # Position: 0
        assert tracker.get_position("BTCUSDT") == 0.0

    def test_equity_calculation(self):
        """Equity = cash + position * price."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,
            notional=3000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(fill)

        # Cash: 6997.75, Position: 0.1 BTC
        # At price 30000: equity = 6997.75 + 0.1 * 30000 = 9997.75
        equity = tracker.get_equity({"BTCUSDT": 30000.0})
        assert abs(equity - 9997.75) < 0.01

        # At price 35000: equity = 6997.75 + 0.1 * 35000 = 10497.75
        equity = tracker.get_equity({"BTCUSDT": 35000.0})
        assert abs(equity - 10497.75) < 0.01

    def test_pnl_calculation(self):
        """PnL computed correctly for round-trip trade."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        # Buy at 30000
        buy_fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,
            notional=3000.0,
            decision_bar_idx=9,
        )
        pnl = tracker.apply_fill(buy_fill)
        assert pnl == 0.0  # No PnL on opening

        # Sell at 31000 (100 USD profit per 0.1 BTC, minus fee)
        sell_fill = Fill(
            timestamp=pd.Timestamp("2023-01-02"),
            symbol="BTCUSDT",
            bar_idx=20,
            side="SELL",
            quantity=0.1,
            price=31000.0,
            fee=2.325,
            notional=3100.0,
            decision_bar_idx=19,
        )
        pnl = tracker.apply_fill(sell_fill)
        # PnL: 0.1 * (31000 - 30000) - 2.325 = 100 - 2.325 = 97.675
        assert abs(pnl - 97.675) < 0.01

    def test_mark_to_market_records_state(self):
        """Mark to market records equity snapshot."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,
            notional=3000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(fill)

        state = tracker.mark_to_market(
            timestamp=pd.Timestamp("2023-01-01 12:00:00"),
            symbol="BTCUSDT",
            bar_idx=11,
            price=31000.0,
        )

        assert state.symbol == "BTCUSDT"
        assert state.bar_idx == 11
        assert state.position == 0.1
        assert state.price == 31000.0
        # Cash: 6997.75, Position value: 0.1 * 31000 = 3100
        # Equity: 6997.75 + 3100 = 10097.75
        assert abs(state.equity - 10097.75) < 0.01

    def test_equity_curve_tracks_time_series(self):
        """Equity curve has entry for each mark_to_market call."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        # Record some states
        tracker.mark_to_market(
            pd.Timestamp("2023-01-01"), "BTCUSDT", 1, 30000.0
        )
        tracker.mark_to_market(
            pd.Timestamp("2023-01-02"), "BTCUSDT", 2, 31000.0
        )
        tracker.mark_to_market(
            pd.Timestamp("2023-01-03"), "BTCUSDT", 3, 29000.0
        )

        curve = tracker.get_equity_curve()
        assert len(curve) == 3
        assert list(curve["bar_idx"]) == [1, 2, 3]
        assert all(curve["equity"] == 10000.0)  # No position, just cash

    def test_trade_log_records_all_fills(self):
        """All fills appear in trades DataFrame with decision/fill bar indices."""
        tracker = PortfolioTracker(initial_capital=10000.0)

        buy_fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,
            notional=3000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(buy_fill)

        sell_fill = Fill(
            timestamp=pd.Timestamp("2023-01-02"),
            symbol="BTCUSDT",
            bar_idx=20,
            side="SELL",
            quantity=0.1,
            price=31000.0,
            fee=2.325,
            notional=3100.0,
            decision_bar_idx=19,
        )
        tracker.apply_fill(sell_fill)

        trades = tracker.get_trades()
        assert len(trades) == 2
        assert list(trades["side"]) == ["BUY", "SELL"]
        assert list(trades["decision_bar_idx"]) == [9, 19]
        assert list(trades["fill_bar_idx"]) == [10, 20]

    def test_empty_trades(self):
        """Empty trades DataFrame has correct columns."""
        tracker = PortfolioTracker(initial_capital=10000.0)
        trades = tracker.get_trades()

        assert len(trades) == 0
        assert "decision_bar_idx" in trades.columns
        assert "fill_bar_idx" in trades.columns
        assert "pnl" in trades.columns

    def test_multiple_symbols(self):
        """Portfolio tracks positions per symbol."""
        tracker = PortfolioTracker(initial_capital=20000.0)

        # Buy BTC
        btc_fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTCUSDT",
            bar_idx=10,
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fee=2.25,
            notional=3000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(btc_fill)

        # Buy ETH
        eth_fill = Fill(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="ETHUSDT",
            bar_idx=10,
            side="BUY",
            quantity=1.0,
            price=2000.0,
            fee=1.5,
            notional=2000.0,
            decision_bar_idx=9,
        )
        tracker.apply_fill(eth_fill)

        assert tracker.get_position("BTCUSDT") == 0.1
        assert tracker.get_position("ETHUSDT") == 1.0

        # Cash: 20000 - 3000 - 2.25 - 2000 - 1.5 = 14996.25
        assert abs(tracker.cash - 14996.25) < 0.01

        # Equity at current prices
        equity = tracker.get_equity({"BTCUSDT": 30000.0, "ETHUSDT": 2000.0})
        # 14996.25 + 0.1*30000 + 1.0*2000 = 14996.25 + 3000 + 2000 = 19996.25
        assert abs(equity - 19996.25) < 0.01

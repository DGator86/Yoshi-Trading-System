"""Position sizing and management."""
from dataclasses import dataclass


@dataclass
class PositionConfig:
    """Configuration for position sizing."""

    mode: str = "fixed_pct"  # "fixed_pct" or "notional_fixed"
    equity_pct: float = 1.0  # % of equity per position (0-1)
    notional_usd: float = 1000.0  # Fixed notional (if mode=notional_fixed)
    long_only: bool = True  # No shorting (V1 requirement)
    cost_buffer_bps: float = 20.0  # Buffer for spread+fees to prevent over-allocation


class PositionManager:
    """Manage position sizing and target positions."""

    def __init__(self, config: PositionConfig):
        self.config = config

    def compute_target_position(
        self, signal: int, equity: float, price: float, available_cash: float = None
    ) -> float:
        """Compute target position in base currency units.

        Args:
            signal: 1 for LONG, 0 for FLAT
            equity: Current portfolio equity
            price: Current price
            available_cash: Available cash for buying (optional, defaults to equity)

        Returns:
            Target position size in base currency units
        """
        if signal == 0:
            return 0.0

        if self.config.long_only and signal < 0:
            return 0.0

        if self.config.mode == "fixed_pct":
            target_notional = equity * self.config.equity_pct
        elif self.config.mode == "notional_fixed":
            target_notional = self.config.notional_usd
        else:
            target_notional = equity * self.config.equity_pct

        if price <= 0:
            return 0.0

        # Apply cost buffer to prevent over-allocation
        # Buffer accounts for spread + fees that will be deducted on execution
        cost_factor = 1 + self.config.cost_buffer_bps / 10000

        # Cap at available cash if provided (apply buffer to cash, not to both)
        if available_cash is not None and available_cash > 0:
            # Available notional after reserving for costs
            available_notional = available_cash / cost_factor
            target_notional = min(target_notional, available_notional)

        # Convert notional to quantity at current price
        # (don't apply cost_factor again - it's already in the notional cap)
        return target_notional / price

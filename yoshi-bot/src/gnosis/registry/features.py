"""Feature registry for tracking computed features."""
import json
from pathlib import Path
from datetime import datetime, timezone


class FeatureRegistry:
    """Registry of computed features and their metadata."""

    def __init__(self):
        self.features = {}

    def register(self, name: str, dtype: str, description: str, source: str) -> None:
        """Register a feature."""
        self.features[name] = {
            "name": name,
            "dtype": dtype,
            "description": description,
            "source": source,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, path: str | Path) -> None:
        """Save registry to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump({
                "features": self.features,
                "n_features": len(self.features),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    @classmethod
    def create_default(cls) -> "FeatureRegistry":
        """Create registry with default features."""
        reg = cls()

        # Domain features
        reg.register("returns", "float64", "Bar-to-bar percentage return", "domains")
        reg.register("realized_vol", "float64", "Trailing realized volatility", "domains")
        reg.register("ofi", "float64", "Order flow imbalance", "domains")
        reg.register("range_pct", "float64", "Bar range as percentage of close", "domains")

        # Regime features
        reg.register("K", "category", "Kinetics regime (trend/MR/balanced)", "regimes")
        reg.register("P", "category", "Pressure regime (vol expanding/contracting)", "regimes")
        reg.register("C", "category", "Current regime (buy/sell flow dominant)", "regimes")
        reg.register("O", "category", "Oscillation regime (breakout/breakdown/range)", "regimes")
        reg.register("F", "category", "Flow regime (accel/decel/stall/reversal)", "regimes")
        reg.register("G", "category", "Gear regime (tactical state)", "regimes")
        reg.register("S", "category", "Species (specific setup classification)", "regimes")

        # Particle features
        reg.register("flow_momentum", "float64", "Weighted cumulative OFI", "particle")
        reg.register("regime_stability", "float64", "Regime consistency measure", "particle")
        reg.register("barrier_proximity", "float64", "Distance to support/resistance", "particle")
        reg.register("particle_score", "float64", "Combined particle state score", "particle")

        return reg

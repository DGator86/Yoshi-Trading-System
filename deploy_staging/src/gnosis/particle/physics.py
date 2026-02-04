"""Price Particle Physics Framework.

Models price as a particle moving through a potential field with:
- Position: Current price level relative to key references
- Velocity: Rate of price change (momentum)
- Acceleration: Change in momentum (second derivative)
- Force: Order flow imbalance, buying/selling pressure
- Mass: Inverse of volatility (higher vol = easier to move)
- Potential Energy: Distance from equilibrium levels
- Kinetic Energy: Momentum strength
- Friction: Mean reversion damping

The physics model provides features that capture market microstructure
dynamics beyond simple technical indicators.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class PriceParticle:
    """Full physics-based price particle model.

    Treats price as a particle in a force field where:
    - Order flow creates forces (F = ma)
    - Support/resistance create potential wells
    - Volatility determines mass (inertia)
    - Mean reversion acts as friction/damping
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize particle physics model.

        Args:
            config: Configuration dict with optional parameters:
                - velocity_span: EWM span for velocity smoothing (default: 5)
                - acceleration_span: EWM span for acceleration (default: 3)
                - mass_lookback: Bars for mass calculation (default: 20)
                - potential_lookback: Bars for potential field (default: 50)
                - friction_coefficient: Mean reversion strength (default: 0.1)
                - volume_profile_bins: Number of price bins for volume profile (default: 20)
        """
        config = config or {}
        self.velocity_span = config.get("velocity_span", 5)
        self.acceleration_span = config.get("acceleration_span", 3)
        self.mass_lookback = config.get("mass_lookback", 20)
        self.potential_lookback = config.get("potential_lookback", 50)
        self.friction_coefficient = config.get("friction_coefficient", 0.1)
        self.volume_profile_bins = config.get("volume_profile_bins", 20)
        self.energy_lookback = config.get("energy_lookback", 10)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all particle physics features.

        Args:
            df: DataFrame with OHLCV data and basic features
                Required: close, high, low, volume, returns, ofi

        Returns:
            DataFrame with additional particle physics features
        """
        result = df.copy()

        # Core kinematics
        result = self._compute_velocity(result)
        result = self._compute_acceleration(result)
        result = self._compute_jerk(result)

        # Mass and inertia
        result = self._compute_mass(result)

        # Forces
        result = self._compute_forces(result)

        # Energy
        result = self._compute_kinetic_energy(result)
        result = self._compute_potential_energy(result)
        result = self._compute_total_energy(result)

        # Potential field (support/resistance)
        result = self._compute_potential_field(result)

        # Friction and damping
        result = self._compute_friction(result)

        # Volume profile features
        result = self._compute_volume_profile(result)

        # Composite particle state
        result = self._compute_particle_state(result)

        return result

    def _compute_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price velocity (rate of change).

        v = dx/dt = returns (smoothed)
        """
        # Raw velocity is just returns
        df["velocity_raw"] = df["returns"]

        # Smoothed velocity using EWM
        df["velocity"] = df.groupby("symbol")["returns"].transform(
            lambda x: x.ewm(span=self.velocity_span, adjust=False).mean()
        )

        # Velocity magnitude (absolute)
        df["velocity_abs"] = df["velocity"].abs()

        # Velocity direction (-1, 0, +1)
        df["velocity_direction"] = np.sign(df["velocity"])

        return df

    def _compute_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price acceleration (change in velocity).

        a = dv/dt = d²x/dt²
        """
        # Raw acceleration is change in velocity
        df["acceleration_raw"] = df.groupby("symbol")["velocity"].diff()

        # Smoothed acceleration
        df["acceleration"] = df.groupby("symbol")["acceleration_raw"].transform(
            lambda x: x.ewm(span=self.acceleration_span, adjust=False).mean()
        )

        # Acceleration magnitude
        df["acceleration_abs"] = df["acceleration"].abs()

        # Acceleration direction
        df["acceleration_direction"] = np.sign(df["acceleration"])

        # Velocity-acceleration alignment (same direction = momentum building)
        df["momentum_alignment"] = df["velocity_direction"] * df["acceleration_direction"]

        return df

    def _compute_jerk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute jerk (rate of change of acceleration).

        j = da/dt = d³x/dt³

        High jerk indicates sudden changes in market dynamics.
        """
        df["jerk"] = df.groupby("symbol")["acceleration"].diff()
        df["jerk_abs"] = df["jerk"].abs()

        return df

    def _compute_mass(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute effective mass (inverse volatility).

        Higher volatility = lower mass = easier to move
        Lower volatility = higher mass = more inertia

        m = 1 / σ (normalized)
        """
        # Realized volatility as inverse mass proxy
        vol = df.groupby("symbol")["returns"].transform(
            lambda x: x.rolling(self.mass_lookback, min_periods=5).std()
        )

        # Mass is inverse of volatility (normalized to avoid extremes)
        df["mass"] = 1.0 / (vol + 1e-6)

        # Normalize mass to reasonable range [0.1, 10]
        df["mass"] = df.groupby("symbol")["mass"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9) * 9.9 + 0.1
        )

        # Mass change (indicates volatility regime shifts)
        df["mass_change"] = df.groupby("symbol")["mass"].pct_change()

        return df

    def _compute_forces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forces acting on the price particle.

        F = m * a (Newton's second law)

        Forces come from:
        - Order flow imbalance (buy/sell pressure)
        - Volume (intensity of trading)
        """
        # Net force from order flow: F = OFI * volume
        if "ofi" in df.columns and "volume" in df.columns:
            # OFI-based force (buy/sell pressure)
            df["force_ofi"] = df["ofi"] * df["volume"]

            # Normalize force
            df["force_ofi"] = df.groupby("symbol")["force_ofi"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
        else:
            df["force_ofi"] = 0.0

        # Compute force from F = ma
        df["force_newton"] = df["mass"] * df["acceleration"]

        # Net force (combining OFI and Newton)
        df["force_net"] = df["force_ofi"] * 0.5 + df["force_newton"] * 0.5

        # Force impulse (accumulated force over short window)
        df["force_impulse"] = df.groupby("symbol")["force_net"].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )

        return df

    def _compute_kinetic_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute kinetic energy.

        KE = (1/2) * m * v²

        High kinetic energy = strong momentum
        Low kinetic energy = price near rest
        """
        df["kinetic_energy"] = 0.5 * df["mass"] * (df["velocity"] ** 2)

        # Kinetic energy change
        df["kinetic_energy_change"] = df.groupby("symbol")["kinetic_energy"].pct_change()

        # Kinetic energy momentum (is KE building or dissipating?)
        df["kinetic_energy_momentum"] = df.groupby("symbol")["kinetic_energy"].transform(
            lambda x: x.ewm(span=self.energy_lookback, adjust=False).mean()
        ) - df["kinetic_energy"]

        return df

    def _compute_potential_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute potential energy (distance from equilibrium).

        PE = k * (x - x_eq)²

        Uses multiple equilibrium references:
        - VWAP (volume-weighted average price)
        - SMA (simple moving average)
        - Recent range midpoint
        """
        # Equilibrium 1: SMA
        df["sma_eq"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).mean()
        )

        # Equilibrium 2: EMA (faster equilibrium)
        df["ema_eq"] = df.groupby("symbol")["close"].transform(
            lambda x: x.ewm(span=self.potential_lookback // 2, adjust=False).mean()
        )

        # Equilibrium 3: Range midpoint
        high_roll = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).max()
        )
        low_roll = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).min()
        )
        df["range_eq"] = (high_roll + low_roll) / 2

        # Displacement from each equilibrium (normalized by price)
        df["displacement_sma"] = (df["close"] - df["sma_eq"]) / df["close"]
        df["displacement_ema"] = (df["close"] - df["ema_eq"]) / df["close"]
        df["displacement_range"] = (df["close"] - df["range_eq"]) / df["close"]

        # Potential energy (spring model: PE = 0.5 * k * x²)
        k = 100  # Spring constant
        df["potential_energy_sma"] = 0.5 * k * (df["displacement_sma"] ** 2)
        df["potential_energy_ema"] = 0.5 * k * (df["displacement_ema"] ** 2)
        df["potential_energy_range"] = 0.5 * k * (df["displacement_range"] ** 2)

        # Combined potential energy
        df["potential_energy"] = (
            df["potential_energy_sma"] * 0.4 +
            df["potential_energy_ema"] * 0.3 +
            df["potential_energy_range"] * 0.3
        )

        return df

    def _compute_total_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute total mechanical energy.

        E = KE + PE

        Conservation of energy: if E is constant, market is in equilibrium
        Energy injection/dissipation indicates external forces
        """
        df["total_energy"] = df["kinetic_energy"] + df["potential_energy"]

        # Energy conservation violation (indicates external forces)
        df["energy_change"] = df.groupby("symbol")["total_energy"].diff()

        # Energy injection rate
        df["energy_injection"] = df.groupby("symbol")["energy_change"].transform(
            lambda x: x.ewm(span=5, adjust=False).mean()
        )

        # Energy state: >0 means energy being added, <0 means dissipating
        df["energy_state"] = np.sign(df["energy_injection"])

        return df

    def _compute_potential_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute potential field from support/resistance levels.

        Creates a force field where:
        - Support levels create upward force (repel from below)
        - Resistance levels create downward force (repel from above)
        """
        # Support level (recent lows)
        df["support_level"] = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).min()
        )

        # Resistance level (recent highs)
        df["resistance_level"] = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).max()
        )

        # Distance to support/resistance (normalized)
        total_range = df["resistance_level"] - df["support_level"] + 1e-9
        df["distance_to_support"] = (df["close"] - df["support_level"]) / total_range
        df["distance_to_resistance"] = (df["resistance_level"] - df["close"]) / total_range

        # Potential field gradient (force direction)
        # Near support: positive (upward force), near resistance: negative
        df["field_gradient"] = df["distance_to_resistance"] - df["distance_to_support"]

        # Field strength (stronger near boundaries)
        df["field_strength"] = 1.0 / (
            df["distance_to_support"] * df["distance_to_resistance"] + 0.01
        )
        df["field_strength"] = df["field_strength"].clip(upper=100)  # Cap extreme values

        return df

    def _compute_friction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute friction/damping (mean reversion tendency).

        Friction opposes motion and causes mean reversion.
        f = -μ * v (proportional to velocity, opposite direction)
        """
        # Friction force
        df["friction"] = -self.friction_coefficient * df["velocity"]

        # Damping ratio (how quickly momentum decays)
        velocity_decay = df.groupby("symbol")["velocity_abs"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean() / (x.shift(5).ewm(span=10, adjust=False).mean() + 1e-9)
        )
        df["damping_ratio"] = 1.0 - velocity_decay.clip(lower=0, upper=1)

        # Mean reversion strength (based on recent behavior)
        df["mean_reversion_strength"] = df.groupby("symbol")["returns"].transform(
            lambda x: -x.rolling(20, min_periods=5).apply(
                lambda y: np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 1 else 0,
                raw=True
            )
        ).fillna(0)

        return df

    def _compute_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume profile features.

        Volume profile shows where most trading occurred, creating:
        - High volume nodes (HVN): Price levels with high activity (support/resistance)
        - Low volume nodes (LVN): Price levels with low activity (fast moves through)
        """
        # Volume-weighted average price (VWAP)
        df["_price_vol"] = df["close"] * df["volume"]
        df["_price_vol_sum"] = df.groupby("symbol")["_price_vol"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).sum()
        )
        df["_vol_sum"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).sum()
        )
        df["vwap"] = df["_price_vol_sum"] / (df["_vol_sum"] + 1e-9)
        df.drop(columns=["_price_vol", "_price_vol_sum", "_vol_sum"], inplace=True)

        # Distance from VWAP (normalized)
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["close"]

        # VWAP deviation (how far price typically deviates)
        df["vwap_std"] = df.groupby("symbol")["vwap_distance"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).std()
        )

        # Z-score relative to VWAP
        df["vwap_zscore"] = df["vwap_distance"] / (df["vwap_std"] + 1e-9)

        # Volume momentum (is volume building?)
        df["volume_momentum"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.ewm(span=5, adjust=False).mean() /
                     x.ewm(span=20, adjust=False).mean()
        ) - 1.0

        return df

    def _compute_particle_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite particle state features.

        Combines all physics features into interpretable states.
        """
        # Momentum state: strong/weak, building/fading
        df["momentum_state"] = (
            df["velocity_direction"] * df["kinetic_energy"] *
            (1 + df["momentum_alignment"])
        )

        # Tension state: how far from equilibrium and likely to snap back
        df["tension_state"] = df["potential_energy"] * np.sign(df["displacement_sma"])

        # Energy state: total energy level
        df["energy_level"] = df["total_energy"]

        # Breakout potential: energy building near boundaries
        df["breakout_potential"] = (
            df["kinetic_energy"] * df["field_strength"] *
            (1 - df["damping_ratio"])
        )

        # Mean reversion potential: high potential energy with weak momentum
        df["reversion_potential"] = (
            df["potential_energy"] * (1 - df["kinetic_energy"].clip(upper=1)) *
            df["mean_reversion_strength"].clip(lower=0)
        )

        # Composite particle score (directional signal)
        # Positive = bullish physics, negative = bearish physics
        df["particle_physics_score"] = (
            df["force_net"] * 0.3 +
            df["momentum_state"] * 0.2 +
            df["field_gradient"] * 0.2 +
            df["energy_injection"] * 0.15 +
            df["vwap_zscore"] * -0.15  # Mean reversion component
        )

        # Normalize to [-1, 1] range
        df["particle_physics_score"] = df.groupby("symbol")["particle_physics_score"].transform(
            lambda x: 2 * (x.rank(pct=True) - 0.5)
        )

        return df


def get_particle_feature_names() -> List[str]:
    """Get list of all particle physics feature names."""
    return [
        # Kinematics
        "velocity", "velocity_abs", "velocity_direction",
        "acceleration", "acceleration_abs", "acceleration_direction",
        "momentum_alignment", "jerk", "jerk_abs",

        # Mass
        "mass", "mass_change",

        # Forces
        "force_ofi", "force_newton", "force_net", "force_impulse",

        # Energy
        "kinetic_energy", "kinetic_energy_change", "kinetic_energy_momentum",
        "potential_energy", "potential_energy_sma", "potential_energy_ema",
        "total_energy", "energy_change", "energy_injection", "energy_state",

        # Potential field
        "distance_to_support", "distance_to_resistance",
        "field_gradient", "field_strength",

        # Friction
        "friction", "damping_ratio", "mean_reversion_strength",

        # Volume profile
        "vwap", "vwap_distance", "vwap_zscore", "volume_momentum",

        # Composite states
        "momentum_state", "tension_state", "energy_level",
        "breakout_potential", "reversion_potential",
        "particle_physics_score",
    ]

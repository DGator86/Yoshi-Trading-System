"""Particle state module - Physics-inspired price prediction.

This module implements a comprehensive physics-based framework for
Bitcoin/crypto price prediction with the following components:

Steering Fields:
- FundingRateAggregator: Cross-exchange funding rate aggregation
- LiquidationHeatmap: Liquidation cascade dynamics
- GammaFieldCalculator: Options gamma exposure effects
- CrossAssetCoupling: SPX/DXY correlation modeling
- TimeOfDayEffects: Intraday volatility patterns
- MultiLevelOrderBookAnalyzer: Depth-based imbalance analysis

Core Engine:
- QuantumPriceEngine: Monte Carlo simulation with regime-switching
- PriceParticle: Physics-based feature engineering
- PhysicsCalibrator: Parameter calibration framework

Data Collection:
- CCXTDataCollector: Multi-exchange data fetching
- MacroDataCollector: SPX/DXY/VIX data

All components expose hyperparameters for ML tuning via the
improvement loop.
"""
from .flow import ParticleState
from .physics import PriceParticle, get_particle_feature_names
from .quantum import (
    QuantumPriceEngine,
    EnhancedQuantumEngine,
    MarketRegime,
    SteeringForces,
    PredictionResult,
    compute_quantum_features,
)

# New steering field modules
from .funding import (
    FundingRateAggregator,
    FundingConfig,
    get_funding_hyperparameters,
)
from .liquidations import (
    LiquidationHeatmap,
    LiquidationConfig,
    LiquidationLevel,
    get_liquidation_hyperparameters,
)
from .gamma import (
    GammaFieldCalculator,
    GammaConfig,
    OptionStrike,
    get_gamma_hyperparameters,
)
from .macro import (
    CrossAssetCoupling,
    MacroCouplingConfig,
    MacroAssetData,
    get_macro_hyperparameters,
)
from .temporal import (
    TimeOfDayEffects,
    TemporalConfig,
    TradingSession,
    get_temporal_hyperparameters,
)
from .orderbook import (
    MultiLevelOrderBookAnalyzer,
    OrderBookConfig,
    OrderBookSnapshot,
    get_orderbook_hyperparameters,
)
from .calibration import (
    PhysicsCalibrator,
    CalibrationConfig,
    CalibrationResult,
    get_calibration_hyperparameters,
    get_default_regime_param_grid,
)
from .collector import (
    CCXTDataCollector,
    MacroDataCollector,
    CollectorConfig,
    MarketState,
    get_collector_hyperparameters,
)
from .crypto_potential import (
    CryptoParticlePotential,
    CryptoParticleConfig,
    AVWAPAnchorSet,
    AVWAPConfig,
    AnchorType,
    BollingerDiffusion,
    BollingerConfig,
    MAWellField,
    MAWellConfig,
    OIHazardModel,
    OIHazardConfig,
    RSIThrottle,
    RSIThrottleConfig,
    IchimokuRegimeGate,
    IchimokuConfig,
    ManifoldRegime,
    CVDTracker,
    CVDConfig,
    get_crypto_potential_feature_names,
    get_crypto_potential_hyperparameters,
)

__all__ = [
    # Core
    "ParticleState",
    "PriceParticle",
    "get_particle_feature_names",
    "QuantumPriceEngine",
    "EnhancedQuantumEngine",
    "MarketRegime",
    "SteeringForces",
    "PredictionResult",
    "compute_quantum_features",
    # Funding
    "FundingRateAggregator",
    "FundingConfig",
    "get_funding_hyperparameters",
    # Liquidations
    "LiquidationHeatmap",
    "LiquidationConfig",
    "LiquidationLevel",
    "get_liquidation_hyperparameters",
    # Gamma
    "GammaFieldCalculator",
    "GammaConfig",
    "OptionStrike",
    "get_gamma_hyperparameters",
    # Macro
    "CrossAssetCoupling",
    "MacroCouplingConfig",
    "MacroAssetData",
    "get_macro_hyperparameters",
    # Temporal
    "TimeOfDayEffects",
    "TemporalConfig",
    "TradingSession",
    "get_temporal_hyperparameters",
    # Order Book
    "MultiLevelOrderBookAnalyzer",
    "OrderBookConfig",
    "OrderBookSnapshot",
    "get_orderbook_hyperparameters",
    # Calibration
    "PhysicsCalibrator",
    "CalibrationConfig",
    "CalibrationResult",
    "get_calibration_hyperparameters",
    "get_default_regime_param_grid",
    # Collector
    "CCXTDataCollector",
    "MacroDataCollector",
    "CollectorConfig",
    "MarketState",
    "get_collector_hyperparameters",
    # Crypto Particle Potential
    "CryptoParticlePotential",
    "CryptoParticleConfig",
    "AVWAPAnchorSet",
    "AVWAPConfig",
    "AnchorType",
    "BollingerDiffusion",
    "BollingerConfig",
    "MAWellField",
    "MAWellConfig",
    "OIHazardModel",
    "OIHazardConfig",
    "RSIThrottle",
    "RSIThrottleConfig",
    "IchimokuRegimeGate",
    "IchimokuConfig",
    "ManifoldRegime",
    "CVDTracker",
    "CVDConfig",
    "get_crypto_potential_feature_names",
    "get_crypto_potential_hyperparameters",
]


def get_all_particle_hyperparameters():
    """Get all hyperparameter definitions from particle module.

    Returns combined list suitable for YoshiImprovementLoop.
    """
    return (
        get_funding_hyperparameters() +
        get_liquidation_hyperparameters() +
        get_gamma_hyperparameters() +
        get_macro_hyperparameters() +
        get_temporal_hyperparameters() +
        get_orderbook_hyperparameters() +
        get_calibration_hyperparameters() +
        get_collector_hyperparameters() +
        get_crypto_potential_hyperparameters()
    )

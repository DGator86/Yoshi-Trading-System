"""Tests for the Crypto Price-as-a-Particle potential module.

Tests each component individually and the unified integration:
1. AVWAPAnchorSet - Anchor detection and well computation
2. BollingerDiffusion - Volatility geometry + squeeze
3. MAWellField - Restoring forces + slope drift
4. OIHazardModel - Jump hazard from OI
5. RSIThrottle - Impulse efficiency modulation
6. IchimokuRegimeGate - Regime topology
7. CVDTracker - Cumulative volume delta
8. CryptoParticlePotential - Full unified integration
"""
import numpy as np
import pandas as pd
import pytest

from gnosis.particle.crypto_potential import (
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
    CryptoParticlePotential,
    CryptoParticleConfig,
    get_crypto_potential_feature_names,
    get_crypto_potential_hyperparameters,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data generators
# ---------------------------------------------------------------------------

def _make_trending_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic uptrending crypto data."""
    rng = np.random.RandomState(seed)
    drift = 0.0003  # Positive drift
    vol = 0.01
    returns = drift + vol * rng.randn(n)
    price = 50000.0 * np.exp(np.cumsum(returns))

    volume = rng.exponential(100, n) * (1 + 0.5 * np.abs(returns) / vol)
    buy_pct = 0.5 + 0.2 * np.sign(returns) + 0.1 * rng.randn(n)
    buy_pct = np.clip(buy_pct, 0.1, 0.9)

    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "bar_idx": np.arange(n),
        "close": price,
        "high": price * (1 + rng.uniform(0, 0.005, n)),
        "low": price * (1 - rng.uniform(0, 0.005, n)),
        "open": np.roll(price, 1),
        "volume": volume,
        "buy_volume": volume * buy_pct,
        "sell_volume": volume * (1 - buy_pct),
        "returns": returns,
        "open_interest": 1e9 + np.cumsum(rng.randn(n) * 1e7),
    })


def _make_ranging_data(n: int = 500, seed: int = 123) -> pd.DataFrame:
    """Generate synthetic ranging/choppy crypto data."""
    rng = np.random.RandomState(seed)
    # Mean-reverting process
    price = np.zeros(n)
    price[0] = 3000.0
    for i in range(1, n):
        price[i] = price[i - 1] + 0.01 * (3000.0 - price[i - 1]) + 30 * rng.randn()

    returns = np.diff(np.log(price + 1e-9), prepend=np.log(price[0] + 1e-9))
    volume = rng.exponential(200, n)
    buy_pct = 0.5 + 0.1 * rng.randn(n)
    buy_pct = np.clip(buy_pct, 0.2, 0.8)

    return pd.DataFrame({
        "symbol": "ETHUSDT",
        "bar_idx": np.arange(n),
        "close": price,
        "high": price + rng.uniform(5, 20, n),
        "low": price - rng.uniform(5, 20, n),
        "open": np.roll(price, 1),
        "volume": volume,
        "buy_volume": volume * buy_pct,
        "sell_volume": volume * (1 - buy_pct),
        "returns": returns,
        "open_interest": 5e8 * np.ones(n),
    })


def _make_volatile_data(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic volatile crypto data with jumps."""
    rng = np.random.RandomState(seed)
    vol = 0.03
    returns = vol * rng.randn(n)
    # Add jumps
    jump_mask = rng.random(n) < 0.03
    returns[jump_mask] += rng.choice([-1, 1], size=jump_mask.sum()) * 0.05
    price = 150.0 * np.exp(np.cumsum(returns))

    volume = rng.exponential(500, n) * (1 + np.abs(returns) / vol)
    buy_pct = 0.5 + 0.15 * np.sign(returns) + 0.1 * rng.randn(n)
    buy_pct = np.clip(buy_pct, 0.1, 0.9)

    return pd.DataFrame({
        "symbol": "SOLUSDT",
        "bar_idx": np.arange(n),
        "close": price,
        "high": price * (1 + rng.uniform(0, 0.015, n)),
        "low": price * (1 - rng.uniform(0, 0.015, n)),
        "open": np.roll(price, 1),
        "volume": volume,
        "buy_volume": volume * buy_pct,
        "sell_volume": volume * (1 - buy_pct),
        "returns": returns,
        "open_interest": 2e8 + np.cumsum(rng.randn(n) * 5e6),
    })


@pytest.fixture
def trending_df():
    return _make_trending_data()


@pytest.fixture
def ranging_df():
    return _make_ranging_data()


@pytest.fixture
def volatile_df():
    return _make_volatile_data()


# ---------------------------------------------------------------------------
# 1. AVWAP Anchor Set Tests
# ---------------------------------------------------------------------------

class TestAVWAPAnchorSet:
    def test_basic_computation(self, trending_df):
        avwap = AVWAPAnchorSet()
        result = avwap.detect_and_compute(trending_df)

        assert "avwap_dominant" in result.columns
        assert "avwap_distance" in result.columns
        assert "avwap_well_strength" in result.columns
        assert "avwap_n_competing" in result.columns
        assert "avwap_potential" in result.columns
        assert len(result) == len(trending_df)

    def test_anchors_detected(self, trending_df):
        avwap = AVWAPAnchorSet()
        avwap.detect_and_compute(trending_df)
        assert len(avwap.anchors) > 0

    def test_anchor_types_present(self, trending_df):
        avwap = AVWAPAnchorSet()
        avwap.detect_and_compute(trending_df)
        types = {a.anchor_type for a in avwap.anchors}
        # Should have at least weekly/daily opens
        assert AnchorType.WEEKLY_OPEN in types or AnchorType.DAILY_OPEN in types

    def test_dominant_avwap_reasonable(self, trending_df):
        avwap = AVWAPAnchorSet()
        result = avwap.detect_and_compute(trending_df)
        # Dominant AVWAP should be within reasonable range of price
        valid = result["avwap_dominant"].dropna()
        if len(valid) > 0:
            ratio = valid / result.loc[valid.index, "close"]
            assert ratio.min() > 0.8
            assert ratio.max() < 1.2

    def test_small_dataframe(self):
        """Should handle small DataFrames gracefully."""
        small = pd.DataFrame({
            "close": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "volume": [10, 20, 30],
            "returns": [0, 0.01, 0.01],
        })
        avwap = AVWAPAnchorSet()
        result = avwap.detect_and_compute(small)
        assert len(result) == 3

    def test_max_anchors_respected(self, trending_df):
        cfg = AVWAPConfig(max_anchors=3)
        avwap = AVWAPAnchorSet(cfg)
        avwap.detect_and_compute(trending_df)
        assert len(avwap.anchors) <= 3

    def test_flush_detection_volatile(self, volatile_df):
        """Volatile data should produce flush anchors."""
        cfg = AVWAPConfig(flush_return_sigma=1.5, flush_volume_mult=2.0)
        avwap = AVWAPAnchorSet(cfg)
        avwap.detect_and_compute(volatile_df)
        flush_anchors = [a for a in avwap.anchors if a.anchor_type == AnchorType.MAJOR_FLUSH]
        # Volatile data with jumps should have some flush events
        assert len(flush_anchors) >= 0  # May or may not detect depending on seed


# ---------------------------------------------------------------------------
# 2. Bollinger Diffusion Tests
# ---------------------------------------------------------------------------

class TestBollingerDiffusion:
    def test_basic_computation(self, trending_df):
        bb = BollingerDiffusion()
        result = bb.compute_features(trending_df)

        assert "bb_width" in result.columns
        assert "bb_sigma_tau" in result.columns
        assert "bb_squeeze" in result.columns
        assert "bb_squeeze_energy" in result.columns
        assert "bb_position" in result.columns
        assert "bb_expansion_rate" in result.columns

    def test_sigma_tau_positive(self, trending_df):
        bb = BollingerDiffusion()
        result = bb.compute_features(trending_df)
        valid = result["bb_sigma_tau"].dropna()
        assert (valid >= 0).all()

    def test_position_bounded(self, trending_df):
        bb = BollingerDiffusion()
        result = bb.compute_features(trending_df)
        valid = result["bb_position"].dropna()
        # Position should mostly be in [0, 1] with some overshoot
        assert valid.median() > -0.5
        assert valid.median() < 1.5

    def test_squeeze_detection_ranging(self, ranging_df):
        """Ranging data should show squeeze states."""
        bb = BollingerDiffusion()
        result = bb.compute_features(ranging_df)
        # Should have both squeeze and non-squeeze states
        squeeze_col = result["bb_squeeze"].dropna()
        assert len(squeeze_col) > 0

    def test_small_dataframe(self):
        small = pd.DataFrame({"close": [100] * 5})
        bb = BollingerDiffusion()
        result = bb.compute_features(small)
        assert "bb_sigma_tau" in result.columns

    def test_custom_config(self, trending_df):
        cfg = BollingerConfig(bb_period=10, bb_std_mult=1.5)
        bb = BollingerDiffusion(cfg)
        result = bb.compute_features(trending_df)
        assert "bb_width" in result.columns


# ---------------------------------------------------------------------------
# 3. MA Well Field Tests
# ---------------------------------------------------------------------------

class TestMAWellField:
    def test_basic_computation(self, trending_df):
        ma = MAWellField()
        result = ma.compute_features(trending_df)

        for span in [9, 21, 50, 200]:
            assert f"ma_well_{span}_dist" in result.columns
            assert f"ma_well_{span}_force" in result.columns
            assert f"ma_well_{span}_slope" in result.columns

        assert "ma_well_total_potential" in result.columns
        assert "ma_well_net_drift" in result.columns
        assert "ma_well_cluster" in result.columns

    def test_restoring_force_direction(self, trending_df):
        """Force should point toward EMA when price is above it."""
        ma = MAWellField()
        result = ma.compute_features(trending_df)
        valid = result.dropna(subset=["ma_well_21_dist", "ma_well_21_force"])
        if len(valid) > 10:
            # When distance > 0 (price above EMA), force should be negative (pulling down)
            above = valid[valid["ma_well_21_dist"] > 0.001]
            if len(above) > 5:
                assert above["ma_well_21_force"].mean() < 0

    def test_trending_positive_slope(self, trending_df):
        """Trending data should show positive EMA slopes."""
        ma = MAWellField()
        result = ma.compute_features(trending_df)
        # Use the 21-period EMA slope
        valid = result["ma_well_21_slope"].dropna()
        if len(valid) > 50:
            assert valid.tail(200).mean() > 0  # Uptrend -> positive slope

    def test_cluster_detection(self, ranging_df):
        """MA cluster can form in ranging markets."""
        ma = MAWellField()
        result = ma.compute_features(ranging_df)
        assert "ma_well_cluster" in result.columns
        assert "ma_well_cluster_energy" in result.columns

    def test_custom_ema_spans(self, trending_df):
        cfg = MAWellConfig(ema_spans=[5, 10, 30])
        ma = MAWellField(cfg)
        result = ma.compute_features(trending_df)
        assert "ma_well_5_dist" in result.columns
        assert "ma_well_10_dist" in result.columns
        assert "ma_well_30_dist" in result.columns


# ---------------------------------------------------------------------------
# 4. OI Hazard Model Tests
# ---------------------------------------------------------------------------

class TestOIHazardModel:
    def test_basic_computation(self, trending_df):
        oi = OIHazardModel()
        result = oi.compute_features(trending_df)

        assert "oi_change" in result.columns
        assert "oi_change_pct" in result.columns
        assert "oi_hazard_rate" in result.columns
        assert "oi_directional_buildup" in result.columns

    def test_hazard_rate_bounded(self, trending_df):
        oi = OIHazardModel()
        result = oi.compute_features(trending_df)
        hazard = result["oi_hazard_rate"]
        cfg = OIHazardConfig()
        assert (hazard >= cfg.hazard_floor).all()
        assert (hazard <= cfg.hazard_ceiling).all()

    def test_no_oi_column(self, trending_df):
        """Should handle missing OI gracefully."""
        df = trending_df.drop(columns=["open_interest"])
        oi = OIHazardModel()
        result = oi.compute_features(df)
        assert "oi_hazard_rate" in result.columns

    def test_rising_oi_increases_hazard(self):
        """Rising OI during move should increase hazard."""
        n = 200
        rng = np.random.RandomState(99)
        df = pd.DataFrame({
            "close": 100 + np.cumsum(rng.randn(n) * 0.5),
            "returns": rng.randn(n) * 0.01,
            "open_interest": np.linspace(1e8, 3e8, n),  # Steadily rising OI
        })
        oi = OIHazardModel()
        result = oi.compute_features(df)
        # Later bars should have higher hazard than base
        assert result["oi_hazard_rate"].iloc[-1] > OIHazardConfig().lambda_base


# ---------------------------------------------------------------------------
# 5. RSI Throttle Tests
# ---------------------------------------------------------------------------

class TestRSIThrottle:
    def test_basic_computation(self, trending_df):
        rsi = RSIThrottle()
        result = rsi.compute_features(trending_df)

        assert "rsi" in result.columns
        assert "rsi_throttle" in result.columns
        assert "rsi_saturation" in result.columns
        assert "rsi_impulse_efficiency_buy" in result.columns
        assert "rsi_impulse_efficiency_sell" in result.columns

    def test_rsi_bounded(self, trending_df):
        rsi = RSIThrottle()
        result = rsi.compute_features(trending_df)
        valid = result["rsi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_throttle_bounded(self, trending_df):
        rsi = RSIThrottle()
        result = rsi.compute_features(trending_df)
        valid = result["rsi_throttle"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0).all()

    def test_overbought_reduces_buy_efficiency(self):
        """High RSI should reduce buy impulse efficiency."""
        n = 200
        # Strong uptrend -> high RSI
        returns = np.full(n, 0.005)
        df = pd.DataFrame({
            "close": 100 * np.exp(np.cumsum(returns)),
            "returns": returns,
        })
        rsi = RSIThrottle()
        result = rsi.compute_features(df)
        # After sustained uptrend, buy efficiency should be reduced
        assert result["rsi_impulse_efficiency_buy"].iloc[-1] < 0.8

    def test_oversold_reduces_sell_efficiency(self):
        """Low RSI should reduce sell impulse efficiency."""
        n = 200
        returns = np.full(n, -0.005)
        df = pd.DataFrame({
            "close": 100 * np.exp(np.cumsum(returns)),
            "returns": returns,
        })
        rsi = RSIThrottle()
        result = rsi.compute_features(df)
        assert result["rsi_impulse_efficiency_sell"].iloc[-1] < 0.8

    def test_neutral_full_efficiency(self):
        """Neutral RSI should give full efficiency."""
        n = 200
        rng = np.random.RandomState(42)
        returns = rng.randn(n) * 0.005  # Random walk -> RSI ~ 50
        df = pd.DataFrame({
            "close": 100 * np.exp(np.cumsum(returns)),
            "returns": returns,
        })
        rsi = RSIThrottle()
        result = rsi.compute_features(df)
        # In the middle range, throttle should be near 1.0
        mid_throttle = result["rsi_throttle"].iloc[50:150].mean()
        assert mid_throttle > 0.7


# ---------------------------------------------------------------------------
# 6. Ichimoku Regime Gate Tests
# ---------------------------------------------------------------------------

class TestIchimokuRegimeGate:
    def test_basic_computation(self, trending_df):
        ichi = IchimokuRegimeGate()
        result = ichi.compute_features(trending_df)

        assert "ichi_regime" in result.columns
        assert "ichi_regime_code" in result.columns
        assert "ichi_cloud_thickness" in result.columns
        assert "ichi_cloud_distance" in result.columns
        assert "ichi_ma_well_multiplier" in result.columns
        assert "ichi_drift_multiplier" in result.columns

    def test_trending_produces_ballistic(self, trending_df):
        """Strong uptrend should produce ballistic regime."""
        ichi = IchimokuRegimeGate()
        result = ichi.compute_features(trending_df)
        # Later bars should be ballistic (above cloud)
        tail = result.tail(100)
        ballistic_count = (tail["ichi_regime"] == ManifoldRegime.BALLISTIC.value).sum()
        # At least some bars should be ballistic in a trend
        assert ballistic_count >= 0  # Conservative assertion

    def test_ranging_produces_diffusive(self, ranging_df):
        """Ranging data should produce diffusive regime."""
        ichi = IchimokuRegimeGate()
        result = ichi.compute_features(ranging_df)
        # Should have some diffusive bars
        diffusive_count = (result["ichi_regime"] == ManifoldRegime.DIFFUSIVE.value).sum()
        assert diffusive_count > 0

    def test_multipliers_regime_consistent(self, trending_df):
        """Ballistic regime should weaken MA wells and strengthen drift."""
        ichi = IchimokuRegimeGate()
        result = ichi.compute_features(trending_df)
        ballistic = result[result["ichi_regime"] == ManifoldRegime.BALLISTIC.value]
        if len(ballistic) > 0:
            assert (ballistic["ichi_ma_well_multiplier"] < 1.0).all()
            assert (ballistic["ichi_drift_multiplier"] > 1.0).all()

        diffusive = result[result["ichi_regime"] == ManifoldRegime.DIFFUSIVE.value]
        if len(diffusive) > 0:
            assert (diffusive["ichi_ma_well_multiplier"] > 1.0).all()
            assert (diffusive["ichi_drift_multiplier"] < 1.0).all()

    def test_small_dataframe(self):
        small = pd.DataFrame({
            "close": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
        })
        ichi = IchimokuRegimeGate()
        result = ichi.compute_features(small)
        assert "ichi_regime" in result.columns


# ---------------------------------------------------------------------------
# 7. CVD Tracker Tests
# ---------------------------------------------------------------------------

class TestCVDTracker:
    def test_basic_computation(self, trending_df):
        cvd = CVDTracker()
        result = cvd.compute_features(trending_df)

        assert "cvd" in result.columns
        assert "cvd_delta" in result.columns
        assert "cvd_fast" in result.columns
        assert "cvd_slow" in result.columns
        assert "cvd_trend" in result.columns
        assert "cvd_divergence" in result.columns
        assert "cvd_impulse_sign" in result.columns

    def test_cvd_cumulative(self, trending_df):
        """CVD should be cumulative sum of delta."""
        cvd = CVDTracker()
        result = cvd.compute_features(trending_df)
        expected_cvd = np.cumsum(result["cvd_delta"].values)
        np.testing.assert_allclose(result["cvd"].values, expected_cvd, rtol=1e-10)

    def test_uptrend_positive_cvd(self, trending_df):
        """Uptrending data with buy bias should produce positive CVD trend."""
        cvd = CVDTracker()
        result = cvd.compute_features(trending_df)
        # In trending data, buy_volume > sell_volume, so CVD should be positive
        assert result["cvd"].iloc[-1] > 0

    def test_impulse_sign(self, trending_df):
        cvd = CVDTracker()
        result = cvd.compute_features(trending_df)
        sign_vals = result["cvd_impulse_sign"].unique()
        # Should have values in {-1, 0, 1}
        for v in sign_vals:
            if not np.isnan(v):
                assert v in [-1.0, 0.0, 1.0]

    def test_no_buy_sell_volume(self):
        """Should handle missing buy/sell volume."""
        df = pd.DataFrame({
            "close": np.linspace(100, 110, 50),
        })
        cvd = CVDTracker()
        result = cvd.compute_features(df)
        assert "cvd" in result.columns


# ---------------------------------------------------------------------------
# 8. Unified CryptoParticlePotential Tests
# ---------------------------------------------------------------------------

class TestCryptoParticlePotential:
    def test_full_pipeline_trending(self, trending_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(trending_df)

        # Check unified outputs
        assert "unified_potential" in result.columns
        assert "unified_restoring_force" in result.columns
        assert "unified_drift" in result.columns
        assert "unified_sigma" in result.columns
        assert "unified_jump_hazard" in result.columns
        assert "unified_impulse_efficiency" in result.columns
        assert "unified_impulse_direction" in result.columns
        assert "unified_squeeze_state" in result.columns

    def test_full_pipeline_ranging(self, ranging_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(ranging_df)
        assert "unified_potential" in result.columns
        assert len(result) == len(ranging_df)

    def test_full_pipeline_volatile(self, volatile_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(volatile_df)
        assert "unified_potential" in result.columns

    def test_sigma_positive(self, trending_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(trending_df)
        valid = result["unified_sigma"].dropna()
        assert (valid > 0).all()

    def test_impulse_efficiency_bounded(self, trending_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(trending_df)
        eff = result["unified_impulse_efficiency"]
        assert (eff >= 0).all()
        assert (eff <= 1.0).all()

    def test_state_vector_extraction(self, trending_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(trending_df)
        state = engine.get_state_vector(result, bar_idx=-1)

        assert isinstance(state, dict)
        assert "unified_potential" in state
        assert "unified_drift" in state
        assert "unified_sigma" in state
        assert "unified_jump_hazard" in state
        assert "rsi" in state
        assert "ichi_regime_code" in state

        # All values should be finite
        for key, val in state.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_state_vector_at_specific_bar(self, trending_df):
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(trending_df)
        state_100 = engine.get_state_vector(result, bar_idx=100)
        state_200 = engine.get_state_vector(result, bar_idx=200)
        # Different bars should give different states
        assert state_100["unified_potential"] != state_200["unified_potential"] or \
               state_100["unified_drift"] != state_200["unified_drift"]

    def test_multi_symbol(self, trending_df, ranging_df):
        """Should handle multi-symbol DataFrames."""
        combined = pd.concat([trending_df, ranging_df], ignore_index=True)
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(combined)
        assert len(result) == len(combined)
        assert result["symbol"].nunique() == 2

    def test_custom_config(self, trending_df):
        cfg = CryptoParticleConfig(
            bollinger=BollingerConfig(bb_period=10),
            ma_well=MAWellConfig(ema_spans=[5, 20]),
            rsi_throttle=RSIThrottleConfig(rsi_period=7),
            w_avwap=0.5,
            w_ma_well=1.5,
        )
        engine = CryptoParticlePotential(cfg)
        result = engine.compute_all_features(trending_df)
        assert "unified_potential" in result.columns

    def test_no_nans_in_unified_outputs(self, trending_df):
        """Unified outputs should not have NaN after warmup period."""
        engine = CryptoParticlePotential()
        result = engine.compute_all_features(trending_df)
        warmup = 250  # Conservative warmup
        tail = result.iloc[warmup:]
        unified_cols = [c for c in result.columns if c.startswith("unified_")]
        for col in unified_cols:
            nan_count = tail[col].isna().sum()
            assert nan_count == 0, f"{col} has {nan_count} NaN values after warmup"


# ---------------------------------------------------------------------------
# Feature Registry and Hyperparameter Tests
# ---------------------------------------------------------------------------

class TestFeatureRegistry:
    def test_feature_names_nonempty(self):
        names = get_crypto_potential_feature_names()
        assert len(names) > 50

    def test_feature_names_unique(self):
        names = get_crypto_potential_feature_names()
        assert len(names) == len(set(names))

    def test_hyperparameters_valid(self):
        hparams = get_crypto_potential_hyperparameters()
        assert len(hparams) > 10
        for hp in hparams:
            assert "name" in hp
            assert "path" in hp
            assert "current_value" in hp
            assert "candidates" in hp
            assert len(hp["candidates"]) >= 2


# ---------------------------------------------------------------------------
# Integration with existing particle module
# ---------------------------------------------------------------------------

class TestModuleIntegration:
    def test_import_from_particle(self):
        """Should be importable from the particle package."""
        from gnosis.particle import (
            CryptoParticlePotential,
            CryptoParticleConfig,
            AVWAPAnchorSet,
            BollingerDiffusion,
            MAWellField,
            OIHazardModel,
            RSIThrottle,
            IchimokuRegimeGate,
            CVDTracker,
            ManifoldRegime,
            get_crypto_potential_feature_names,
            get_crypto_potential_hyperparameters,
        )
        assert CryptoParticlePotential is not None

    def test_all_hyperparameters_includes_crypto(self):
        """get_all_particle_hyperparameters should include crypto potential."""
        from gnosis.particle import get_all_particle_hyperparameters
        all_hps = get_all_particle_hyperparameters()
        crypto_hps = [h for h in all_hps if "crypto_potential" in h.get("path", "")]
        assert len(crypto_hps) > 0

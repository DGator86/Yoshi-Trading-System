"""KPCOFGS regime classification with probability distributions."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# =============================================================================
# Classification Constants
# =============================================================================

# Temperature for softmax (lower = sharper probability distributions)
DEFAULT_TEMPERATURE = 2.0

# Kinetics (K-level) thresholds
# K_TRENDING: |returns| > vol * K_TRENDING_VOL_MULT
# K_MEAN_REVERTING: |returns| < vol * K_MEAN_REV_VOL_MULT
K_TRENDING_VOL_MULT = 1.5
K_MEAN_REV_VOL_MULT = 0.5

# Pressure (P-level) thresholds
# P_VOL_EXPANDING: vol > vol_prev * P_VOL_EXPAND_MULT
# P_VOL_CONTRACTING: vol < vol_prev * P_VOL_CONTRACT_MULT
P_VOL_EXPAND_MULT = 1.2
P_VOL_CONTRACT_MULT = 0.8

# Current (C-level) thresholds
# Buy/Sell flow dominant when |ofi| exceeds this threshold
C_OFI_THRESHOLD = 0.3
C_OFI_LOGIT_MULT = 3  # Multiplier for OFI logit scaling

# Oscillation (O-level) thresholds
O_SWEEP_OFI_THRESHOLD = 0.5
O_SWEEP_LOGIT_MULT = 2  # Multiplier for sweep-revert logit

# Flow (F-level) thresholds
F_MOM_CHANGE_VOL_MULT = 0.5  # Accel/decel threshold relative to vol
F_STALL_VOL_MULT = 0.2  # Stall threshold relative to vol

# Logit scores for rule-based classification
# Higher positive = strong match, negative = non-match
LOGIT_STRONG_MATCH = 3.0
LOGIT_STRONG_MISMATCH = -2.0
LOGIT_MODERATE_MATCH = 2.0
LOGIT_MODERATE_MISMATCH = -1.0
LOGIT_WEAK_MATCH = 2.5
LOGIT_WEAK_MISMATCH = -1.5

# Species (S-level) thresholds
S_SMALL_RETURN_THRESHOLD = 0.001  # For S_RANGE_MID_MEANREV classification


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Compute entropy from probability distribution. probs shape: (N, K)."""
    # Clip to avoid log(0)
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


class KPCOFGSClassifier:
    """Rule-based KPCOFGS regime classifier with probability distributions.

    Produces for each level:
    - {level}_label: categorical label (string)
    - {level}_pmax: maximum probability (float in [0,1])
    - {level}_entropy: entropy of probability distribution (float >= 0)
    - {level}_probs_*: individual class probabilities

    Also produces:
    - regime_entropy: sum of all level entropies
    """

    def __init__(self, regimes_config: dict):
        self.config = regimes_config
        # Define label sets for each level
        self.K_labels = ["K_TRENDING", "K_MEAN_REVERTING", "K_BALANCED"]
        self.P_labels = ["P_VOL_EXPANDING", "P_VOL_CONTRACTING", "P_VOL_STABLE"]
        self.C_labels = ["C_BUY_FLOW_DOMINANT", "C_SELL_FLOW_DOMINANT", "C_FLOW_NEUTRAL"]
        self.O_labels = ["O_BREAKOUT", "O_BREAKDOWN", "O_RANGE", "O_SWEEP_REVERT"]
        self.F_labels = ["F_ACCEL", "F_DECEL", "F_STALL", "F_REVERSAL"]
        self.G_labels = ["G_TREND_CONT", "G_TREND_EXH", "G_MR_BOUNCE", "G_MR_FADE", "G_BO_HOLD", "G_BO_FAIL"]
        self.S_labels = [
            "S_TC_PULLBACK_RESUME", "S_TC_ACCEL_BREAK", "S_TX_TOPPING_ROLL",
            "S_MR_OVERSHOOT_SNAPBACK", "S_MR_GRIND_BACK", "S_BO_LEVEL_BREAK_HOLD",
            "S_BO_LEVEL_BREAK_FAIL", "S_RANGE_EDGE_FADE", "S_RANGE_MID_MEANREV",
            "S_SWEEP_UP_REVERT", "S_SWEEP_DOWN_REVERT", "S_UNCERTAIN"
        ]

        # Temperature for softmax (lower = sharper distributions)
        self.temperature = self.config.get("temperature", DEFAULT_TEMPERATURE)

    def _classify_K(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Classify K-level (Kinetics) with probabilities.

        Returns: (probs array shape (N, 3), labels array shape (N,))
        """
        n = len(df)
        returns_abs = df["returns"].abs().fillna(0).values
        vol = df["realized_vol"].fillna(df["realized_vol"].median()).values
        # Ensure no NaN in vol (use small positive value if all NaN)
        vol = np.where(np.isnan(vol), 0.01, vol)

        # Compute logits based on distance from decision boundaries
        # K_TRENDING: |returns| > vol * K_TRENDING_VOL_MULT
        # K_MEAN_REVERTING: |returns| < vol * K_MEAN_REV_VOL_MULT
        # K_BALANCED: otherwise

        logits = np.zeros((n, 3))

        # Trending score: how much |returns| exceeds vol * K_TRENDING_VOL_MULT
        trending_score = (returns_abs - vol * K_TRENDING_VOL_MULT) / (vol + 1e-10)
        logits[:, 0] = trending_score * self.temperature

        # Mean-reverting score: how much vol * K_MEAN_REV_VOL_MULT exceeds |returns|
        mr_score = (vol * K_MEAN_REV_VOL_MULT - returns_abs) / (vol + 1e-10)
        logits[:, 1] = mr_score * self.temperature

        # Balanced: base case (neutral logit)
        logits[:, 2] = 0.0

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.K_labels)[labels_idx]

        return probs, labels

    def _classify_P(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Classify P-level (Pressure/Volatility regime) with probabilities."""
        n = len(df)
        vol = df["realized_vol"].fillna(df["realized_vol"].median()).values
        vol = np.where(np.isnan(vol), 0.01, vol)
        vol_prev = np.roll(vol, 1)
        vol_prev[0] = vol[0]  # Handle first element

        logits = np.zeros((n, 3))

        # Vol expanding: vol > vol_prev * P_VOL_EXPAND_MULT
        expanding_score = (vol - vol_prev * P_VOL_EXPAND_MULT) / (vol_prev + 1e-10)
        logits[:, 0] = expanding_score * self.temperature

        # Vol contracting: vol < vol_prev * P_VOL_CONTRACT_MULT
        contracting_score = (vol_prev * P_VOL_CONTRACT_MULT - vol) / (vol_prev + 1e-10)
        logits[:, 1] = contracting_score * self.temperature

        # Stable: base case
        logits[:, 2] = 0.0

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.P_labels)[labels_idx]

        return probs, labels

    def _classify_C(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Classify C-level (Current/Order flow) with probabilities."""
        n = len(df)
        ofi = df["ofi"].fillna(0).values

        logits = np.zeros((n, 3))

        # Buy flow dominant: ofi > C_OFI_THRESHOLD
        logits[:, 0] = (ofi - C_OFI_THRESHOLD) * self.temperature * C_OFI_LOGIT_MULT

        # Sell flow dominant: ofi < -C_OFI_THRESHOLD
        logits[:, 1] = (-ofi - C_OFI_THRESHOLD) * self.temperature * C_OFI_LOGIT_MULT

        # Neutral: base case
        logits[:, 2] = 0.0

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.C_labels)[labels_idx]

        return probs, labels

    def _classify_O(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Classify O-level (Oscillation/Price structure) with probabilities."""
        n = len(df)
        returns = df["returns"].fillna(0).values
        range_pct = df["range_pct"].fillna(0).values
        range_mean = pd.Series(range_pct).rolling(20, min_periods=1).mean().fillna(0).values
        ofi_abs = df["ofi"].abs().fillna(0).values

        logits = np.zeros((n, 4))

        # Breakout: returns > 0 AND range > range_mean
        breakout_cond = (returns > 0) & (range_pct > range_mean)
        logits[:, 0] = np.where(breakout_cond, LOGIT_MODERATE_MATCH, LOGIT_MODERATE_MISMATCH) * self.temperature

        # Breakdown: returns < 0 AND range > range_mean
        breakdown_cond = (returns < 0) & (range_pct > range_mean)
        logits[:, 1] = np.where(breakdown_cond, LOGIT_MODERATE_MATCH, LOGIT_MODERATE_MISMATCH) * self.temperature

        # Range: base case
        logits[:, 2] = 0.0

        # Sweep revert: |ofi| > O_SWEEP_OFI_THRESHOLD
        logits[:, 3] = (ofi_abs - O_SWEEP_OFI_THRESHOLD) * self.temperature * O_SWEEP_LOGIT_MULT

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.O_labels)[labels_idx]

        return probs, labels

    def _classify_F(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Classify F-level (Flow/Momentum state) with probabilities."""
        n = len(df)
        returns = df["returns"].fillna(0).values
        vol = df["realized_vol"].fillna(df["realized_vol"].median()).values
        vol = np.where(np.isnan(vol), 0.01, vol)

        # Compute momentum and momentum change
        mom = pd.Series(returns).rolling(5, min_periods=1).mean().fillna(0).values
        mom_shifted = np.roll(mom, 5)
        mom_shifted[:5] = mom[:5]
        mom_change = mom - mom_shifted

        logits = np.zeros((n, 4))

        # Accel: mom_change > vol * F_MOM_CHANGE_VOL_MULT
        logits[:, 0] = (mom_change - vol * F_MOM_CHANGE_VOL_MULT) / (vol + 1e-10) * self.temperature

        # Decel: mom_change < -vol * F_MOM_CHANGE_VOL_MULT
        logits[:, 1] = (-mom_change - vol * F_MOM_CHANGE_VOL_MULT) / (vol + 1e-10) * self.temperature

        # Stall: |mom| < vol * F_STALL_VOL_MULT
        stall_score = (vol * F_STALL_VOL_MULT - np.abs(mom)) / (vol + 1e-10)
        logits[:, 2] = stall_score * self.temperature

        # Reversal: base case
        logits[:, 3] = 0.0

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.F_labels)[labels_idx]

        return probs, labels

    def _classify_G(self, df: pd.DataFrame, K_labels: np.ndarray, F_labels: np.ndarray,
                    O_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify G-level (Gear/Tactical state) with probabilities."""
        n = len(df)
        returns = df["returns"].fillna(0).values

        logits = np.zeros((n, 6))

        # G_TREND_CONT: K=TRENDING AND F=ACCEL
        tc_cond = (K_labels == "K_TRENDING") & (F_labels == "F_ACCEL")
        logits[:, 0] = np.where(tc_cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # G_TREND_EXH: K=TRENDING AND F=DECEL
        te_cond = (K_labels == "K_TRENDING") & (F_labels == "F_DECEL")
        logits[:, 1] = np.where(te_cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # G_MR_BOUNCE: K=MR AND returns > 0
        mrb_cond = (K_labels == "K_MEAN_REVERTING") & (returns > 0)
        logits[:, 2] = np.where(mrb_cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # G_MR_FADE: K=MR AND returns < 0
        mrf_cond = (K_labels == "K_MEAN_REVERTING") & (returns < 0)
        logits[:, 3] = np.where(mrf_cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # G_BO_HOLD: O=BREAKOUT
        bh_cond = (O_labels == "O_BREAKOUT")
        logits[:, 4] = np.where(bh_cond, LOGIT_MODERATE_MATCH, LOGIT_MODERATE_MISMATCH) * self.temperature

        # G_BO_FAIL: base case
        logits[:, 5] = 0.0

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.G_labels)[labels_idx]

        return probs, labels

    def _classify_S(self, df: pd.DataFrame, G_labels: np.ndarray, F_labels: np.ndarray,
                    O_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify S-level (Species/Specific setup) with probabilities."""
        n = len(df)
        ofi = df["ofi"].fillna(0).values
        returns = df["returns"].fillna(0).values

        # 12 species labels
        logits = np.zeros((n, 12))

        # S_TC_PULLBACK_RESUME: G=TREND_CONT AND F=DECEL
        cond = (G_labels == "G_TREND_CONT") & (F_labels == "F_DECEL")
        logits[:, 0] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_TC_ACCEL_BREAK: G=TREND_CONT AND F=ACCEL
        cond = (G_labels == "G_TREND_CONT") & (F_labels == "F_ACCEL")
        logits[:, 1] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_TX_TOPPING_ROLL: G=TREND_EXH
        cond = (G_labels == "G_TREND_EXH")
        logits[:, 2] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_MR_OVERSHOOT_SNAPBACK: G=MR_BOUNCE AND ofi > C_OFI_THRESHOLD
        cond = (G_labels == "G_MR_BOUNCE") & (ofi > C_OFI_THRESHOLD)
        logits[:, 3] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_MR_GRIND_BACK: G=MR_FADE
        cond = (G_labels == "G_MR_FADE")
        logits[:, 4] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_BO_LEVEL_BREAK_HOLD: G=BO_HOLD
        cond = (G_labels == "G_BO_HOLD")
        logits[:, 5] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_BO_LEVEL_BREAK_FAIL: G=BO_FAIL
        cond = (G_labels == "G_BO_FAIL")
        logits[:, 6] = np.where(cond, LOGIT_MODERATE_MATCH, LOGIT_MODERATE_MISMATCH) * self.temperature

        # S_RANGE_EDGE_FADE: O=RANGE
        cond = (O_labels == "O_RANGE")
        logits[:, 7] = np.where(cond, LOGIT_MODERATE_MATCH, LOGIT_MODERATE_MISMATCH) * self.temperature

        # S_RANGE_MID_MEANREV: O=RANGE and small returns
        cond = (O_labels == "O_RANGE") & (np.abs(returns) < S_SMALL_RETURN_THRESHOLD)
        logits[:, 8] = np.where(cond, LOGIT_WEAK_MATCH, LOGIT_WEAK_MISMATCH) * self.temperature

        # S_SWEEP_UP_REVERT: O=SWEEP_REVERT AND returns > 0
        cond = (O_labels == "O_SWEEP_REVERT") & (returns > 0)
        logits[:, 9] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_SWEEP_DOWN_REVERT: O=SWEEP_REVERT AND returns < 0
        cond = (O_labels == "O_SWEEP_REVERT") & (returns < 0)
        logits[:, 10] = np.where(cond, LOGIT_STRONG_MATCH, LOGIT_STRONG_MISMATCH) * self.temperature

        # S_UNCERTAIN: base case (default)
        logits[:, 11] = 0.0

        probs = _softmax(logits)
        labels_idx = np.argmax(probs, axis=1)
        labels = np.array(self.S_labels)[labels_idx]

        return probs, labels

    def classify(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar into KPCOFGS regimes with probability distributions.

        Adds columns for each level:
        - {level}_label: the predicted label
        - {level}_pmax: maximum probability
        - {level}_entropy: entropy of distribution

        Plus overall regime_entropy (sum of all level entropies).
        """
        df = features_df.copy()

        # Classify each level hierarchically
        K_probs, K_labels = self._classify_K(df)
        P_probs, P_labels = self._classify_P(df)
        C_probs, C_labels = self._classify_C(df)
        O_probs, O_labels = self._classify_O(df)
        F_probs, F_labels = self._classify_F(df)
        G_probs, G_labels = self._classify_G(df, K_labels, F_labels, O_labels)
        S_probs, S_labels = self._classify_S(df, G_labels, F_labels, O_labels)

        # Store labels (keep old column names K, P, C, O, F, G, S for compatibility)
        df["K"] = K_labels
        df["P"] = P_labels
        df["C"] = C_labels
        df["O"] = O_labels
        df["F"] = F_labels
        df["G"] = G_labels
        df["S"] = S_labels

        # Store labeled columns (new Phase C format)
        df["K_label"] = K_labels
        df["P_label"] = P_labels
        df["C_label"] = C_labels
        df["O_label"] = O_labels
        df["F_label"] = F_labels
        df["G_label"] = G_labels
        df["S_label"] = S_labels

        # Store pmax and entropy for each level
        df["K_pmax"] = np.max(K_probs, axis=1)
        df["K_entropy"] = _entropy(K_probs)

        df["P_pmax"] = np.max(P_probs, axis=1)
        df["P_entropy"] = _entropy(P_probs)

        df["C_pmax"] = np.max(C_probs, axis=1)
        df["C_entropy"] = _entropy(C_probs)

        df["O_pmax"] = np.max(O_probs, axis=1)
        df["O_entropy"] = _entropy(O_probs)

        df["F_pmax"] = np.max(F_probs, axis=1)
        df["F_entropy"] = _entropy(F_probs)

        df["G_pmax"] = np.max(G_probs, axis=1)
        df["G_entropy"] = _entropy(G_probs)

        df["S_pmax"] = np.max(S_probs, axis=1)
        df["S_entropy"] = _entropy(S_probs)

        # Store S probabilities for calibration (needed for isotonic calibration)
        for i, label in enumerate(self.S_labels):
            df[f"S_prob_{label}"] = S_probs[:, i]

        # Overall regime entropy = sum of level entropies
        df["regime_entropy"] = (
            df["K_entropy"] + df["P_entropy"] + df["C_entropy"] +
            df["O_entropy"] + df["F_entropy"] + df["G_entropy"] + df["S_entropy"]
        )

        return df

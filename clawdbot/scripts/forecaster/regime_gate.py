"""
Regime gating and arbitrage helpers for the forecaster stack.

This module intentionally keeps policy logic deterministic and compact:
- damp/invert direction probabilities in weak regimes
- require a minimum post-gate edge before "trade"
- expose simple arbitrage checks used by scanner/orchestrator paths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import PredictionTargets, Regime


@dataclass
class RegimeProfile:
    """Historical performance prior for a regime."""

    name: str
    tier: str = "weak"  # "strong" | "weak" | "blocked"
    observed_hr: float = 0.50
    n_samples: int = 0
    confidence_multiplier: float = 1.0
    min_confidence_to_trade: float = 0.55
    invert_signal: bool = False
    notes: str = ""


DEFAULT_REGIME_PROFILES = {
    Regime.TREND_UP: RegimeProfile(
        name="trend_up",
        tier="weak",
        observed_hr=0.53,
        n_samples=8,
        confidence_multiplier=0.85,
        min_confidence_to_trade=0.56,
        notes="Small sample; keep cautious.",
    ),
    Regime.TREND_DOWN: RegimeProfile(
        name="trend_down",
        tier="strong",
        observed_hr=0.58,
        n_samples=12,
        confidence_multiplier=1.05,
        min_confidence_to_trade=0.53,
        notes="Best empirical regime so far.",
    ),
    Regime.RANGE: RegimeProfile(
        name="range",
        tier="blocked",
        observed_hr=0.46,
        n_samples=52,
        confidence_multiplier=0.50,
        min_confidence_to_trade=0.60,
        notes="Historically anti-predictive/low edge.",
    ),
    Regime.VOL_EXPANSION: RegimeProfile(
        name="vol_expansion",
        tier="weak",
        observed_hr=0.50,
        n_samples=7,
        confidence_multiplier=0.75,
        min_confidence_to_trade=0.57,
    ),
    Regime.POST_JUMP: RegimeProfile(
        name="post_jump",
        tier="weak",
        observed_hr=0.50,
        n_samples=6,
        confidence_multiplier=0.75,
        min_confidence_to_trade=0.57,
    ),
    Regime.CASCADE_RISK: RegimeProfile(
        name="cascade_risk",
        tier="blocked",
        observed_hr=0.40,
        n_samples=5,
        confidence_multiplier=0.45,
        min_confidence_to_trade=0.65,
        invert_signal=False,
        notes="Tail regime; avoid directional trading.",
    ),
    Regime.EVENT_WINDOW: RegimeProfile(
        name="event_window",
        tier="blocked",
        observed_hr=0.50,
        n_samples=0,
        confidence_multiplier=0.40,
        min_confidence_to_trade=0.70,
    ),
    Regime.ILLIQUID: RegimeProfile(
        name="illiquid",
        tier="blocked",
        observed_hr=0.45,
        n_samples=2,
        confidence_multiplier=0.35,
        min_confidence_to_trade=0.65,
    ),
}


@dataclass
class GateDecision:
    """Result of applying the regime gate."""

    action: str = "skip"  # "trade" | "trade_cautious" | "skip"
    regime: str = "range"
    tier: str = "weak"
    original_direction_prob: float = 0.50
    gated_direction_prob: float = 0.50
    confidence_multiplier: float = 1.0
    ev_edge: float = 0.0  # abs(p - 0.5) after gate
    min_ev_required: float = 0.04
    reason: str = ""


class RegimeGate:
    """Applies deterministic confidence/edge policy by regime."""

    DEFAULT_MIN_EV_EDGE = 0.04

    def __init__(
        self,
        profiles: dict[Regime, RegimeProfile] | None = None,
        min_ev_edge: float | None = None,
    ):
        self.profiles = profiles or DEFAULT_REGIME_PROFILES
        self.min_ev_edge = (
            float(min_ev_edge)
            if min_ev_edge is not None
            else float(self.DEFAULT_MIN_EV_EDGE)
        )
        self._live_hits: dict[str, list[bool]] = {}

    def _dominant_regime(self, regime_probs: dict[Regime, float]) -> Regime:
        if not regime_probs:
            return Regime.RANGE
        return max(regime_probs, key=regime_probs.get)

    def apply(
        self,
        targets: PredictionTargets,
        regime_probs: dict[Regime, float],
    ) -> GateDecision:
        """Mutate targets in-place and return gate decision."""
        dominant = self._dominant_regime(regime_probs)
        profile = self.profiles.get(dominant, self.profiles[Regime.RANGE])
        original = float(targets.direction_prob)

        # 1) Scale distance from 0.50 by regime confidence multiplier.
        dist = original - 0.50
        gated = 0.50 + dist * float(profile.confidence_multiplier)

        # 2) Optional inversion for truly anti-predictive profiles.
        if profile.invert_signal:
            gated = 1.0 - gated
            targets.expected_return = -float(targets.expected_return)

        # 3) Extra dampening for blocked regimes.
        if profile.tier == "blocked":
            dampen = 0.60
            gated = gated * (1 - dampen) + 0.50 * dampen
            targets.expected_return *= 0.60

        # Keep in sane bounds.
        gated = max(0.05, min(0.95, float(gated)))
        targets.direction_prob = gated

        # 4) Trade decision from confidence and minimum EV edge.
        edge = abs(gated - 0.50)
        regime_threshold = abs(float(profile.min_confidence_to_trade) - 0.50)
        should_trade = edge >= max(self.min_ev_edge, regime_threshold)

        if profile.tier == "strong":
            action = "trade" if should_trade else "skip"
        elif profile.tier == "weak":
            action = "trade_cautious" if should_trade else "skip"
        else:
            action = "skip"

        return GateDecision(
            action=action,
            regime=dominant.value,
            tier=profile.tier,
            original_direction_prob=original,
            gated_direction_prob=gated,
            confidence_multiplier=float(profile.confidence_multiplier),
            ev_edge=round(edge, 4),
            min_ev_required=round(self.min_ev_edge, 4),
            reason=profile.notes,
        )

    def record_outcome(self, regime: str, was_correct: bool):
        """Track online outcomes and gently adapt profile confidence."""
        self._live_hits.setdefault(regime, []).append(bool(was_correct))
        hits = self._live_hits[regime]
        if len(hits) < 20:
            return
        live_hr = sum(hits) / len(hits)
        for _, profile in self.profiles.items():
            if profile.name != regime:
                continue
            blended = 0.7 * live_hr + 0.3 * profile.observed_hr
            profile.observed_hr = blended
            if blended >= 0.55:
                profile.tier = "strong"
                profile.confidence_multiplier = min(1.20, max(0.80, blended + 0.25))
            elif blended >= 0.48:
                profile.tier = "weak"
                profile.confidence_multiplier = min(1.0, max(0.65, blended + 0.15))
            else:
                profile.tier = "blocked"
                profile.confidence_multiplier = max(0.35, blended)
            break


@dataclass
class ArbitrageOpportunity:
    """Structured arbitrage/edge opportunity."""

    ticker: str = ""
    type: str = "spread"
    buy_side: str = "yes"
    buy_price_cents: int = 0
    sell_side: str = "payout"
    sell_price_cents: int = 100
    spread_cents: int = 0
    profit_pct: float = 0.0
    confidence: float = 0.95
    max_contracts: int = 10
    notes: str = ""


class ArbitrageDetector:
    """Simple spread/model-edge arbitrage checks."""

    def __init__(self, min_spread_cents: int = 2):
        self.min_spread_cents = int(min_spread_cents)

    def check_spread_arb(
        self,
        yes_ask: int,
        no_ask: int,
        ticker: str = "",
    ) -> Optional[ArbitrageOpportunity]:
        if yes_ask <= 0 or no_ask <= 0:
            return None
        total = yes_ask + no_ask
        spread = 100 - total
        if spread < self.min_spread_cents:
            return None
        pct = (spread / total * 100) if total > 0 else 0.0
        max_contracts = min(50, max(1, 500 // max(total, 1)))
        return ArbitrageOpportunity(
            ticker=ticker,
            type="spread",
            buy_side="yes+no",
            buy_price_cents=total,
            sell_side="guaranteed_payout",
            sell_price_cents=100,
            spread_cents=spread,
            profit_pct=round(pct, 2),
            confidence=0.99,
            max_contracts=max_contracts,
            notes=f"YES@{yes_ask} + NO@{no_ask} = {total}c",
        )

    def check_model_edge_arb(
        self,
        model_prob: float,
        market_prob: float,
        yes_ask: int,
        no_ask: int,
        ticker: str = "",
        min_edge: float = 0.15,
    ) -> Optional[ArbitrageOpportunity]:
        edge = abs(float(model_prob) - float(market_prob))
        if edge < float(min_edge):
            return None

        if model_prob >= market_prob:
            if yes_ask <= 0:
                return None
            ev = model_prob * (100 - yes_ask) - (1 - model_prob) * yes_ask
            if ev < 5:
                return None
            return ArbitrageOpportunity(
                ticker=ticker,
                type="model_edge",
                buy_side="yes",
                buy_price_cents=yes_ask,
                spread_cents=int(ev),
                profit_pct=round((ev / yes_ask) * 100, 2) if yes_ask > 0 else 0.0,
                confidence=min(0.95, float(model_prob)),
                notes=f"Model {model_prob:.0%} vs Market {market_prob:.0%}",
            )

        if no_ask <= 0:
            return None
        no_prob = 1.0 - float(model_prob)
        ev = no_prob * (100 - no_ask) - float(model_prob) * no_ask
        if ev < 5:
            return None
        return ArbitrageOpportunity(
            ticker=ticker,
            type="model_edge",
            buy_side="no",
            buy_price_cents=no_ask,
            spread_cents=int(ev),
            profit_pct=round((ev / no_ask) * 100, 2) if no_ask > 0 else 0.0,
            confidence=min(0.95, no_prob),
            notes=f"Model {model_prob:.0%} vs Market {market_prob:.0%}",
        )

    def scan_markets(
        self,
        markets: list[dict],
        model_probs: dict[str, float] | None = None,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []
        for mkt in markets:
            ticker = mkt.get("ticker", "")
            yes_ask = int(mkt.get("yes_ask", 0) or 0)
            no_ask = int(mkt.get("no_ask", 0) or 0)
            yes_bid = int(mkt.get("yes_bid", 0) or 0)

            spread = self.check_spread_arb(yes_ask=yes_ask, no_ask=no_ask, ticker=ticker)
            if spread is not None:
                opportunities.append(spread)

            if model_probs and ticker in model_probs:
                if yes_bid > 0 and yes_ask > 0:
                    market_prob = (yes_bid + yes_ask) / 200.0
                elif yes_ask > 0:
                    market_prob = yes_ask / 100.0
                else:
                    market_prob = 0.5
                edge = self.check_model_edge_arb(
                    model_prob=float(model_probs[ticker]),
                    market_prob=float(market_prob),
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                    ticker=ticker,
                )
                if edge is not None:
                    opportunities.append(edge)

        opportunities.sort(key=lambda x: x.spread_cents, reverse=True)
        return opportunities


def should_trade(
    targets: PredictionTargets,
    regime_probs: dict[Regime, float],
    gate: RegimeGate | None = None,
) -> tuple[bool, GateDecision]:
    gate = gate or RegimeGate()
    decision = gate.apply(targets, regime_probs)
    return decision.action in ("trade", "trade_cautious"), decision


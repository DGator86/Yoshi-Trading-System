"""
Kalshi LLM Value Analyzer — AI-powered binary market analysis.
================================================================
Takes ScanResults from the scanner and uses the LLM reasoning layer
to identify genuine value plays vs statistical noise.

Uses gnosis.reasoning.client for environment-aware LLM routing
(GenSpark proxy, OpenRouter free tier, direct OpenAI, or stub).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gnosis.reasoning.client import LLMClient, LLMConfig, LLMResponse
from gnosis.kalshi.scanner import ScanResult


# ── Value Play ─────────────────────────────────────────────
@dataclass
class ValuePlay:
    """An LLM-analyzed value opportunity."""
    scan: ScanResult
    value_score: float = 0.0       # 0-10 composite value score
    recommendation: str = "SKIP"   # BUY / SKIP / WATCH
    reasoning: str = ""
    risk_level: str = "UNKNOWN"    # LOW / MODERATE / HIGH / EXTREME
    confidence: float = 0.0        # LLM's self-assessed confidence 0-1
    suggested_size: int = 0        # Contracts to buy
    max_loss: float = 0.0          # Max $ loss
    expected_profit: float = 0.0   # Expected $ profit
    is_stub: bool = False
    llm_model: str = ""
    analysis_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "scan"}
        d["scan"] = self.scan.to_dict()
        return d

    def summary_line(self) -> str:
        emoji = {"BUY": "🟢", "SKIP": "🔴", "WATCH": "🟡"}.get(self.recommendation, "⚪")
        return (
            f"{emoji} {self.scan.ticker}: {self.recommendation} "
            f"(value={self.value_score:.1f}/10, risk={self.risk_level}) "
            f"| {self.scan.summary_line()}"
        )


# ── Prompt Builder ─────────────────────────────────────────
_SYSTEM_PROMPT = """You are a Kalshi binary options analyst. You receive market data
and statistical edge calculations. Your job is to evaluate whether the
statistical edge is REAL and TRADEABLE, or just noise.

You MUST return valid JSON only. No markdown, no explanation outside JSON.

Key rules:
- Be skeptical of small edges (<5%) — they often vanish after fees
- Volume and spread matter: illiquid markets have phantom edges
- Time to expiry matters: closer = more certain pricing
- Never recommend trading when the edge source is "market-baseline"
- A model_source of "price-distance" is a simple logistic — be conservative
- When forecast context is provided, treat it as the primary thesis:
  compare forecast end-of-hour expectation vs market-implied odds and
  focus on contracts with genuine forecast-vs-market mispricing.

Return this JSON structure:
{
  "value_score": <0.0 to 10.0>,
  "recommendation": "<BUY|SKIP|WATCH>",
  "reasoning": "<1-2 sentence explanation>",
  "risk_level": "<LOW|MODERATE|HIGH|EXTREME>",
  "confidence": <0.0 to 1.0>,
  "suggested_contracts": <integer>,
  "concerns": ["<list of risk factors>"],
  "catalyst": "<what would confirm or invalidate the edge>"
}"""


def _build_user_prompt(
    results: List[ScanResult],
    forecast_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build user prompt from scan results."""
    ctx_by_ticker: Dict[str, Dict[str, Any]] = {}
    if forecast_context:
        for row in forecast_context.get("opportunity_overrides", []) or []:
            ticker = str(row.get("ticker", ""))
            if ticker:
                ctx_by_ticker[ticker] = row

    lines = [
        "Analyze these Kalshi binary market opportunities for value:",
        "",
    ]
    if forecast_context:
        lines.extend(
            [
                "Forecast context (end-of-hour view):",
                f"- Symbol: {forecast_context.get('symbol', '?')}",
                f"- Current Price: ${float(forecast_context.get('current_price', 0.0)):,.2f}",
                f"- Predicted End Price: ${float(forecast_context.get('predicted_price_end_hour', 0.0)):,.2f}",
                f"- Forecast Confidence: {float(forecast_context.get('forecast_confidence', 0.0)):.1%}",
                f"- Horizon: {float(forecast_context.get('horizon_hours', 1.0)):.2f}h",
                f"- Regime: {forecast_context.get('regime', 'unknown')}",
                "",
            ]
        )
    for i, r in enumerate(results, 1):
        ctx = ctx_by_ticker.get(r.ticker, {})
        lines.extend([
            f"--- Opportunity #{i} ---",
            f"Ticker: {r.ticker}",
            f"Title: {r.title}" if r.title else "",
            f"Series: {r.series}",
            f"Side: {r.action.upper()} {r.side.upper()}",
            f"Strike: ${r.strike:,.2f}" if r.strike > 0 else "",
            f"Market Implied Prob: {r.market_prob:.1%}",
            f"Model Prob: {r.model_prob:.1%} (source: {r.model_source})",
            f"Edge: {r.edge_pct:+.2f}%",
            f"Cost: {r.cost_cents}c/contract",
            f"EV: {r.ev_cents:+.2f}c/contract",
            f"Kelly (quarter): {r.kelly_fraction:.2%}",
            f"Spread: {r.spread_cents}c",
            f"Volume: {r.volume}",
            f"Time to expiry: {r.minutes_to_expiry:.0f} min" if r.minutes_to_expiry else "",
            f"Composite Score: {r.composite_score:.3f}",
            f"Forecast Prob (selected side): {float(ctx.get('forecast_prob_side', 0.0)):.1%}" if ctx else "",
            f"Forecast-vs-market mispricing: {float(ctx.get('forecast_market_gap_pct', 0.0)):+.2f}%" if ctx else "",
            "",
        ])

    lines.append(
        "For EACH opportunity, provide your analysis. "
        "Focus on whether the edge is real and tradeable."
    )
    return "\n".join(l for l in lines if l is not None)


# ── Analyzer ───────────────────────────────────────────────
class KalshiAnalyzer:
    """
    Uses the LLM reasoning layer to evaluate scanner results.

    Usage:
        analyzer = KalshiAnalyzer()
        plays = analyzer.analyze(scan_results)
    """

    def __init__(self, llm_client: LLMClient = None):
        self.client = llm_client or LLMClient()

    def analyze(
        self,
        results: List[ScanResult],
        forecast_context: Optional[Dict[str, Any]] = None,
        enable_llm: bool = True,
    ) -> List[ValuePlay]:
        """
        Analyze scan results and return value plays.

        Sends all results in one prompt (batched) for efficiency.
        Falls back to rule-based analysis if LLM is in stub mode.
        """
        if not results:
            return []

        t0 = time.time()
        user_prompt = _build_user_prompt(results, forecast_context=forecast_context)

        if not enable_llm:
            elapsed = (time.time() - t0) * 1000
            return [self._rule_based(r, elapsed, forecast_context=forecast_context) for r in results]

        resp = self.client.chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format="json",
        )

        elapsed = (time.time() - t0) * 1000

        if resp.is_stub or resp.error:
            # Rule-based fallback
            return [self._rule_based(r, elapsed, forecast_context=forecast_context) for r in results]

        # Parse LLM response
        return self._parse_response(resp, results, elapsed)

    def analyze_single(self, result: ScanResult) -> ValuePlay:
        """Analyze a single scan result."""
        plays = self.analyze([result])
        return plays[0] if plays else self._rule_based(result, 0, forecast_context=None)

    def _parse_response(
        self, resp: LLMResponse, results: List[ScanResult], elapsed: float,
    ) -> List[ValuePlay]:
        """Parse LLM response into ValuePlay objects."""
        plays = []
        parsed = resp.parsed or {}

        # Handle both single-result and multi-result responses
        if isinstance(parsed, list):
            analyses = parsed
        elif "opportunities" in parsed:
            analyses = parsed["opportunities"]
        else:
            # Single analysis — apply to first result
            analyses = [parsed]

        for i, result in enumerate(results):
            if i < len(analyses):
                a = analyses[i]
            else:
                a = parsed  # Reuse single analysis for all

            value_score = float(a.get("value_score", 0))
            recommendation = a.get("recommendation", "SKIP").upper()
            if recommendation not in ("BUY", "SKIP", "WATCH"):
                recommendation = "SKIP"

            suggested = int(a.get("suggested_contracts", result.suggested_contracts))
            cost = suggested * result.cost_cents / 100.0

            plays.append(ValuePlay(
                scan=result,
                value_score=value_score,
                recommendation=recommendation,
                reasoning=a.get("reasoning", ""),
                risk_level=a.get("risk_level", "UNKNOWN"),
                confidence=float(a.get("confidence", 0)),
                suggested_size=suggested,
                max_loss=round(cost, 2),
                expected_profit=round(suggested * result.ev_cents / 100.0, 2),
                is_stub=False,
                llm_model=resp.model,
                analysis_ms=round(elapsed, 1),
            ))

        return plays

    def _rule_based(
        self,
        result: ScanResult,
        elapsed: float,
        forecast_context: Optional[Dict[str, Any]] = None,
    ) -> ValuePlay:
        """Simple rule-based analysis when LLM is unavailable."""
        # Conservative rules
        if result.edge_pct >= 10 and result.ev_cents >= 3 and result.volume >= 10:
            rec = "BUY"
            score = min(8.0, result.edge_pct / 2 + result.ev_cents)
            risk = "MODERATE"
        elif result.edge_pct >= 5 and result.ev_cents >= 1:
            rec = "WATCH"
            score = min(5.0, result.edge_pct / 3 + result.ev_cents / 2)
            risk = "HIGH"
        else:
            rec = "SKIP"
            score = max(0, result.edge_pct / 5)
            risk = "HIGH"

        if result.model_source == "market-baseline":
            rec = "SKIP"
            score = 0
            risk = "EXTREME"

        # Forecast-aware adjustment: require selected side to align with
        # end-of-hour forecast mispricing when context is present.
        if forecast_context:
            ctx_map = {
                str(x.get("ticker", "")): x
                for x in (forecast_context.get("opportunity_overrides", []) or [])
            }
            ctx = ctx_map.get(result.ticker)
            if ctx:
                gap = float(ctx.get("forecast_market_gap_pct", 0.0))
                if gap < 0:
                    rec = "SKIP"
                    score = min(score, 1.0)
                    risk = "EXTREME"
                elif gap >= 8 and rec == "WATCH":
                    rec = "BUY"
                    score = max(score, 7.0)
                    risk = "MODERATE"

        return ValuePlay(
            scan=result,
            value_score=round(score, 1),
            recommendation=rec,
            reasoning=f"Rule-based: edge={result.edge_pct:.1f}%, "
                      f"EV={result.ev_cents:.1f}c, vol={result.volume}",
            risk_level=risk,
            confidence=0.3,
            suggested_size=result.suggested_contracts if rec == "BUY" else 0,
            max_loss=round(result.suggested_contracts * result.cost_cents / 100, 2) if rec == "BUY" else 0,
            expected_profit=round(result.suggested_contracts * result.ev_cents / 100, 2) if rec == "BUY" else 0,
            is_stub=True,
            llm_model="rule-based",
            analysis_ms=round(elapsed, 1),
        )

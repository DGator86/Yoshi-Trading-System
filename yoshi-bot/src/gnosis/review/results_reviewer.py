"""LLM reviewer for experiment/backtest outputs.

Designed to be:
  - Non-fatal: review failures must not break the core pipeline.
  - JSON-first: model outputs are parsed as JSON and then rendered to Markdown.
  - Safe: only a small, whitelisted set of adaptive config overrides are accepted.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

from gnosis.execution.moltbot import (
    AIProviderConfig,
    OpenAIChatClient,
    OpenRouterClient,
    OllamaClient,
)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _deep_update(dst: dict, src: dict) -> dict:
    """Deep-merge src into dst (dicts only)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _flatten_keys(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_keys(v, prefix=key))
        else:
            out[key] = v
    return out


def _unflatten_keys(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        parts = str(k).split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def sanitize_experiment_overrides(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Whitelist + clamp adaptive overrides for experiment config.

    Accepts either nested dicts or dot-path dicts.
    Returns a nested dict suitable for deep-merge into experiment config.
    """
    if not isinstance(raw, dict) or not raw:
        return {}

    # Allow both forms.
    if any("." in str(k) for k in raw.keys()):
        flat = {str(k): raw[k] for k in raw.keys()}
    else:
        flat = _flatten_keys(raw)

    allowed: dict[str, dict[str, Any]] = {
        # Interval calibration (QuantilePredictor widens/narrows q05/q95).
        "models.predictor.sigma_scale": {"type": "float", "min": 0.5, "max": 3.0},
        # Abstention threshold scaling (get_confidence_floor uses regimes.confidence_floor_scale).
        "regimes.confidence_floor_scale": {"type": "float", "min": 0.5, "max": 1.5},
        # Predictor backend choice (bounded set).
        "models.predictor.backend": {
            "type": "enum",
            "values": {"ridge", "quantile", "bregman_fw", "gradient_boost"},
        },
        # Feature toggles (safe, deterministic).
        "models.predictor.extended_features": {"type": "bool"},
        "models.predictor.normalize": {"type": "bool"},
        # Regularization tuning (kept narrow to reduce destabilization).
        "models.predictor.l2_reg": {"type": "float", "min": 1e-4, "max": 50.0},
    }

    sanitized_flat: dict[str, Any] = {}
    for key, spec in allowed.items():
        if key not in flat:
            continue
        val = flat[key]
        typ = spec["type"]
        if typ == "float":
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            f = max(float(spec["min"]), min(float(spec["max"]), f))
            sanitized_flat[key] = float(f)
        elif typ == "bool":
            if isinstance(val, bool):
                sanitized_flat[key] = bool(val)
            elif isinstance(val, (int, float)):
                sanitized_flat[key] = bool(val)
            else:
                s = str(val).strip().lower()
                if s in {"1", "true", "yes", "y", "on"}:
                    sanitized_flat[key] = True
                elif s in {"0", "false", "no", "n", "off"}:
                    sanitized_flat[key] = False
        elif typ == "enum":
            s = str(val).strip()
            if s in spec["values"]:
                sanitized_flat[key] = s

    return _unflatten_keys(sanitized_flat)


def _render_review_markdown(review: dict[str, Any]) -> str:
    verdict = str(review.get("verdict", "unknown")).upper()
    headline = str(review.get("headline", "")).strip()
    confidence = review.get("confidence", None)
    findings = review.get("findings", []) or []
    recs = review.get("recommended_actions", []) or []
    overrides = review.get("config_overrides", {}) or {}

    lines: list[str] = []
    lines.append("# LLM Run Review")
    lines.append("")
    lines.append(f"- **Verdict**: {verdict}")
    if headline:
        lines.append(f"- **Headline**: {headline}")
    if confidence is not None:
        try:
            lines.append(f"- **Confidence**: {float(confidence):.2f}")
        except (TypeError, ValueError):
            pass

    if findings:
        lines.append("")
        lines.append("## Findings")
        for f in findings:
            if isinstance(f, dict):
                sev = str(f.get("severity", "info")).upper()
                msg = str(f.get("message", "")).strip()
                lines.append(f"- **{sev}**: {msg}")
            else:
                lines.append(f"- {str(f).strip()}")

    if recs:
        lines.append("")
        lines.append("## Recommended actions")
        for r in recs:
            if isinstance(r, dict):
                act = str(r.get("action", "")).strip()
                why = str(r.get("rationale", "")).strip()
                if act and why:
                    lines.append(f"- {act} — {why}")
                elif act:
                    lines.append(f"- {act}")
                else:
                    lines.append(f"- {json.dumps(r)}")
            else:
                lines.append(f"- {str(r).strip()}")

    if overrides:
        lines.append("")
        lines.append("## Proposed adaptive overrides (sanitized)")
        lines.append("```json")
        lines.append(json.dumps(overrides, indent=2, sort_keys=True))
        lines.append("```")

    return "\n".join(lines) + "\n"


def _build_ai_client(cfg: AIProviderConfig):
    provider = str(cfg.provider).strip().lower()
    if provider == "openai":
        return OpenAIChatClient(cfg)
    if provider in {"openrouter", "claude", "clawdbot"}:
        return OpenRouterClient(cfg)
    if provider == "ollama":
        return OllamaClient(cfg)
    return None


@dataclass
class LLMReviewConfig:
    enabled: bool = True
    config_path: str = "configs/llm_review.yaml"


class LLMResultsReviewer:
    """LLM reviewer that emits review JSON + safe config overrides."""

    def __init__(self, ai_config: AIProviderConfig):
        self.ai_config = ai_config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMResultsReviewer":
        path = Path(path)
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        # Accept either {ai: {...}} (preferred) or a bare provider dict.
        ai_raw = raw.get("ai", raw) or {}
        ai_cfg = AIProviderConfig(**ai_raw)
        return cls(ai_config=ai_cfg)

    def build_context(self, run_dir: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        run_dir = Path(run_dir)
        ctx: dict[str, Any] = {
            "built_at": _now_iso(),
            "run_dir": str(run_dir),
        }

        def _read_json(p: Path) -> dict[str, Any] | None:
            if not p.exists():
                return None
            try:
                with open(p, encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return None

        report = _read_json(run_dir / "report.json") or {}
        backtest_stats = _read_json(run_dir / "backtest" / "stats.json") or {}
        regime_snapshot = _read_json(run_dir / "regime_snapshot.json") or {}
        domain_summary = _read_json(run_dir / "domain_run_summary.json") or {}

        ctx["experiment"] = {
            "status": report.get("status"),
            "coverage_90": report.get("coverage_90"),
            "sharpness": report.get("sharpness"),
            "mae": report.get("mae"),
            "abstention_rate": report.get("abstention_rate"),
            "stability": report.get("stability", {}),
            "calibration": report.get("calibration", {}),
            "n_folds": report.get("n_folds"),
            "report_hash": report.get("report_hash"),
        }
        ctx["backtest"] = backtest_stats
        ctx["regime_snapshot"] = regime_snapshot
        ctx["domain_run"] = {
            "timeframe": domain_summary.get("timeframe"),
            "run_reason": domain_summary.get("run_reason"),
            "fetch_n": domain_summary.get("fetch_n"),
            "run_every_bars": domain_summary.get("run_every_bars"),
            "forecasting_gating": domain_summary.get("forecasting_gating"),
        }
        if extra:
            ctx["extra"] = copy.deepcopy(extra)
        return ctx

    def review(self, context: dict[str, Any]) -> dict[str, Any]:
        """Call the configured model and return parsed JSON."""
        prompt = (
            "Review this forecasting run. Focus on statistical validity, calibration, "
            "stability/flip-rate risk, backtest sanity (lookahead, overtrading), and whether "
            "forecasting module weights make sense given regime probabilities.\n\n"
            "Return ONLY JSON with keys:\n"
            "  verdict: one of ['pass','warn','fail']\n"
            "  headline: short string\n"
            "  confidence: float 0..1\n"
            "  findings: list of {severity, message}\n"
            "  recommended_actions: list of {action, rationale}\n"
            "  config_overrides: nested dict with ONLY allowed keys:\n"
            "    models.predictor.sigma_scale (0.5..3.0)\n"
            "    regimes.confidence_floor_scale (0.5..1.5)\n"
            "    models.predictor.backend in ['ridge','quantile','bregman_fw','gradient_boost']\n"
            "    models.predictor.extended_features (bool)\n"
            "    models.predictor.normalize (bool)\n"
            "    models.predictor.l2_reg (1e-4..50)\n"
        )

        client = _build_ai_client(self.ai_config)
        if client is None:
            # Offline fallback.
            return {
                "verdict": "warn",
                "headline": "LLM review disabled/unavailable (no provider client).",
                "confidence": 0.0,
                "findings": [{"severity": "warn", "message": "No AI provider configured; skipping review."}],
                "recommended_actions": [],
                "config_overrides": {},
            }

        raw = client.generate_plan(prompt=prompt, context={"run": context})
        if not isinstance(raw, dict):
            raise RuntimeError(f"LLM review returned non-dict: {type(raw)}")
        return raw

    def review_run_dir(
        self,
        run_dir: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Review a run directory and return (review, sanitized_overrides)."""
        ctx = self.build_context(run_dir=run_dir, extra=extra)
        review = self.review(ctx)
        overrides = sanitize_experiment_overrides(review.get("config_overrides") or {})
        review = copy.deepcopy(review)
        review["reviewed_at"] = _now_iso()
        review["config_overrides"] = overrides
        return review, overrides

    @staticmethod
    def write_review(run_dir: str | Path, review: dict[str, Any]) -> tuple[Path, Path]:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        json_path = run_dir / "llm_review.json"
        md_path = run_dir / "llm_review.md"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(review, handle, indent=2, sort_keys=True)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(_render_review_markdown(review))
        return json_path, md_path


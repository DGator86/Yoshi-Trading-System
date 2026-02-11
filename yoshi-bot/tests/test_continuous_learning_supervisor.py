"""Tests for always-on continuous learning supervisor."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.loop.continuous_learning import (  # noqa: E402
    ContinuousLearningConfig,
    ContinuousLearningSupervisor,
    DomainSpec,
)
from gnosis.mtf.timeframes import TF_SECONDS  # noqa: E402


class _FakeFetcher:
    def latest_common_closed_open_ms(self, symbols, timeframe, now_epoch_s):
        # Last closed bar open for timeframe at this boundary.
        return int((now_epoch_s - TF_SECONDS[timeframe]) * 1000)


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def run_domain_job(self, spec, run_reason, state_snapshot, now_ts):
        self.calls.append(
            {
                "timeframe": spec.timeframe,
                "reason": run_reason,
                "state_snapshot": state_snapshot,
                "now_ts": str(now_ts),
            }
        )
        return True, {"ok": True, "reason": run_reason}


def _cfg(tmp_path: Path, domains, bootstrap_run=False):
    return ContinuousLearningConfig(
        symbols=["BTCUSDT"],
        exchange="kraken",
        check_interval_sec=1.0,
        bootstrap_run=bootstrap_run,
        run_backtest=False,
        run_extra_improvement_loop=False,
        output_root=str(tmp_path / "out"),
        state_path=str(tmp_path / "state.json"),
        lock_path=str(tmp_path / "lock.json"),
        root_dir=str(tmp_path),
        domains=domains,
    )


def test_trigger_after_n_bars(tmp_path: Path):
    runner = _FakeRunner()
    config = _cfg(tmp_path, domains=[DomainSpec(timeframe="1m", fetch_n=2000, run_every_bars=3, enabled=True)])
    supervisor = ContinuousLearningSupervisor(
        config=config,
        candle_fetcher=_FakeFetcher(),
        job_runner=runner,
    )

    for minute in (1, 2, 3):
        now = pd.Timestamp(f"2024-01-01T00:{minute:02d}:00Z")
        supervisor.run_once(now_ts=now)

    assert len(runner.calls) == 1
    assert runner.calls[0]["reason"] == "quantized_1m_bars_3"
    dstate = supervisor.state["domains"]["1m"]
    assert dstate["bars_since_run"] == 0
    assert dstate["runs_total"] == 1


def test_bootstrap_trigger_runs_immediately(tmp_path: Path):
    runner = _FakeRunner()
    config = _cfg(
        tmp_path,
        domains=[DomainSpec(timeframe="1m", fetch_n=2000, run_every_bars=1, enabled=True)],
        bootstrap_run=True,
    )
    supervisor = ContinuousLearningSupervisor(
        config=config,
        candle_fetcher=_FakeFetcher(),
        job_runner=runner,
    )

    supervisor.run_once(now_ts=pd.Timestamp("2024-01-01T00:01:00Z"))
    assert len(runner.calls) == 1
    assert runner.calls[0]["reason"] == "bootstrap"
    assert supervisor.state["domains"]["1m"]["bootstrap_done"] is True


def test_min_wall_interval_blocks_rapid_retrigger(tmp_path: Path):
    runner = _FakeRunner()
    config = _cfg(
        tmp_path,
        domains=[DomainSpec(timeframe="1m", fetch_n=2000, run_every_bars=1, min_wall_interval_sec=120, enabled=True)],
        bootstrap_run=False,
    )
    supervisor = ContinuousLearningSupervisor(
        config=config,
        candle_fetcher=_FakeFetcher(),
        job_runner=runner,
    )

    supervisor.run_once(now_ts=pd.Timestamp("2024-01-01T00:01:00Z"))  # trigger 1
    supervisor.run_once(now_ts=pd.Timestamp("2024-01-01T00:02:00Z"))  # blocked by cooldown
    supervisor.run_once(now_ts=pd.Timestamp("2024-01-01T00:03:00Z"))  # trigger 2

    assert len(runner.calls) == 2
    assert runner.calls[0]["reason"].startswith("quantized_1m_bars_")
    assert runner.calls[1]["reason"].startswith("quantized_1m_bars_")


def test_domain_spec_back_compat_trigger_n_maps_to_fetch_n():
    spec = DomainSpec.from_dict(
        {
            "timeframe": "1m",
            "trigger_n": 2000,
            "run_every_bars": 2,
            "enabled": True,
        }
    )
    assert spec.fetch_n == 2000
    assert spec.run_every_bars == 2


def test_run_metadata_includes_forecasting_gating_weights_and_confidence(tmp_path: Path):
    runner = _FakeRunner()
    spec = DomainSpec(
        timeframe="1m",
        fetch_n=2000,
        run_every_bars=1,
        enabled=True,
        forecast_gating_inputs={
            "regime_probs": {"cascade_risk": 0.55, "range": 0.15, "trend_up": 0.30},
            "spread_bps": 15.0,
            "depth_norm": 0.25,
            "lfi": 1.7,
            "jump_probability": 0.30,
        },
    )
    config = _cfg(tmp_path, domains=[spec], bootstrap_run=False)
    supervisor = ContinuousLearningSupervisor(
        config=config,
        candle_fetcher=_FakeFetcher(),
        job_runner=runner,
    )

    supervisor.run_once(now_ts=pd.Timestamp("2024-01-01T00:01:00Z"))

    assert len(runner.calls) == 1
    snap = runner.calls[0]["state_snapshot"]
    gating = snap["forecasting_gating"]
    assert "weights" in gating and "confidence" in gating
    assert 0.0 < gating["confidence"] <= 1.0
    assert gating["weights"]["derivatives_positioning"] > gating["weights"]["technical_price_action"]

    dstate = supervisor.state["domains"]["1m"]
    assert "forecasting_gating" in dstate["last_run_meta"]
    assert dstate["last_forecasting_gating"]["top_modules"]

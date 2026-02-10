"""Always-on ML and optimization supervisor.

Semantics:
- Domain jobs trigger on relative quantized timescale closes (per timeframe).
- `fetch_n` defines the rolling timeline units used for historical fetch +
  backtest windowing (not trigger cadence).
- Trigger cadence is controlled by `run_every_bars` (default: every close).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd
import yaml

from gnosis.mtf.scheduler import due_timeframes, floor_to_second
from gnosis.mtf.timeframes import TF_SECONDS, TF_LIST


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_utc_epoch_s(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_convert("UTC").timestamp())


def _normalize_symbol_to_ccxt(symbol: str) -> str:
    s = str(symbol).upper()
    if "/" in s:
        return s
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _format_duration(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {mins}m"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


@dataclass
class DomainSpec:
    """Per-timeframe settings.

    fetch_n: rolling timeline units used for data lookback/backtest slices.
    run_every_bars: run domain cycle every N new closed bars.
    """

    timeframe: str
    fetch_n: int = 2000
    run_every_bars: int = 1
    forecast_horizon_bars: int = 1
    min_wall_interval_sec: int = 0
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DomainSpec":
        # Backward compatibility: trigger_n used to be overloaded.
        fetch_n = int(raw.get("fetch_n", raw.get("trigger_n", 2000)))
        return cls(
            timeframe=str(raw.get("timeframe", "1m")),
            fetch_n=fetch_n,
            run_every_bars=int(raw.get("run_every_bars", 1)),
            forecast_horizon_bars=int(raw.get("forecast_horizon_bars", 1)),
            min_wall_interval_sec=int(raw.get("min_wall_interval_sec", 0)),
            enabled=bool(raw.get("enabled", True)),
        )

    def recommended_run_seconds(self) -> int:
        seconds = TF_SECONDS.get(self.timeframe, 60)
        return int(max(self.run_every_bars, 1) * seconds)

    def recommended_backtest_window_seconds(self) -> int:
        seconds = TF_SECONDS.get(self.timeframe, 60)
        return int(max(self.fetch_n, 1) * seconds)


@dataclass
class ContinuousLearningConfig:
    """Config for always-on learning supervisor."""

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    exchange: str = "kraken"
    check_interval_sec: float = 5.0
    bootstrap_run: bool = True
    run_backtest: bool = True
    run_extra_improvement_loop: bool = False
    output_root: str = "reports/continuous"
    state_path: str = "reports/continuous/supervisor_state.json"
    lock_path: str = "reports/continuous/supervisor.lock"
    root_dir: str = "."

    base_experiment_config: str = "configs/experiment.yaml"
    hparams_config: str = "configs/hparams.yaml"
    backtest_config: str = "configs/backtest.yaml"

    experiment_script: str = "scripts/run_experiment.py"
    backtest_script: str = "scripts/run_backtest.py"
    improvement_script: str = "scripts/run_improvement_loop.py"

    ccxt_rate_limit_ms: int = 250
    ccxt_sandbox: bool = False

    domains: list[DomainSpec] = field(
        default_factory=lambda: [
            DomainSpec(timeframe="1m"),
            DomainSpec(timeframe="5m"),
            DomainSpec(timeframe="15m"),
            DomainSpec(timeframe="30m"),
            DomainSpec(timeframe="1h"),
            DomainSpec(timeframe="4h"),
            DomainSpec(timeframe="1d"),
        ]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ContinuousLearningConfig":
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        cfg = raw.get("continuous_learning", raw)
        domains_raw = cfg.get("domains", [])
        domains = [DomainSpec.from_dict(d) for d in domains_raw] if domains_raw else cls().domains

        return cls(
            symbols=[str(x) for x in cfg.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])],
            exchange=str(cfg.get("exchange", "kraken")),
            check_interval_sec=float(cfg.get("check_interval_sec", 5.0)),
            bootstrap_run=bool(cfg.get("bootstrap_run", True)),
            run_backtest=bool(cfg.get("run_backtest", True)),
            run_extra_improvement_loop=bool(cfg.get("run_extra_improvement_loop", False)),
            output_root=str(cfg.get("output_root", "reports/continuous")),
            state_path=str(cfg.get("state_path", "reports/continuous/supervisor_state.json")),
            lock_path=str(cfg.get("lock_path", "reports/continuous/supervisor.lock")),
            root_dir=str(cfg.get("root_dir", ".")),
            base_experiment_config=str(cfg.get("base_experiment_config", "configs/experiment.yaml")),
            hparams_config=str(cfg.get("hparams_config", "configs/hparams.yaml")),
            backtest_config=str(cfg.get("backtest_config", "configs/backtest.yaml")),
            experiment_script=str(cfg.get("experiment_script", "scripts/run_experiment.py")),
            backtest_script=str(cfg.get("backtest_script", "scripts/run_backtest.py")),
            improvement_script=str(cfg.get("improvement_script", "scripts/run_improvement_loop.py")),
            ccxt_rate_limit_ms=int(cfg.get("ccxt_rate_limit_ms", 250)),
            ccxt_sandbox=bool(cfg.get("ccxt_sandbox", False)),
            domains=domains,
        )


class CandleFetcher(Protocol):
    """Provider protocol for closed-candle monitoring."""

    def latest_common_closed_open_ms(
        self,
        symbols: list[str],
        timeframe: str,
        now_epoch_s: int,
    ) -> Optional[int]:
        """Return common closed bar open timestamp in ms."""


class DomainJobRunner(Protocol):
    """Protocol for executing a domain learning/backtest cycle."""

    def run_domain_job(
        self,
        spec: DomainSpec,
        run_reason: str,
        state_snapshot: dict[str, Any],
        now_ts: pd.Timestamp,
    ) -> tuple[bool, dict[str, Any]]:
        """Execute one domain cycle and return (success, metadata)."""


class CCXTCandleFetcher:
    """CCXT-backed closed-candle fetcher."""

    def __init__(self, exchange: str, rate_limit_ms: int = 250, sandbox: bool = False):
        from gnosis.ingest.ccxt_loader import CCXTLoader

        self.loader = CCXTLoader(exchange=exchange, rate_limit_ms=rate_limit_ms, sandbox=sandbox)
        self.exchange = exchange

    def _latest_closed_open_ms(self, symbol: str, timeframe: str, now_epoch_s: int) -> Optional[int]:
        ccxt_symbol = _normalize_symbol_to_ccxt(symbol)
        rows = self.loader._retry_request(  # pylint: disable=protected-access
            self.loader.exchange.fetch_ohlcv,
            ccxt_symbol,
            timeframe,
            None,
            5,
        )
        if not rows:
            return None
        tf_s = TF_SECONDS[timeframe]
        now_ms = int(now_epoch_s * 1000)
        candidates = []
        for row in rows:
            if not row:
                continue
            open_ms = _safe_int(row[0], default=0)
            close_ms = open_ms + tf_s * 1000
            if close_ms <= now_ms:
                candidates.append(open_ms)
        if not candidates:
            return None
        return int(max(candidates))

    def latest_common_closed_open_ms(
        self,
        symbols: list[str],
        timeframe: str,
        now_epoch_s: int,
    ) -> Optional[int]:
        closes = []
        for sym in symbols:
            ts_ms = self._latest_closed_open_ms(sym, timeframe, now_epoch_s)
            if ts_ms is None:
                return None
            closes.append(ts_ms)
        # Use min to ensure all symbols have at least this bar.
        return int(min(closes)) if closes else None


class SubprocessDomainJobRunner:
    """Run experiment/backtest/improvement as subprocess jobs."""

    def __init__(self, config: ContinuousLearningConfig):
        self.config = config
        self.root_dir = Path(config.root_dir).resolve()
        self.output_root = (self.root_dir / config.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("continuous_learning.job_runner")

    def _load_base_experiment_config(self) -> dict[str, Any]:
        cfg_path = (self.root_dir / self.config.base_experiment_config).resolve()
        with open(cfg_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

        # Inline auxiliary configs for standalone temp config.
        cfg_dir = cfg_path.parent
        for name in ("domains", "models", "regimes", "costs"):
            aux = cfg_dir / f"{name}.yaml"
            if aux.exists():
                with open(aux, encoding="utf-8") as handle:
                    cfg[name] = yaml.safe_load(handle) or {}
        return cfg

    def _domain_run_dir(self, spec: DomainSpec, now_ts: pd.Timestamp) -> Path:
        run_id = now_ts.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / spec.timeframe / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_domain_experiment_config(
        self,
        spec: DomainSpec,
        run_dir: Path,
    ) -> dict[str, Any]:
        cfg = self._load_base_experiment_config()
        cfg.setdefault("dataset", {})
        cfg["dataset"]["mode"] = "ohlcv"
        live = cfg["dataset"].setdefault("live", {})
        live["exchange"] = self.config.exchange
        live["timeframe"] = spec.timeframe

        # Ensure enough lookback for fetch_n timeline units.
        tf_days = (TF_SECONDS[spec.timeframe] * spec.fetch_n) / 86400.0
        min_days = max(30, int(math.ceil(tf_days * 1.3)))
        live["days"] = max(int(live.get("days", 0)), min_days)

        cfg.setdefault("forecast", {})
        cfg["forecast"]["horizon_bars"] = int(spec.forecast_horizon_bars)

        # Embed supervisor domain metadata for traceability.
        cfg.setdefault("continuous_learning", {})
        cfg["continuous_learning"]["timeframe"] = spec.timeframe
        cfg["continuous_learning"]["fetch_n"] = int(spec.fetch_n)
        cfg["continuous_learning"]["run_every_bars"] = int(spec.run_every_bars)

        cfg.setdefault("artifacts", {})
        cfg["artifacts"]["out_dir"] = str(run_dir)
        return cfg

    @staticmethod
    def _build_backtest_slice(
        predictions_path: Path,
        spec: DomainSpec,
        out_path: Path,
    ) -> tuple[Path, int]:
        """Slice predictions to last fetch_n timeline units per symbol."""
        if not predictions_path.exists():
            return predictions_path, 0

        df = pd.read_parquet(predictions_path)
        if df.empty:
            return predictions_path, 0

        n = max(int(spec.fetch_n), 1)
        if "symbol" in df.columns:
            sort_cols = [c for c in ("symbol", "bar_idx", "timestamp_end") if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            sliced = df.groupby("symbol", group_keys=False).tail(n).reset_index(drop=True)
        else:
            sort_cols = [c for c in ("bar_idx", "timestamp_end") if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            sliced = df.tail(n).reset_index(drop=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sliced.to_parquet(out_path, index=False)
        return out_path, len(sliced)

    def _run_cmd(self, cmd: list[str], log_path: Path) -> tuple[bool, int]:
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n\n[{_now_iso()}] RUN: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                stdout=logf,
                stderr=logf,
                text=True,
                check=False,
            )
            return proc.returncode == 0, proc.returncode

    def run_domain_job(
        self,
        spec: DomainSpec,
        run_reason: str,
        state_snapshot: dict[str, Any],
        now_ts: pd.Timestamp,
    ) -> tuple[bool, dict[str, Any]]:
        run_dir = self._domain_run_dir(spec, now_ts=now_ts)
        runtime_cfg_dir = self.output_root / "runtime_configs"
        runtime_cfg_dir.mkdir(parents=True, exist_ok=True)
        exp_cfg_path = runtime_cfg_dir / f"experiment_{spec.timeframe}.yaml"
        exp_cfg = self._build_domain_experiment_config(spec=spec, run_dir=run_dir)
        with open(exp_cfg_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(exp_cfg, handle, sort_keys=False)

        log_path = run_dir / "job.log"
        py = sys.executable
        experiment_script = str((self.root_dir / self.config.experiment_script).resolve())
        backtest_script = str((self.root_dir / self.config.backtest_script).resolve())
        improvement_script = str((self.root_dir / self.config.improvement_script).resolve())
        hparams_path = str((self.root_dir / self.config.hparams_config).resolve())
        backtest_cfg_path = str((self.root_dir / self.config.backtest_config).resolve())

        metadata: dict[str, Any] = {
            "timeframe": spec.timeframe,
            "run_reason": run_reason,
            "run_dir": str(run_dir),
            "state_snapshot": state_snapshot,
            "fetch_n": int(spec.fetch_n),
            "run_every_bars": int(spec.run_every_bars),
            "steps": [],
        }

        exp_cmd = [py, experiment_script, "--config", str(exp_cfg_path)]
        if Path(hparams_path).exists():
            exp_cmd += ["--hparams", hparams_path]
        ok, rc = self._run_cmd(exp_cmd, log_path=log_path)
        metadata["steps"].append({"name": "experiment", "return_code": rc, "ok": ok})
        if not ok:
            return False, metadata

        predictions_path = run_dir / "predictions.parquet"
        if self.config.run_backtest and predictions_path.exists():
            backtest_out = run_dir / "backtest"
            backtest_out.mkdir(parents=True, exist_ok=True)
            backtest_input = backtest_out / "predictions_backtest_window.parquet"
            backtest_input, n_rows = self._build_backtest_slice(
                predictions_path=predictions_path,
                spec=spec,
                out_path=backtest_input,
            )
            bt_cmd = [
                py,
                backtest_script,
                "--predictions",
                str(backtest_input),
                "--config",
                backtest_cfg_path,
                "--out",
                str(backtest_out),
            ]
            ok_bt, rc_bt = self._run_cmd(bt_cmd, log_path=log_path)
            metadata["steps"].append(
                {
                    "name": "backtest",
                    "return_code": rc_bt,
                    "ok": ok_bt,
                    "input_rows": int(n_rows),
                    "input_path": str(backtest_input),
                }
            )
            if not ok_bt:
                return False, metadata

        if self.config.run_extra_improvement_loop:
            data_parquet = Path(exp_cfg.get("dataset", {}).get("parquet_dir", "data/large_history")) / "prints.parquet"
            imp_out = run_dir / "improvement"
            imp_cmd = [
                py,
                improvement_script,
                "--data",
                str(data_parquet),
                "--output",
                str(imp_out),
            ]
            ok_imp, rc_imp = self._run_cmd(imp_cmd, log_path=log_path)
            metadata["steps"].append({"name": "improvement_loop", "return_code": rc_imp, "ok": ok_imp})
            if not ok_imp:
                return False, metadata

        summary_path = run_dir / "domain_run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return True, metadata


class ContinuousLearningSupervisor:
    """Always-on supervisor for domain-triggered learning jobs."""

    def __init__(
        self,
        config: ContinuousLearningConfig,
        candle_fetcher: Optional[CandleFetcher] = None,
        job_runner: Optional[DomainJobRunner] = None,
    ):
        self.config = config
        self.logger = logging.getLogger("continuous_learning.supervisor")
        self.domains = {d.timeframe: d for d in config.domains if d.enabled}
        self.enabled_timeframes = [tf for tf in TF_LIST if tf in self.domains]

        self.state_path = Path(config.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(config.lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.fetcher = candle_fetcher or CCXTCandleFetcher(
            exchange=config.exchange,
            rate_limit_ms=config.ccxt_rate_limit_ms,
            sandbox=config.ccxt_sandbox,
        )
        self.runner = job_runner or SubprocessDomainJobRunner(config=config)

        self.state = self._load_state()
        self._log_domain_plan()

    def _log_domain_plan(self) -> None:
        for tf, spec in self.domains.items():
            cadence_s = spec.recommended_run_seconds()
            backtest_s = spec.recommended_backtest_window_seconds()
            self.logger.info(
                "Domain %s run_every_bars=%d -> cadence %s | fetch_n=%d -> backtest window %s",
                tf,
                spec.run_every_bars,
                _format_duration(cadence_s),
                spec.fetch_n,
                _format_duration(backtest_s),
            )

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            state = {"domains": {}}

        state.setdefault("domains", {})
        for tf in self.domains:
            d = state["domains"].setdefault(tf, {})
            d.setdefault("last_closed_open_ms", None)
            d.setdefault("bars_total_seen", 0)
            d.setdefault("bars_since_run", d.get("bars_since_trigger", 0))
            d.setdefault("bars_since_trigger", d.get("bars_since_run", 0))
            d.setdefault("last_run_epoch_s", d.get("last_trigger_epoch_s", 0))
            d.setdefault("last_run_reason", d.get("last_trigger_reason", ""))
            d.setdefault("last_run_ok", d.get("last_trigger_ok", None))
            d.setdefault("last_trigger_epoch_s", d.get("last_run_epoch_s", 0))
            d.setdefault("last_trigger_reason", d.get("last_run_reason", ""))
            d.setdefault("last_trigger_ok", d.get("last_run_ok", None))
            d.setdefault("runs_total", 0)
            d.setdefault("bootstrap_done", False)
        return state

    def _save_state(self) -> None:
        payload = copy.deepcopy(self.state)
        payload["updated_at"] = _now_iso()
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _domain_state(self, timeframe: str) -> dict[str, Any]:
        return self.state["domains"][timeframe]

    def process_closed_bar(
        self,
        timeframe: str,
        closed_open_ms: int,
        now_epoch_s: int,
    ) -> tuple[bool, str]:
        """Process one new closed bar for a timeframe.

        Returns:
            (triggered, reason)
        """
        spec = self.domains[timeframe]
        dstate = self._domain_state(timeframe)
        last_ms = dstate.get("last_closed_open_ms")

        if last_ms is None:
            delta = 1
        elif int(closed_open_ms) <= int(last_ms):
            return False, "no_new_bar"
        else:
            step_ms = TF_SECONDS[timeframe] * 1000
            delta = max(1, (int(closed_open_ms) - int(last_ms)) // step_ms)

        dstate["last_closed_open_ms"] = int(closed_open_ms)
        dstate["bars_total_seen"] = int(dstate.get("bars_total_seen", 0)) + int(delta)
        dstate["bars_since_run"] = int(dstate.get("bars_since_run", 0)) + int(delta)
        dstate["bars_since_trigger"] = int(dstate["bars_since_run"])

        reason = ""
        should_trigger = False
        if self.config.bootstrap_run and not bool(dstate.get("bootstrap_done", False)):
            should_trigger = True
            reason = "bootstrap"
        elif int(dstate["bars_since_run"]) >= max(int(spec.run_every_bars), 1):
            min_wait = int(spec.min_wall_interval_sec)
            last_run = int(dstate.get("last_run_epoch_s", 0))
            if min_wait <= 0 or now_epoch_s - last_run >= min_wait:
                should_trigger = True
                reason = f"quantized_{timeframe}_bars_{dstate['bars_since_run']}"
            else:
                reason = "cooldown_wait"
        else:
            reason = "collecting_bars"

        if not should_trigger:
            return False, reason

        snapshot = copy.deepcopy(dstate)
        ok, meta = self.runner.run_domain_job(
            spec=spec,
            run_reason=reason,
            state_snapshot=snapshot,
            now_ts=pd.Timestamp(now_epoch_s, unit="s", tz="UTC"),
        )
        dstate["last_run_epoch_s"] = int(now_epoch_s)
        dstate["last_run_reason"] = reason
        dstate["last_run_ok"] = bool(ok)
        # Backward-compatible mirror keys.
        dstate["last_trigger_epoch_s"] = int(now_epoch_s)
        dstate["last_trigger_reason"] = reason
        dstate["last_trigger_ok"] = bool(ok)
        dstate["runs_total"] = int(dstate.get("runs_total", 0)) + 1
        dstate["bootstrap_done"] = True
        dstate["last_run_meta"] = meta
        if ok:
            dstate["bars_since_run"] = 0
            dstate["bars_since_trigger"] = 0
        return True, reason

    def _write_lock(self) -> None:
        with open(self.lock_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": _now_iso()}))

    def _remove_lock(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()

    def run_once(self, now_ts: Optional[pd.Timestamp] = None) -> dict[str, Any]:
        now = floor_to_second(now_ts or pd.Timestamp.now(tz="UTC"))
        now_epoch_s = _to_utc_epoch_s(now)
        due_nominal = due_timeframes(now, self.enabled_timeframes)

        triggered: list[dict[str, Any]] = []
        # Probe all enabled timeframes each tick to avoid missing closes when
        # check_interval_sec is not phase-aligned with timeframe boundaries.
        for tf in self.enabled_timeframes:
            spec = self.domains.get(tf)
            if spec is None or not spec.enabled:
                continue
            closed_open_ms = self.fetcher.latest_common_closed_open_ms(
                symbols=self.config.symbols,
                timeframe=tf,
                now_epoch_s=now_epoch_s,
            )
            if closed_open_ms is None:
                continue

            did_trigger, reason = self.process_closed_bar(
                timeframe=tf,
                closed_open_ms=closed_open_ms,
                now_epoch_s=now_epoch_s,
            )
            if did_trigger:
                triggered.append({"timeframe": tf, "reason": reason})

        self._save_state()
        return {
            "now": str(now),
            "due_nominal": due_nominal,
            "checked_timeframes": list(self.enabled_timeframes),
            "triggered": triggered,
        }

    def run_forever(self) -> None:
        if self.lock_path.exists():
            raise RuntimeError(f"Lock already exists: {self.lock_path}")

        self._write_lock()
        self.logger.info(
            "Continuous learning supervisor started. exchange=%s interval=%.2fs",
            self.config.exchange,
            self.config.check_interval_sec,
        )
        try:
            while True:
                started = time.time()
                try:
                    tick = self.run_once()
                    if tick["triggered"]:
                        self.logger.info("Triggered domain runs: %s", tick["triggered"])
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.exception("Supervisor tick failed: %s", exc)

                elapsed = time.time() - started
                sleep_s = max(float(self.config.check_interval_sec) - elapsed, 0.0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self._remove_lock()

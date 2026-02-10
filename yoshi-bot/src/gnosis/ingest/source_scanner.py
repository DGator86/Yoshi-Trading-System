"""Continuous crypto source scanner.

Scans configured exchange sources on a fixed interval, normalizes the market
snapshots, and persists both per-source snapshots and cross-source consensus.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_ms(ts: Any) -> Optional[int]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        try:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return None
    return None


def _normalize_symbol_to_ccxt(symbol: str) -> str:
    s = str(symbol).upper()
    if "/" in s:
        return s
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _book_summary(order_book: dict[str, Any], depth_levels: int = 10) -> dict[str, float]:
    bids = order_book.get("bids", []) or []
    asks = order_book.get("asks", []) or []
    bid = _safe_float(bids[0][0]) if bids else 0.0
    ask = _safe_float(asks[0][0]) if asks else 0.0
    spread_bps = 0.0
    if bid > 0.0 and ask > 0.0:
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / max(mid, 1e-12) * 10_000.0

    depth_bid_notional = 0.0
    depth_ask_notional = 0.0
    for price, size in bids[: max(depth_levels, 1)]:
        depth_bid_notional += _safe_float(price) * _safe_float(size)
    for price, size in asks[: max(depth_levels, 1)]:
        depth_ask_notional += _safe_float(price) * _safe_float(size)

    return {
        "best_bid": bid,
        "best_ask": ask,
        "spread_bps": spread_bps,
        "depth_bid_notional": depth_bid_notional,
        "depth_ask_notional": depth_ask_notional,
    }


@dataclass
class SourceSpec:
    """Single source scanner specification."""

    exchange: str
    symbols: list[str] = field(default_factory=list)
    ccxt_symbol_overrides: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    rate_limit_ms: int = 250
    sandbox: bool = False
    tag: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(
            exchange=str(raw.get("exchange", "binance")),
            symbols=[str(x) for x in (raw.get("symbols") or [])],
            ccxt_symbol_overrides={str(k): str(v) for k, v in (raw.get("ccxt_symbol_overrides") or {}).items()},
            enabled=bool(raw.get("enabled", True)),
            rate_limit_ms=int(raw.get("rate_limit_ms", 250)),
            sandbox=bool(raw.get("sandbox", False)),
            tag=str(raw.get("tag", "")),
        )


@dataclass
class CryptoSourceScannerConfig:
    """Config for the continuous scanner loop."""

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    sources: list[SourceSpec] = field(default_factory=lambda: [SourceSpec(exchange="binance")])
    poll_interval_sec: float = 5.0
    fail_sleep_sec: float = 2.0
    output_dir: str = "data/live/crypto_sources"
    enable_orderbook: bool = True
    orderbook_limit: int = 50
    orderbook_depth_levels: int = 10
    enable_funding: bool = True
    enable_open_interest: bool = True
    log_every_n_cycles: int = 12

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CryptoSourceScannerConfig":
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        scanner = raw.get("scanner", raw)
        sources = [SourceSpec.from_dict(s) for s in scanner.get("sources", [])]
        if not sources:
            sources = [SourceSpec(exchange="binance")]
        return cls(
            symbols=[str(x) for x in scanner.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])],
            sources=sources,
            poll_interval_sec=float(scanner.get("poll_interval_sec", 5.0)),
            fail_sleep_sec=float(scanner.get("fail_sleep_sec", 2.0)),
            output_dir=str(scanner.get("output_dir", "data/live/crypto_sources")),
            enable_orderbook=bool(scanner.get("enable_orderbook", True)),
            orderbook_limit=int(scanner.get("orderbook_limit", 50)),
            orderbook_depth_levels=int(scanner.get("orderbook_depth_levels", 10)),
            enable_funding=bool(scanner.get("enable_funding", True)),
            enable_open_interest=bool(scanner.get("enable_open_interest", True)),
            log_every_n_cycles=int(scanner.get("log_every_n_cycles", 12)),
        )


class CryptoSourceScanner:
    """Continuously poll configured exchanges for market snapshots."""

    def __init__(
        self,
        config: CryptoSourceScannerConfig,
        loader_factory: Optional[Callable[[SourceSpec], Any]] = None,
    ):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("crypto_source_scanner")
        self._loader_factory = loader_factory or self._default_loader_factory
        self._loaders: dict[str, Any] = {}
        self._cycle = 0

        self._init_loaders()

    @staticmethod
    def _default_loader_factory(source: SourceSpec):
        # Local import to keep optional CCXT dependency lazy.
        from gnosis.ingest.ccxt_loader import CCXTLoader

        return CCXTLoader(
            exchange=source.exchange,
            rate_limit_ms=source.rate_limit_ms,
            sandbox=source.sandbox,
        )

    def _init_loaders(self) -> None:
        for source in self.config.sources:
            if not source.enabled:
                continue
            key = source.exchange.lower()
            if key in self._loaders:
                continue
            self._loaders[key] = self._loader_factory(source)
            self.logger.info("Initialized scanner loader for %s", key)

    def _source_symbols(self, source: SourceSpec) -> list[str]:
        return source.symbols or self.config.symbols

    def _to_ccxt_symbol(self, source: SourceSpec, symbol: str) -> str:
        if symbol in source.ccxt_symbol_overrides:
            return source.ccxt_symbol_overrides[symbol]
        return _normalize_symbol_to_ccxt(symbol)

    @staticmethod
    def _call_loader_method(loader: Any, method_name: str, *args, **kwargs):
        # Preferred: explicit loader methods.
        if hasattr(loader, method_name):
            return getattr(loader, method_name)(*args, **kwargs)

        # Fallback: CCXT exchange methods through retry wrapper.
        if hasattr(loader, "exchange") and hasattr(loader, "_retry_request"):
            ex_method = getattr(loader.exchange, method_name, None)
            if ex_method is not None:
                return loader._retry_request(ex_method, *args, **kwargs)
        raise AttributeError(f"Loader missing method '{method_name}'")

    def _scan_one(self, source: SourceSpec, symbol: str) -> dict[str, Any]:
        ex_key = source.exchange.lower()
        loader = self._loaders[ex_key]
        ccxt_symbol = self._to_ccxt_symbol(source, symbol)
        scanned_at = _now_iso()

        ticker = self._call_loader_method(loader, "fetch_ticker", ccxt_symbol)
        bid = _safe_float(ticker.get("bid"))
        ask = _safe_float(ticker.get("ask"))
        last = _safe_float(ticker.get("last", ticker.get("close")))
        mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else max(last, 0.0)
        spread_bps = ((ask - bid) / max(mid, 1e-12) * 10_000.0) if bid > 0.0 and ask > 0.0 else 0.0

        snapshot: dict[str, Any] = {
            "scanned_at": scanned_at,
            "exchange": ex_key,
            "source_tag": source.tag or ex_key,
            "symbol": symbol,
            "ccxt_symbol": ccxt_symbol,
            "timestamp_exchange_ms": _to_ms(ticker.get("timestamp")),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "last": last,
            "spread_bps": spread_bps,
            "base_volume": _safe_float(ticker.get("baseVolume")),
            "quote_volume": _safe_float(ticker.get("quoteVolume")),
            "high": _safe_float(ticker.get("high")),
            "low": _safe_float(ticker.get("low")),
            "change_pct": _safe_float(ticker.get("percentage")),
        }

        if self.config.enable_orderbook:
            try:
                ob = self._call_loader_method(
                    loader,
                    "fetch_order_book",
                    ccxt_symbol,
                    self.config.orderbook_limit,
                )
                snapshot.update(
                    {
                        "orderbook_timestamp_ms": _to_ms(ob.get("timestamp")),
                        **_book_summary(ob, depth_levels=self.config.orderbook_depth_levels),
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                snapshot["orderbook_error"] = str(exc)

        if self.config.enable_funding:
            try:
                fr = self._call_loader_method(loader, "fetch_funding_rate", ccxt_symbol)
                snapshot["funding_rate"] = _safe_float(fr.get("fundingRate", fr.get("funding")))
                snapshot["next_funding_time_ms"] = _to_ms(
                    fr.get("nextFundingTimestamp", fr.get("nextFundingTime"))
                )
            except Exception as exc:  # pylint: disable=broad-except
                snapshot["funding_error"] = str(exc)

        if self.config.enable_open_interest:
            try:
                oi = self._call_loader_method(loader, "fetch_open_interest", ccxt_symbol)
                snapshot["open_interest"] = _safe_float(
                    oi.get("openInterest", oi.get("open_interest", oi.get("oi")))
                )
            except Exception as exc:  # pylint: disable=broad-except
                snapshot["open_interest_error"] = str(exc)

        return snapshot

    @staticmethod
    def _build_consensus(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for snap in snapshots:
            by_symbol.setdefault(str(snap["symbol"]), []).append(snap)

        consensus: dict[str, Any] = {}
        for symbol, snaps in by_symbol.items():
            vals = [float(s["last"]) for s in snaps if _safe_float(s.get("last")) > 0.0]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            med = vals_sorted[n // 2] if n % 2 == 1 else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
            mn = vals_sorted[0]
            mx = vals_sorted[-1]
            consensus[symbol] = {
                "median_last": med,
                "min_last": mn,
                "max_last": mx,
                "dispersion_bps": ((mx - mn) / max(med, 1e-12) * 10_000.0),
                "n_sources": len(vals_sorted),
                "sources": [str(s["exchange"]) for s in snaps],
            }
        return consensus

    def scan_once(self) -> dict[str, Any]:
        """Perform one full scan across all configured sources/symbols."""
        snapshots: list[dict[str, Any]] = []
        for source in self.config.sources:
            if not source.enabled:
                continue
            for symbol in self._source_symbols(source):
                try:
                    snapshots.append(self._scan_one(source=source, symbol=symbol))
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.exception(
                        "Scan failed for %s %s: %s",
                        source.exchange,
                        symbol,
                        exc,
                    )
                    snapshots.append(
                        {
                            "scanned_at": _now_iso(),
                            "exchange": source.exchange.lower(),
                            "source_tag": source.tag or source.exchange.lower(),
                            "symbol": symbol,
                            "ccxt_symbol": self._to_ccxt_symbol(source, symbol),
                            "scan_error": str(exc),
                        }
                    )

        result = {
            "scanned_at": _now_iso(),
            "snapshots": snapshots,
            "consensus": self._build_consensus(snapshots),
        }
        return result

    def persist_scan(self, scan_result: dict[str, Any]) -> None:
        """Persist latest + append-only snapshot log."""
        scanned_at = str(scan_result.get("scanned_at", _now_iso()))
        day = scanned_at[:10].replace("-", "")
        latest_path = self.output_dir / "latest.json"
        consensus_path = self.output_dir / "consensus_latest.json"
        snapshots_path = self.output_dir / f"snapshots_{day}.ndjson"

        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(scan_result, handle, indent=2, sort_keys=True)

        with open(consensus_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scanned_at": scanned_at,
                    "consensus": scan_result.get("consensus", {}),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        with open(snapshots_path, "a", encoding="utf-8") as handle:
            for row in scan_result.get("snapshots", []):
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def run_once(self) -> dict[str, Any]:
        """Scan and persist one cycle."""
        result = self.scan_once()
        self.persist_scan(result)
        return result

    def run_forever(self) -> None:
        """Run continuous scan loop."""
        self.logger.info(
            "Starting continuous crypto source scan: interval=%.2fs output=%s",
            self.config.poll_interval_sec,
            self.output_dir,
        )
        while True:
            started = time.time()
            try:
                result = self.run_once()
                self._cycle += 1
                if self._cycle % max(self.config.log_every_n_cycles, 1) == 0:
                    self.logger.info(
                        "Scan cycle %d complete: snapshots=%d symbols=%d",
                        self._cycle,
                        len(result.get("snapshots", [])),
                        len(result.get("consensus", {})),
                    )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Scan cycle failed: %s", exc)
                time.sleep(max(self.config.fail_sleep_sec, 0.1))
                continue

            elapsed = time.time() - started
            sleep_s = max(self.config.poll_interval_sec - elapsed, 0.0)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

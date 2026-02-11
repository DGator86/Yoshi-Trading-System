#!/usr/bin/env python3
"""Run regime-first crypto walk-forward and emit ledger/trade artifacts.

Example:
  python3 yoshi-bot/scripts/run_regime_first_walkforward.py \
    --config yoshi-bot/configs/crypto_regime_first.yaml \
    --out-dir runs/regime_first \
    --days 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    yoshi_root = here.parents[1]
    src = yoshi_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _ccxt_symbol(sym: str) -> str:
    # "BTCUSDT" -> "BTC/USDT"
    s = sym.upper().strip()
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"
    return s


def main() -> int:
    _add_repo_paths()

    from gnosis.ingest.ccxt_loader import CCXTLoader
    from gnosis.regime_first.config import load_regime_first_config
    from gnosis.regime_first.ledger import build_regime_ledger
    from gnosis.regime_first.walkforward import run_regime_first_walkforward

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to crypto_regime_first.yaml")
    ap.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    ap.add_argument("--days", type=int, default=30, help="Fetch last N days of 1m bars")
    ap.add_argument("--limit", type=int, default=None, help="Optional per-request OHLCV limit override")
    args = ap.parse_args()

    cfg = load_regime_first_config(args.config)
    sys_cfg = cfg.system
    venue = str(sys_cfg.get("venue", "binance"))
    symbols = list(sys_cfg.get("symbols", []))
    if not symbols:
        raise SystemExit("config.system.symbols is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = CCXTLoader(exchange=venue)

    bars_all = []
    for sym in symbols:
        ccxt_sym = _ccxt_symbol(sym)
        df = loader.fetch_ohlcv(ccxt_sym, timeframe="1m", days=int(args.days), limit=args.limit)
        if df.empty:
            continue
        # Ensure schema
        df = df.rename(columns={"symbol": "symbol"})
        df["symbol"] = sym  # normalize to BTCUSDT-like
        bars_all.append(df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])

    if not bars_all:
        raise SystemExit("No bars fetched (all symbols empty).")

    bars_1m = pd.concat(bars_all, ignore_index=True)
    bars_1m["timestamp"] = pd.to_datetime(bars_1m["timestamp"], utc=True)
    bars_1m = bars_1m.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"])

    # Build ledger (includes regimes + overlays + multihorizon p_final).
    ledger = build_regime_ledger(bars_1m, cfg_raw=cfg.raw)

    # Persist the raw 1m bars used (for reproducibility).
    try:
        bars_1m.to_parquet(out_dir / "bars_1m.parquet", index=False)
    except Exception:
        bars_1m.to_csv(out_dir / "bars_1m.csv", index=False)

    # Run walk-forward.
    summary = run_regime_first_walkforward(ledger, cfg_raw=cfg.raw, out_dir=out_dir)

    # Persist config snapshot.
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.raw, f, indent=2, sort_keys=True)

    print(f"Wrote artifacts to: {out_dir}")
    print(f"Blocks: {summary.get('n_blocks')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


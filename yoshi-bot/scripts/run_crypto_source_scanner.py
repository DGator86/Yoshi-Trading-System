#!/usr/bin/env python3
"""Run the continuous crypto source scanner."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Load local secrets from yoshi-bot/.env if present.
try:
    from dotenv import load_dotenv  # type: ignore

    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

# Add yoshi-bot/src to path so `import gnosis` works.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.ingest.source_scanner import (  # noqa: E402
    CryptoSourceScanner,
    CryptoSourceScannerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous crypto source scanner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/crypto_source_scanner.yaml",
        help="Path to scanner YAML config",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one scan cycle and exit",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Override poll_interval_sec from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_crypto_source_scanner")

    cfg = CryptoSourceScannerConfig.from_yaml(args.config)
    if args.interval is not None:
        cfg.poll_interval_sec = float(args.interval)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)

    scanner = CryptoSourceScanner(config=cfg)

    if args.once:
        result = scanner.run_once()
        logger.info(
            "Single scan complete. snapshots=%d consensus_symbols=%d output=%s",
            len(result.get("snapshots", [])),
            len(result.get("consensus", {})),
            cfg.output_dir,
        )
        return 0

    scanner.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

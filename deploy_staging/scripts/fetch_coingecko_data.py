#!/usr/bin/env python3
"""Example script demonstrating CoinGecko API integration.

Usage:
    # Use stub data (default):
    python scripts/fetch_coingecko_data.py --symbols BTCUSDT ETHUSDT --stub

    # Use CoinGecko API:
    export COINGECKO_API_KEY=your-api-key-here
    python scripts/fetch_coingecko_data.py --symbols BTCUSDT ETHUSDT --days 7
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.ingest import load_or_create_prints, create_data_manifest


def main():
    parser = argparse.ArgumentParser(description="Fetch cryptocurrency market data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Trading symbols to fetch (default: BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Output directory for parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub data instead of CoinGecko API",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="CoinGecko API key (overrides COINGECKO_API_KEY env var)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh data even if cached parquet exists",
    )

    args = parser.parse_args()

    # Remove cached file if force refresh
    if args.force_refresh:
        parquet_path = args.output_dir / "prints.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            print(f"Removed cached file: {parquet_path}")

    use_coingecko = not args.stub
    api_key = args.api_key or os.getenv("COINGECKO_API_KEY")

    print(f"Fetching data for symbols: {args.symbols}")
    print(f"Data source: {'CoinGecko API' if use_coingecko else 'Stub data'}")
    
    if use_coingecko:
        if not api_key:
            print("Error: COINGECKO_API_KEY environment variable or --api-key argument required for CoinGecko data source", file=sys.stderr)
            print("Either set the environment variable or use --stub for synthetic data", file=sys.stderr)
            sys.exit(1)
        print("API key: configured")

    try:
        # Load or create prints
        df = load_or_create_prints(
            parquet_dir=args.output_dir,
            symbols=args.symbols,
            use_coingecko=use_coingecko,
            coingecko_api_key=api_key,
        )

        print(f"\nSuccessfully loaded {len(df)} prints")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Symbols: {df['symbol'].unique().tolist()}")
        
        # Show sample data
        print("\nSample data:")
        print(df.head(10))

        # Create manifest
        manifest_path = Path("data/manifests/data_manifest.json")
        manifest = create_data_manifest(df, manifest_path)
        print(f"\nCreated manifest: {manifest_path}")
        print(f"Data hash: {manifest['data_hash']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Script to start the Yoshi-Bot Trading Core API server.

This API provides the interface for ClawdBot to query system status,
propose trades, and manage positions.
"""
import argparse
import sys
from pathlib import Path

import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run the FastAPI application using uvicorn."""
    parser = argparse.ArgumentParser(
        description="Start Yoshi-Bot Trading Core API"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )

    args = parser.parse_args()

    print(f"Starting Yoshi-Bot Trading Core on http://{args.host}:{args.port}")
    uvicorn.run(
        "src.gnosis.execution.trading_core:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

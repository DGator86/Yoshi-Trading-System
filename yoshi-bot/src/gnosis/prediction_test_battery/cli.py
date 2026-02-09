from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from gnosis.prediction_test_battery.battery import run_all_tests
from gnosis.prediction_test_battery.context import BatteryContext
from gnosis.prediction_test_battery.data import (
    generate_synthetic_data,
    load_candles,
    load_features,
    load_predictions,
)

app = typer.Typer(help="Prediction Test Battery CLI")


@app.command()
def run(
    suite: str = typer.Option("full", help="Suite to run: full, 0, A, B, C, D, E, F, G"),
    artifact_path: Optional[Path] = typer.Option(None, help="Path to predictions CSV"),
    candles_path: Optional[Path] = typer.Option(None, help="Path to candles CSV"),
    features_path: Optional[Path] = typer.Option(None, help="Path to features CSV"),
    report_dir: Path = typer.Option(Path("reports"), help="Report output directory"),
    synthetic: bool = typer.Option(False, help="Use synthetic data"),
) -> None:
    if synthetic:
        candles, artifact = generate_synthetic_data()
    else:
        if artifact_path is None:
            raise typer.BadParameter("artifact_path is required unless using --synthetic")
        artifact = load_predictions(artifact_path)
        candles = load_candles(candles_path) if candles_path else None
        features = load_features(features_path)
        if features is not None:
            artifact.features = features
    context = BatteryContext(artifact=artifact, candles=candles)
    report = run_all_tests(context, suite=suite, report_dir=report_dir)
    typer.echo(f"Report written to {report_dir}/reports_{report.run_id}")


if __name__ == "__main__":
    app()

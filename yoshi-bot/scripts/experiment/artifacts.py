"""Artifact saving utilities for experiments."""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _drop_future_return_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any future_return columns to prevent merge conflicts.

    This prevents pandas from creating future_return_x / future_return_y
    during merges when both DataFrames have future_return columns.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with future_return* columns removed.
    """
    cols = [c for c in df.columns if c.startswith("future_return")]
    if cols:
        return df.drop(columns=cols, errors="ignore")
    return df


class ArtifactSaver:
    """Handles saving of all experiment artifacts to output directory."""

    # Required columns for predictions.parquet
    REQUIRED_PREDICTION_COLS = [
        "symbol", "bar_idx", "timestamp_end", "close",
        "q05", "q50", "q95", "x_hat", "sigma_hat", "fold",
        "K_label", "K_pmax", "K_entropy",
        "P_label", "P_pmax", "P_entropy",
        "C_label", "C_pmax", "C_entropy",
        "O_label", "O_pmax", "O_entropy",
        "F_label", "F_pmax", "F_entropy",
        "G_label", "G_pmax", "G_entropy",
        "S_label", "S_pmax", "S_entropy",
        "S_pmax_calibrated", "regime_entropy", "abstain",
    ]

    def __init__(self, out_dir: Path):
        """Initialize artifact saver.

        Args:
            out_dir: Output directory for all artifacts.
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        horizon_bars: int = 10,
    ) -> None:
        """Save predictions with y_true attached.

        Args:
            predictions_df: DataFrame with model predictions.
            features_df: Optional features DataFrame with future_return.
            horizon_bars: Forecast horizon for computing future_return.
        """
        if predictions_df.empty:
            self._save_minimal_predictions()
            return

        # Ensure all required columns exist with defaults
        preds = self._ensure_required_columns(predictions_df.copy())

        # Attach y_true (future_return) if possible
        preds = self._attach_future_return(preds, features_df, horizon_bars)

        # Save to parquet
        preds.to_parquet(self.out_dir / "predictions.parquet", index=False)

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist with sensible defaults."""
        for col in self.REQUIRED_PREDICTION_COLS:
            if col not in df.columns:
                if col == "abstain":
                    df[col] = False
                elif col.endswith("_label"):
                    df[col] = "UNKNOWN"
                elif col.endswith("_calibrated"):
                    df[col] = 0.5
                else:
                    df[col] = 0.0

        # Select columns in required order
        available_cols = [c for c in self.REQUIRED_PREDICTION_COLS if c in df.columns]
        # Also include future_return if present
        if "future_return" in df.columns and "future_return" not in available_cols:
            available_cols.append("future_return")
        return df[available_cols]

    def _attach_future_return(
        self,
        preds: pd.DataFrame,
        features_df: Optional[pd.DataFrame],
        horizon_bars: int,
    ) -> pd.DataFrame:
        """Attach y_true (future_return) to predictions DataFrame."""
        # First try: merge from features_df if available
        if features_df is not None and "future_return" in features_df.columns:
            key_cols = ["symbol", "bar_idx"]
            if all(c in preds.columns and c in features_df.columns for c in key_cols):
                preds = _drop_future_return_cols(preds)
                y_src = features_df[key_cols + ["future_return"]].copy()
                preds = preds.merge(y_src, on=key_cols, how="left", validate="m:1")
                return preds

        # Fallback: compute from close prices
        if "close" in preds.columns and "symbol" in preds.columns:
            preds = preds.sort_values(["symbol", "bar_idx"]).copy()
            preds["future_return"] = (
                preds.groupby("symbol")["close"].shift(-horizon_bars) / preds["close"] - 1.0
            )

        return preds

    def _save_minimal_predictions(self) -> None:
        """Save a minimal predictions file when no predictions exist."""
        minimal = pd.DataFrame({
            "symbol": ["BTCUSDT"],
            "bar_idx": [0],
            "timestamp_end": [datetime.now(timezone.utc)],
            "close": [30000.0],
            "q05": [-0.01],
            "q50": [0.0],
            "q95": [0.01],
            "x_hat": [0.0],
            "sigma_hat": [0.005],
            "fold": [0],
            "K_label": ["K_BALANCED"],
            "K_pmax": [0.5],
            "K_entropy": [0.5],
            "P_label": ["P_VOL_STABLE"],
            "P_pmax": [0.5],
            "P_entropy": [0.5],
            "C_label": ["C_FLOW_NEUTRAL"],
            "C_pmax": [0.5],
            "C_entropy": [0.5],
            "O_label": ["O_RANGE"],
            "O_pmax": [0.5],
            "O_entropy": [0.5],
            "F_label": ["F_STALL"],
            "F_pmax": [0.5],
            "F_entropy": [0.5],
            "G_label": ["G_BO_FAIL"],
            "G_pmax": [0.5],
            "G_entropy": [0.5],
            "S_label": ["S_UNCERTAIN"],
            "S_pmax": [0.5],
            "S_entropy": [0.5],
            "S_pmax_calibrated": [0.5],
            "regime_entropy": [3.5],
            "abstain": [True],
        })
        minimal.to_parquet(self.out_dir / "predictions.parquet", index=False)

    def save_trades(self, prints_df: pd.DataFrame) -> None:
        """Save trades/prints data.

        Args:
            prints_df: Trade prints DataFrame.
        """
        prints_df.to_parquet(self.out_dir / "trades.parquet", index=False)

    def save_report(self, report: Dict[str, Any]) -> None:
        """Save report as JSON.

        Args:
            report: Report dictionary.
        """
        with open(self.out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save run metadata as JSON.

        Args:
            metadata: Metadata dictionary.
        """
        with open(self.out_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_ralph_loop_artifacts(
        self,
        trials_df: pd.DataFrame,
        selected_json: Dict[str, Any],
    ) -> None:
        """Save Ralph Loop optimization artifacts.

        Args:
            trials_df: DataFrame of hyperparameter trial results.
            selected_json: Selected hyperparameters per fold.
        """
        if not trials_df.empty:
            trials_df.to_parquet(self.out_dir / "hparams_trials.parquet", index=False)

        with open(self.out_dir / "selected_hparams.json", "w") as f:
            json.dump(selected_json, f, indent=2)

    def save_report_markdown(self, content: str) -> None:
        """Save markdown report.

        Args:
            content: Markdown content string.
        """
        with open(self.out_dir / "report.md", "w") as f:
            f.write(content)

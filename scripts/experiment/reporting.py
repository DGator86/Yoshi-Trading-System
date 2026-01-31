"""Report generation utilities for experiments."""
import json
from typing import Any, Dict, List, Optional


class ReportGenerator:
    """Generates markdown experiment reports."""

    def __init__(
        self,
        report: Dict[str, Any],
        config: Dict[str, Any],
        fold_results: List[Dict[str, Any]],
        stability_metrics: Dict[str, float],
        calibration_summary: Dict[str, Any],
        ralph_results: Optional[Dict[str, Any]] = None,
    ):
        """Initialize report generator.

        Args:
            report: Main report dictionary with summary metrics.
            config: Experiment configuration.
            fold_results: List of per-fold result dictionaries.
            stability_metrics: Regime stability metrics.
            calibration_summary: Calibration diagnostics.
            ralph_results: Optional Ralph Loop results.
        """
        self.report = report
        self.config = config
        self.fold_results = fold_results
        self.stability_metrics = stability_metrics
        self.calibration_summary = calibration_summary
        self.ralph_results = ralph_results

    def generate(self) -> str:
        """Generate complete markdown report.

        Returns:
            Complete markdown report as string.
        """
        sections = [
            self._generate_header(),
            self._generate_summary(),
            self._generate_configuration(),
            self._generate_abstention_section(),
            self._generate_stability_section(),
            self._generate_calibration_section(),
            self._generate_fold_results_table(),
        ]

        if self.ralph_results is not None:
            sections.append(self._generate_ralph_loop_section())

        return "\n".join(sections)

    def _generate_header(self) -> str:
        """Generate report header."""
        return "# Experiment Report\n"

    def _generate_summary(self) -> str:
        """Generate summary section."""
        return f"""## Summary
- **Status**: {self.report['status']}
- **90% Coverage**: {self.report['coverage_90']:.4f} (target: 0.87-0.93)
- **Sharpness**: {self.report['sharpness']:.6f}
- **Baseline Sharpness**: {self.report['baseline_sharpness']:.6f}
- **MAE**: {self.report['mae']:.6f}
- **Folds**: {self.report['n_folds']}
"""

    def _generate_configuration(self) -> str:
        """Generate configuration section."""
        symbols = self.config.get("symbols", [])
        seed = self.config.get("random_seed", 1337)
        symbols_str = ", ".join(symbols) if symbols else "N/A"
        return f"""## Configuration
- Symbols: {symbols_str}
- Random Seed: {seed}
"""

    def _generate_abstention_section(self) -> str:
        """Generate abstention section."""
        return f"""## Abstention
- **Abstention Rate**: {self.report['abstention_rate']:.4f}
- Predictions with S_label=S_UNCERTAIN or S_pmax < confidence_floor are marked as abstain
"""

    def _generate_stability_section(self) -> str:
        """Generate stability metrics section."""
        lines = [
            "## Stability Metrics",
            "| Level | Flip Rate | Avg Entropy |",
            "|-------|-----------|-------------|",
        ]

        for level in ["K", "P", "C", "O", "F", "G", "S"]:
            flip_rate = self.stability_metrics.get(f"{level}_flip_rate", 0.0)
            avg_entropy = self.stability_metrics.get(f"{level}_avg_entropy", 0.0)
            lines.append(f"| {level} | {flip_rate:.4f} | {avg_entropy:.4f} |")

        overall_flip = self.stability_metrics.get("overall_flip_rate", 0.0)
        lines.append(f"\n- **Overall Flip Rate**: {overall_flip:.4f}")

        return "\n".join(lines) + "\n"

    def _generate_calibration_section(self) -> str:
        """Generate calibration summary section."""
        return f"""## Calibration Summary
- **ECE (Raw)**: {self.calibration_summary['avg_ece_raw']:.4f}
- **ECE (Calibrated)**: {self.calibration_summary['avg_ece_calibrated']:.4f}
- **Improvement**: {self.calibration_summary['calibration_improvement']:.4f}
"""

    def _generate_fold_results_table(self) -> str:
        """Generate fold results table."""
        lines = [
            "## Fold Results",
            "| Fold | N_Train | N_Test | Coverage | Sharpness | MAE | Abstain |",
            "|------|---------|--------|----------|-----------|-----|---------|",
        ]

        for fr in self.fold_results:
            lines.append(
                f"| {fr['fold']} | {fr['n_train']} | {fr['n_test']} | "
                f"{fr['model_coverage_90']:.4f} | {fr['model_sharpness']:.6f} | "
                f"{fr['model_mae']:.6f} | {fr['abstention_rate']:.4f} |"
            )

        return "\n".join(lines) + "\n"

    def _generate_ralph_loop_section(self) -> str:
        """Generate Ralph Loop section."""
        if self.ralph_results is None:
            return ""

        selected = self.ralph_results["selected_json"]
        robustness = self.ralph_results["robustness"]
        trials_df = self.ralph_results["trials_df"]

        n_candidates = (
            len(trials_df["candidate_id"].unique())
            if not trials_df.empty
            else 0
        )

        lines = [
            "\n## Ralph Loop (Hyperparameter Selection)",
            "- **Enabled**: Yes",
            f"- **Candidates Evaluated**: {n_candidates}",
        ]

        # Global best
        if selected.get("global_best"):
            gb = selected["global_best"]
            lines.extend([
                f"- **Global Best Candidate**: {gb.get('candidate_id', 'N/A')}",
                f"- **Global Best Params**: `{json.dumps(gb.get('params', {}))}`",
                f"- **Selection Count**: {gb.get('selection_count', 0)} folds",
            ])

        # Per-fold table
        lines.extend([
            "\n### Per-Fold Selected Parameters",
            "| Fold | Candidate | Parameters |",
            "|------|-----------|------------|",
        ])

        for fold_str, fold_data in selected.get("per_fold", {}).items():
            params_str = json.dumps(fold_data.get("params", {}))
            cand_id = fold_data.get("candidate_id", "N/A")
            lines.append(f"| {fold_str} | {cand_id} | `{params_str}` |")

        # Robustness stats
        lines.extend([
            "\n### Robustness (Std across Outer Folds)",
            f"- **Coverage Std**: {robustness.get('coverage_90_std', 0.0):.4f}",
            f"- **Sharpness Std**: {robustness.get('sharpness_std', 0.0):.6f}",
            f"- **MAE Std**: {robustness.get('mae_std', 0.0):.6f}",
        ])

        return "\n".join(lines) + "\n"

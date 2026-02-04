# Ralph Contract (DO NOT VIOLATE)

## Objective
Build a deterministic, print-dominant forecasting system that outputs:
- target time y (wall clock), point estimate x_hat, uncertainty sigma_hat
- quantiles q05/q50/q95
and proves out-of-sample calibration + robustness.

## Phases & Permissions

### Phase A (Scaffold) — NO tuning
Allowed:
- create repo structure, minimal working code, smoke pipeline, unit tests
- fixed hyperparams only
Not allowed:
- hyperparameter search/optimization
- changing evaluation gates or score definitions
PASS if:
- scripts/run_smoke.sh runs end-to-end and produces required artifacts
- pytest passes

### Phase B (Correctness & Determinism)
Allowed:
- hashing, manifests, deterministic seeds, reproducible outputs
PASS if:
- two identical runs produce identical report hash

### Phase C (KPCOFGS)
Allowed:
- implement KPCOFGS ladder outputs K..S with probabilities + calibration
- add regime stability checks (no rapid flip-flopping)
Not allowed:
- changing harness gates or baseline definitions

### Phase D (Harness & Baseline)
Allowed:
- nested walk-forward with purge/embargo
- baseline forecast implementation and comparisons
Not allowed:
- loosening gates to “pass”
PASS if:
- fold outputs exist, gates enforced, baseline computed, report.json correct

### Phase E (Constrained Hyperparameter Tuning)
Allowed:
- add tuning ONLY within configs/models.yaml bounds
- nested tuning only (train/val inside each fold)
Not allowed:
- using test folds for selection
- expanding search space without explicit justification in report.md

### Phase F (Situational Awareness)
Allowed:
- slow adaptation driven by KPCOFGS probabilities (horizon selection, abstain thresholds)
Not allowed:
- per-bar reoptimization
- leaking future info

## Immutable Definitions
- Prints are the primary data (trade events).
- Domains D0..D3 are fixed as counts of trades.
- Forecast quality is judged by proper scoring + calibration coverage + sharpness.
- Baseline = random walk + realized-vol cone.
- Artifacts required: data_manifest.json, predictions.parquet, trades.parquet,
  feature_registry.json, report.json, report.md, run_metadata.json.

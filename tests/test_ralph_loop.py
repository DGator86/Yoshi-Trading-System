
import pandas as pd

from gnosis.loop.ralph import RalphLoop
from gnosis.harness.trade_walkforward import TradeWalkForwardHarness
from gnosis.ingest import generate_stub_prints as ingest_generate_stub_prints
def generate_stub_prints(n: int = 5000, seed: int = 123):
    """
    Minimal deterministic prints dataframe for RalphLoop tests.
    Columns chosen to satisfy the pipeline stages used in ralph.py/run_experiment.py.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    # Fake prints: price random walk + volume
    price = 100.0 + np.cumsum(rng.normal(0, 0.05, size=n))
    size = rng.integers(1, 10, size=n)
    ts = pd.date_range("2024-01-01", periods=n, freq="S")

    df = pd.DataFrame({
        "ts": ts,
        "price": price,
        "size": size,
    })
    return df
def test_ralph_loop_runs_on_stub_prints():
    prints = ingest_generate_stub_prints(["BTCUSDT"], n_days=4, trades_per_day=800, seed=123)
    base_cfg = {
        "domains": {"D0": {"n_trades": 200}},
        "targets": {"horizon_bars": 10},
        "walkforward": {"outer_folds": 3, "train_trades": 1200, "val_trades": 400, "test_trades": 400, "purge_trades": 200, "embargo_trades": 200, "horizon_trades": 200},
        "models": {"predictor": {"l2_reg": 1.0}},
        "regimes": {},
        "particle": {"flow": {"span": 5}},
    }
    hcfg = {
        "grid": {
            "domains_D0_n_trades": [100, 200],
            "particle_flow_span": [3, 5],
            "predictor_l2_reg": [0.1, 1.0],
            "confidence_floor_scale": [0.6, 1.0],
        },
        "inner_folds": {"n_folds": 2, "train_ratio": 0.6, "val_ratio": 0.4},
        "target_coverage": 0.90,
        "w1_coverage": 0.6,
        "w2_sharpness": 0.2,
        "w3_mae": 0.2,
        "inner_purge_trades": 200,
    }
    outer = TradeWalkForwardHarness(base_cfg["walkforward"], trades_per_bar=200, horizon_bars_default=10)
    rl = RalphLoop(base_cfg, hcfg)
    trials_df, selected_json = rl.run(prints, outer)
    assert isinstance(trials_df, pd.DataFrame)
    assert "per_fold" in selected_json

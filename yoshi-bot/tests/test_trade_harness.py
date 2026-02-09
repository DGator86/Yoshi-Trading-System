from gnosis.harness.trade_walkforward import TradeWalkForwardHarness

def test_trade_harness_generates_folds():
    cfg = {
        "outer_folds": 4,
        "train_trades": 1000,
        "val_trades": 200,
        "test_trades": 200,
        "purge_trades": 50,
        "embargo_trades": 50,
        "horizon_trades": 50,
    }
    h = TradeWalkForwardHarness(cfg, trades_per_bar=200, horizon_bars_default=10)
    folds = list(h.generate_folds(5000))
    assert len(folds) > 0
    for f in folds:
        assert f.train_start < f.train_end < f.val_start < f.val_end < f.test_start < f.test_end

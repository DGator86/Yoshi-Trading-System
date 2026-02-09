import json
from pathlib import Path
import pandas as pd

from gnosis.metrics.scoring import score_predictions

pred_path = Path("reports/latest/predictions.parquet")
if not pred_path.exists():
    raise SystemExit("ERROR: reports/latest/predictions.parquet not found. Run experiment first.")

df = pd.read_parquet(pred_path)

# Unify y_true column if it came out as _x/_y
if "future_return" not in df.columns:
    fx = "future_return_x" if "future_return_x" in df.columns else None
    fy = "future_return_y" if "future_return_y" in df.columns else None
    if fx and fy:
        df["future_return"] = df[fy].combine_first(df[fx])
    elif fy:
        df["future_return"] = df[fy]
    elif fx:
        df["future_return"] = df[fx]

# If still missing, try to reconstruct from close using horizon_bars from report/config
if "future_return" not in df.columns or df["future_return"].isna().all():
    # best-effort: horizon from report.json if present, else 10
    H = 10
    rep = Path("reports/latest/report.json")
    if rep.exists():
        try:
            r = json.loads(rep.read_text())
            H = int(r.get("horizon_bars", H))
        except Exception:
            pass
    df = df.sort_values(["symbol","bar_idx"]).reset_index(drop=True)
    df["future_return"] = df.groupby("symbol")["close"].shift(-H) / df["close"] - 1.0

# Score on non-abstained
if "abstain" in df.columns:
    df_s = df[~df["abstain"]].copy()
else:
    df_s = df.copy()

scores = score_predictions(df_s, y_col="future_return")
print("n_scored =", scores.get("n"))
print("WIS      =", scores.get("wis"))
print("IS90     =", scores.get("is90"))
print("MAE      =", scores.get("mae"))
print("missing  =", scores.get("missing"))

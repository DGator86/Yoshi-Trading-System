import json
import subprocess
from pathlib import Path

import yaml

BASE_CFG_PATH = Path("configs/experiment.yaml")

SIGMA_SCALES = [1.0, 1.2, 1.4, 1.6, 1.8]
CLIP_HIS     = [0.02, 0.03, 0.04, 0.06, 0.08]

def run(cmd):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(r.returncode)
    return r.stdout.strip()

def score_latest():
    out = run("python3 scripts/score_latest.py")
    lines = {l.split("=",1)[0].strip(): l.split("=",1)[1].strip() for l in out.splitlines() if "=" in l}
    return {
        "n_scored": int(lines.get("n_scored","0")),
        "WIS": float(lines.get("WIS","nan")),
        "IS90": float(lines.get("IS90","nan")),
        "MAE": float(lines.get("MAE","nan")),
        "missing": lines.get("missing", None),
        "raw": out,
    }

def read_report():
    r = json.loads(Path("reports/latest/report.json").read_text())
    return {
        "status": r.get("status"),
        "coverage_90": float(r.get("coverage_90", float("nan"))),
        "sharpness": float(r.get("sharpness", float("nan"))),
        "abstention": float(r.get("abstention_rate", float("nan"))),
    }

base = yaml.safe_load(BASE_CFG_PATH.read_text())

results = []
for ss in SIGMA_SCALES:
    for ch in CLIP_HIS:
        cfg = yaml.safe_load(BASE_CFG_PATH.read_text())

        cfg.setdefault("forecast", {})
        cfg["forecast"]["sigma_scale"] = ss
        cfg["forecast"].setdefault("residual_calibration", {})
        cfg["forecast"]["residual_calibration"]["enabled"] = True
        cfg["forecast"]["residual_calibration"]["method"] = "quantile"
        cfg["forecast"]["residual_calibration"]["alpha"] = 0.10
        cfg["forecast"]["residual_calibration"]["cal_frac"] = 0.20
        cfg["forecast"]["residual_calibration"]["min_cal"] = 50
        cfg["forecast"]["residual_calibration"]["clip_hi"] = ch

        # write temp config
        tmp_cfg = Path("reports/grid_tmp.yaml")
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

        # run
        run("rm -rf reports/latest")
        run(f"python3 scripts/run_experiment.py --config {tmp_cfg}")

        rep = read_report()
        sc = score_latest()

        rec = {
            "sigma_scale": ss,
            "clip_hi": ch,
            **rep,
            **sc,
        }
        results.append(rec)
        print(f"done ss={ss:.2f} clip_hi={ch:.2f}  WIS={sc['WIS']:.6f}  cov={rep['coverage_90']:.3f}  sharp={rep['sharpness']:.4f}")

# rank: minimize WIS, with soft penalty if coverage < 0.88 or > 0.94
def rank_key(r):
    cov = r["coverage_90"]
    penalty = 0.0
    if cov < 0.88: penalty += (0.88 - cov) * 0.5
    if cov > 0.94: penalty += (cov - 0.94) * 0.5
    return r["WIS"] + penalty

results_sorted = sorted(results, key=rank_key)

Path("reports/grid_results.json").write_text(json.dumps(results_sorted, indent=2))
best = results_sorted[0]
Path("reports/grid_best.yaml").write_text(yaml.safe_dump(best, sort_keys=False))

print("\nTOP 10:")
for i, r in enumerate(results_sorted[:10], 1):
    print(f"{i:02d}) ss={r['sigma_scale']:.2f} clip_hi={r['clip_hi']:.2f}  WIS={r['WIS']:.6f} IS90={r['IS90']:.6f} MAE={r['MAE']:.6f}  cov={r['coverage_90']:.3f} sharp={r['sharpness']:.4f}")

print("\nBEST SAVED -> reports/grid_best.yaml")

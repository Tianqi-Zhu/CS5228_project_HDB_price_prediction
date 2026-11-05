"""Rebuild `learnability.json` from the per-run curves that actually reached disk.

Why this exists: `learnability.py` writes its aggregate summary only at the very end of
`main()`. The process was stopped mid-run (to honour the instruction not to start further
budgets), so the aggregate still described the first verification run instead of the real
64/256/2000 outcomes. The per-run `curve.json` files were flushed every epoch and are
intact, so the summary can be reconstructed from them without re-training anything.

This script is a pure reader: it loads each curve, recomputes the same fields
`learnability.py` would have written, and records whether the run finished, hit the
5 SGD early-stop, hit the step cap, or was externally terminated.

Run:
    .venv/bin/python -m experiments.diagnostics_phase2_a.rebuild_summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

OUT = ROOT / "reports" / "diagnostics_phase2_a"
TARGET_RMSE = 5.0


def summarise(directory: Path) -> dict:
    record = {"run_id": directory.name}
    curve = json.loads((directory / "curve.json").read_text())
    evaluations = [row for row in curve if "eval_rmse_sgd" in row]
    last = curve[-1] if curve else {}

    # the file name encodes the settings; parse them rather than guessing
    parts = directory.name.split("_")
    for part in parts:
        if part.startswith("budget"):
            record["budget"] = int(part[len("budget"):])
        elif part.startswith("epochs"):
            record["epochs_requested"] = int(part[len("epochs"):])
        elif part.startswith("lr"):
            record["lr"] = float(part[len("lr"):])
        elif part.startswith("seed"):
            record["seed"] = int(part[len("seed"):])

    record["entries"] = len(curve)
    record["final_steps"] = last.get("steps")
    record["first_eval_rmse_sgd"] = (evaluations[0]["eval_rmse_sgd"]
                                     if evaluations else None)
    record["final_eval_rmse_sgd"] = last.get("final_eval_rmse_sgd",
                                             evaluations[-1]["eval_rmse_sgd"]
                                             if evaluations else None)
    record["final_eval_mae_sgd"] = last.get("final_eval_mae_sgd",
                                            evaluations[-1]["eval_mae_sgd"]
                                            if evaluations else None)
    record["final_eval_bias_sgd"] = last.get("final_eval_bias_sgd",
                                             evaluations[-1]["eval_bias_sgd"]
                                             if evaluations else None)

    half = evaluations[len(evaluations) // 2:]
    if len(half) >= 2:
        xs = np.array([row["epoch"] for row in half], dtype=float)
        ys = np.array([row["eval_rmse_sgd"] for row in half], dtype=float)
        record["second_half_slope_sgd_per_epoch"] = float(np.polyfit(xs, ys, 1)[0])
    else:
        record["second_half_slope_sgd_per_epoch"] = None

    # how this run ended
    final_rmse = record["final_eval_rmse_sgd"]
    if record["final_steps"] is None:
        record["status"] = "unknown"
    elif final_rmse is not None and final_rmse <= TARGET_RMSE:
        record["status"] = "passed_target"
    elif record.get("epochs_requested") and \
            record["entries"] >= record["epochs_requested"]:
        record["status"] = "budget_exhausted"
    else:
        record["status"] = "stopped_before_budget_end"

    record["reconstructed_from"] = "curve.json (per-epoch flush)"
    return record


def main() -> int:
    directories = sorted(d for d in OUT.glob("budget*") if (d / "curve.json").exists())
    if not directories:
        print("no curves found")
        return 1

    runs = [summarise(directory) for directory in directories]
    runs.sort(key=lambda r: (r.get("budget", 0), r["run_id"]))

    print(f"{'run':46s} {'steps':>6} {'first':>9} {'final':>9} {'MAE':>8} "
          f"{'slope':>10} status")
    for run in runs:
        print(f"{run['run_id']:46s} {str(run['final_steps']):>6} "
              f"{run['first_eval_rmse_sgd']:>9.3f} {run['final_eval_rmse_sgd']:>9.3f} "
              f"{(run['final_eval_mae_sgd'] or float('nan')):>8.3f} "
              f"{(run['second_half_slope_sgd_per_epoch'] or float('nan')):>+10.5f} "
              f"{run['status']}")

    payload = {
        "note": ("reconstructed after the training process was stopped mid-run; the "
                 "aggregate learnability.json written by the process described an older "
                 "verification run, so these values come from the per-epoch curves on disk"),
        "target_rmse_sgd": TARGET_RMSE,
        "runs": runs,
        "archived_runs": [
            {"run_id": d.name,
             "note": "superseded or interrupted; kept as evidence"}
            for d in sorted((OUT / "archive").glob("run_*"))],
    }
    (OUT / "learnability_reconstructed.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT / 'learnability_reconstructed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

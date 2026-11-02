"""Compare completed API validation runs on identical original 77-feature rows."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.evaluation import metrics
from analysis.tabpfn_api_77 import file_hash, write_json

OUT = ROOT / "reports/tabpfn_api_77"
RUNS = {
    "TabPFN 3.5 API": "validation_v3.5_default_standard_e1",
    "TabPFN 3.5 Thinking API (medium)": "validation_v3.5_default_medium_e1",
}
REFERENCE = ROOT / "reports/tabpfn_v3_5_fast_calendar_80_20/predictions_outer_categorical.csv"


def summarize():
    reference = pd.read_csv(REFERENCE, float_precision="round_trip")
    if len(reference) != 28995 or not reference.row_id.is_unique:
        raise ValueError("Original validation reference changed")
    baseline = metrics(reference.actual, reference.predicted)
    rows = [dict(model="Local frozen 3.5-Fast (original 77)", n=len(reference),
                 **baseline, delta_rmse_sgd=0.0)]
    monthly, sources, timings = [], {str(REFERENCE.relative_to(ROOT)): file_hash(REFERENCE)}, {}
    for name, run in RUNS.items():
        path = OUT / run / "predictions.csv"
        pred = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(pred.row_id, reference.row_id)
        np.testing.assert_array_equal(pred.actual, reference.actual)
        np.testing.assert_array_equal(pred.month.astype(str), reference.month.astype(str))
        if not np.isfinite(pred.predicted.to_numpy()).all():
            raise ValueError(f"Nonfinite predictions: {name}")
        score = metrics(pred.actual, pred.predicted)
        rows.append(dict(model=name, n=len(pred), **score,
                         delta_rmse_sgd=score["rmse_sgd"] - baseline["rmse_sgd"]))
        timing_path = ROOT / ".cache/tabpfn_api_77" / run / "last_timings.json"
        timings[name] = json.loads(timing_path.read_text(encoding="utf-8"))
        sources[str(path.relative_to(ROOT))] = file_hash(path)
        for month, part in pred.groupby("month", sort=True):
            control = reference.loc[part.index]
            monthly.append(dict(model=name, month=str(month), n=len(part),
                                **metrics(part.actual, part.predicted),
                                delta_rmse_sgd=metrics(part.actual, part.predicted)["rmse_sgd"]
                                - metrics(control.actual, control.predicted)["rmse_sgd"]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False, float_format="%.10f")
    pd.DataFrame(monthly).to_csv(OUT / "monthly_comparison.csv", index=False, float_format="%.10f")
    write_json(OUT / "server_timings.json", timings)
    write_json(OUT / "comparison_verification.json", dict(
        original_validation_rows=28995, identical_row_ids_order_months_and_labels=True,
        predictions_finite=True, prediction_sha256=sources))
    report = ["# TabPFN 3.5 API comparison on the original 77 features", "",
              "All models use the same 121,005 January 2021–May 2024 context rows and "
              "28,995 June 2024–March 2025 validation rows. All 77 original predictors "
              "and missing values are retained, with nine explicitly categorical columns. "
              "Training targets are rental residuals; the temporal baseline is added once.", "",
              "| Model | RMSE (SGD) | MAE (SGD) | Bias (SGD) | RMSE change vs local Fast |",
              "| --- | ---: | ---: | ---: | ---: |"]
    report.extend(f"| {r['model']} | {r['rmse_sgd']:.3f} | {r['mae_sgd']:.3f} | "
                  f"{r['bias_sgd']:+.3f} | {r['delta_rmse_sgd']:+.3f} |" for r in rows)
    report.extend(["", "Server-reported timings (seconds; upload and client polling excluded):", "",
                   "| API model | Fit elapsed | Prediction compute |",
                   "| --- | ---: | ---: |"])
    for name, timing in timings.items():
        fit = (timing.get("fit") or {}).get("elapsed_s")
        predict = (timing.get("predict") or {}).get("predict_s")
        report.append(f"| {name} | {fit if fit is not None else 'unreported'} | "
                      f"{predict if predict is not None else 'unreported'} |")
    report.extend(["", "The standard API run requests one estimator and full context with seed 42. "
                   "Thinking uses medium effort, a 600-second fit budget, MSE optimization, "
                   "and the existing numeric approval-month index for time-ordered validation. "
                   "The provider controls its internal search and fitted ensemble; its "
                   "internal folds do not rebuild the precomputed historical features.", "",
                   "These runs use managed API models, without the locally trained DiagScale "
                   "adapter. The local frozen 3.5-Fast result is the directly aligned feature "
                   "baseline. Managed default checkpoints and preprocessing can change, so "
                   "the comparison does not isolate architecture alone. This is a repeatedly "
                   "inspected development holdout, not a new generalization estimate.", "",
                   "Each run directory contains settings, predictions, pooled and monthly "
                   "metrics. Saved fit references, prediction jobs and server timings are in "
                   "the matching ignored `.cache/tabpfn_api_77/` directory. No final 150,000-row "
                   "refit or competition submission is produced by this validation comparison.", ""])
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    summarize()

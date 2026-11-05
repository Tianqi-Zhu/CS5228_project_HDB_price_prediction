"""Independently replay conditional-mean scores, alignment and earlier selection."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.data import sha256
from analysis.compare_local_economy_landscape import write_json
from analysis.run_economic_conditional_mean import OUT, CACHE, SOURCE, PREPARED, FOLDS, DESIGNS


def main():
    settings = json.loads((OUT / "settings.json").read_text())
    for name, digest in settings["source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    prepared = joblib.load(PREPARED)
    selected = json.loads((OUT / "selection.json").read_text())
    rows, frames = [], {}
    for fold in FOLDS:
        bundle = prepared["folds"][fold + "_d0"]
        for design in DESIGNS:
            tag = f"calibration/{design}/{fold}/seed42"
            report, cache = OUT / tag, CACHE / tag
            result = json.loads((report / "result.json").read_text())
            for path, key in ((cache / "model.pt", "checkpoint_sha256"),
                              (cache / "preprocessors.joblib", "preprocessors_sha256"),
                              (report / "predictions.csv", "predictions_sha256")):
                assert sha256(path) == result[key]
            f = pd.read_csv(report / "predictions.csv", float_precision="round_trip")
            assert len(f) == result["rows"] == 2000 and f.row_id.is_unique
            for actual, expected in ((f.row_id, bundle["X_query"].index),
                                     (f.month, bundle["query_month"]),
                                     (f.actual, bundle["y_query"]),
                                     (f.weight, bundle["query_weights"])):
                np.testing.assert_array_equal(actual, expected)
            assert set(f.row_id).isdisjoint(bundle["X_train"].index)
            assert max(bundle["train_month"]) < min(f.month)
            rmse = float(np.sqrt(np.average((f.predicted - f.actual)**2, weights=f.weight)))
            assert abs(rmse - result["weighted_rmse_sgd"]) < 1e-10
            assert not result["outer_labels_used_for_selection"]
            assert result["checkpoint_replay_max_difference"] == 0
            frames[fold, design] = f
            rows.append(dict(fold=fold, design=design, rmse_sgd=rmse,
                             mean_shift_sgd=float(np.average(f.conditional_shift_sgd, weights=f.weight)),
                             mean_interaction_sgd=float(np.average(f.interaction_sgd, weights=f.weight)),
                             bias_sgd=float(np.average(f.predicted - f.actual, weights=f.weight)),
                             min_prediction_sgd=float(f.predicted.min())))
    expected_gate = True
    for fold in FOLDS:
        coe, rent = [next(r for r in rows if r["fold"] == fold and r["design"] == design) for design in ("rent_coe", "rent")]
        baseline = json.loads((SOURCE / f"calibration/T0/{fold}/result.json").read_text())
        expected_gate &= coe["rmse_sgd"] < rent["rmse_sgd"] and coe["rmse_sgd"] < baseline["weighted_rmse_sgd"]
    assert selected["advanced"] == expected_gate and not selected["outer_labels_used"]
    if not expected_gate:
        assert not list((OUT / "outer").glob("**/result.json"))
    pd.DataFrame(rows).to_csv(OUT / "conditioning_diagnostics.csv", index=False)
    write_json(OUT / "independent_audit.json", dict(verified=True,
        rows=sum(len(f) for f in frames.values()), exact_original_query_alignment=True,
        all_calibration_scores_recomputed=True, source_and_checkpoint_hashes_verified=True,
        earlier_gate_recomputed=True, advanced=expected_gate,
        unsupported_later_scoring_absent=not expected_gate, audit_code_sha256=sha256(Path(__file__))))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

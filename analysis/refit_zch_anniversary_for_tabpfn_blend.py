"""Refit the seven original zch anniversary members on all training labels.

Run this file with the repository's Python environment. It imports the zch
branch from the isolated ``.cache/zch-reproduction`` worktree so that the
current TabPFN branch and its analysis package remain untouched.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

CURRENT = Path(__file__).resolve().parents[1]
ZCH = CURRENT / ".cache/zch-reproduction"
sys.path.insert(0, str(ZCH))

import joblib
import numpy as np
import pandas as pd

from analysis.experiments.best_ensemble_submission import prepare
from analysis.experiments.frontier_search import ROOT
from analysis.experiments.second_wave_cohort_extensions import fit as fit_anniversary
from analysis.experiments.second_wave_normalized import fit as fit_normalized
from analysis.pipeline.data import sha256


WEIGHTS = {
    "normalized_market/compact_ratio15_market_last": 0.13094,
    "anniversary/broad_lgb15": 0.09659,
    "anniversary/broad_lgb63": 0.10352,
    "anniversary/exact_lgb15": 0.17311,
    "anniversary/exact_lgb63": 0.00658,
    "anniversary/window_cat6": 0.41380,
    "anniversary/window_lgb63": 0.07546,
}
OUT = CURRENT / "reports/zch_anniversary_tabpfn_blend"
MODEL_DIR = CURRENT / ".cache/zch_anniversary_tabpfn_blend/models"


def main() -> None:
    if ROOT.resolve() != ZCH.resolve():
        raise AssertionError("zch code was not imported from the isolated worktree")
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle, hashes = prepare()
    if len(bundle["y_fit"]) != 150000 or len(bundle["row_val"]) != 50000:
        raise AssertionError("Unexpected full-data zch split")
    np.testing.assert_array_equal(bundle["row_val"], np.arange(50000))
    combined = np.zeros(50000, float)
    members = {}
    for name, weight in WEIGHTS.items():
        group, member = name.split("/")
        result = ROOT / "output/experiment_runs/round_02_anniversary_history" / group / "screen" / member / "results.json"
        record = json.loads(result.read_text(encoding="utf-8"))
        spec, iterations = record["spec"], int(record["iterations"])
        if spec["name"] != member or iterations <= 0:
            raise AssertionError("zch screen specification changed")
        path = OUT / f"predictions_zch_{member}.csv"
        checkpoint = MODEL_DIR / f"{member}.joblib"
        if path.exists():
            frame = pd.read_csv(path)
            np.testing.assert_array_equal(frame.row_id, bundle["row_val"])
            prediction = frame.predicted.to_numpy(float)
        else:
            print(f"Refitting zch {member}: {iterations} rounds", flush=True)
            if group == "normalized_market":
                model, prediction, count = fit_normalized(spec, bundle, iterations)
            else:
                model, prediction, count, _ = fit_anniversary(spec, bundle, iterations)
            if count != iterations or len(prediction) != 50000 or not np.isfinite(prediction).all():
                raise AssertionError("Incomplete zch full-data refit")
            joblib.dump({"model": model, "spec": spec, "iterations": count,
                         "source_sha256": hashes}, checkpoint, compress=1)
            pd.DataFrame({"row_id": bundle["row_val"], "predicted": prediction}).to_csv(
                path, index=False, float_format="%.17g")
            del model
            gc.collect()
        combined += weight * prediction
        members[name] = {"weight": weight, "iterations": iterations,
                         "prediction_sha256": sha256(path), "model_sha256": sha256(checkpoint) if checkpoint.exists() else None}
    output = OUT / "predictions_zch_full.csv"
    pd.DataFrame({"row_id": bundle["row_val"], "predicted": combined}).to_csv(
        output, index=False, float_format="%.17g")
    (OUT / "zch_full_provenance.json").write_text(json.dumps({
        "members": members, "source_sha256": hashes,
        "prediction_sha256": sha256(output), "fitting_rows": 150000,
        "query_rows": 50000, "weights_source": "zch round-02 earlier-selected published ensemble",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()

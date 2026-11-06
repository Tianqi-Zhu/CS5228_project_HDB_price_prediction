"""Isolated zch imports for the training-only half-median ensemble experiment."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import time

CURRENT = Path(__file__).resolve().parents[1]
ZCH = CURRENT / ".cache/zch-reproduction"
sys.path.insert(0, str(ZCH))

import joblib
import numpy as np
import pandas as pd

from analysis.experiments.frontier_search import ROOT, frame_data, DistributionHistory, month_number
from analysis.experiments.second_wave_anniversary import AnniversaryHistory
from analysis.experiments.second_wave_cohort_extensions import fit as fit_anniversary
from analysis.experiments.second_wave_normalized import fit as fit_normalized
from analysis.pipeline.features import model_inputs, FEATURE_CANDIDATES
from analysis.pipeline.data import sha256

OUT = CURRENT / "reports/half_median_zch_tabpfn_ensemble"
CACHE = CURRENT / ".cache/half_median_zch_tabpfn_ensemble"
WEIGHTS = {
    "normalized_market/compact_ratio15_market_last": .13094,
    "anniversary/broad_lgb15": .09659, "anniversary/broad_lgb63": .10352,
    "anniversary/exact_lgb15": .17311, "anniversary/exact_lgb63": .00658,
    "anniversary/window_cat6": .41380, "anniversary/window_lgb63": .07546,
}


def main() -> None:
    assert ROOT.resolve() == ZCH.resolve()
    manifest = json.loads((OUT / "feature_manifest.json").read_text())
    assert sha256(CURRENT / "data/train.csv") == manifest["source_sha256"]
    assert sha256(OUT / "training_row_ids.csv") == manifest["training_ids_sha256"]
    ids = pd.read_csv(OUT / "training_row_ids.csv").row_id.to_numpy(int)
    source, hashes = frame_data()
    assert sha256(ROOT / "data/train.csv") == manifest["source_sha256"]
    fitting = source.loc[ids].copy()
    query = source.loc[source.RENT_APPROVAL_DATE.ge("2024-06")].copy()
    actual = query.MONTHLY_RENT.to_numpy(float).copy()
    # Query targets are deliberately blinded before all feature generation/fits.
    query["MONTHLY_RENT"] = 0.0
    assert len(fitting) == manifest["context_rows"] and len(query) == manifest["validation_rows"]
    assert fitting.RENT_APPROVAL_DATE.max() < query.RENT_APPROVAL_DATE.min()
    bundle_path = CACHE / "zch_features.joblib"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        np.testing.assert_array_equal(bundle["raw_fit"].index, ids)
    else:
        print("TREES rebuilding compact, distribution, market and anniversary features", flush=True)
        x, q = model_inputs(fitting, query, FEATURE_CANDIDATES["compact_history_27"])
        history = DistributionHistory().fit(fitting)
        extra = history.transform(fitting, month_number(fitting.RENT_APPROVAL_DATE), neighbors=False)
        cutoff = np.full(len(query), int(month_number(pd.Series(["2024-06"]))[0]))
        other = history.transform(query, cutoff, neighbors=False)
        anniversary = AnniversaryHistory().fit(fitting)
        bundle = dict(x_fit=x, x_val=q, extra_fit=extra, extra_val=other,
                      ann_fit=anniversary.transform(fitting), ann_val=anniversary.transform(query),
                      y_fit=fitting.MONTHLY_RENT.to_numpy(float), y_val=np.zeros(len(query)),
                      row_val=query.index.to_numpy(), date_val=query.RENT_APPROVAL_DATE.to_numpy(),
                      raw_fit=fitting, raw_val=query,
                      reference={method:fitting.groupby(["RENT_APPROVAL_DATE", "FLAT_TYPE_CLEAN"]).MONTHLY_RENT.transform(method).to_numpy(float) for method in ["mean", "median"]})
        joblib.dump(bundle, bundle_path, compress=3)
    assert np.all(bundle["y_val"] == 0)
    combined = np.zeros(len(query))
    members = {}
    for name, weight in WEIGHTS.items():
        group, member = name.split("/")
        record_path = ROOT / "output/experiment_runs/round_02_anniversary_history" / group / "screen" / member / "results.json"
        record = json.loads(record_path.read_text())
        spec, iterations = record["spec"], int(record["iterations"])
        path = OUT / f"predictions_zch_{member}.csv"
        begun = time.perf_counter()
        if path.exists():
            saved = pd.read_csv(path)
            np.testing.assert_array_equal(saved.row_id, query.index)
            np.testing.assert_array_equal(saved.actual, actual)
            prediction = saved.predicted.to_numpy(float)
        else:
            print(f"TREES fitting {member}: {iterations} rounds", flush=True)
            if group == "normalized_market":
                model, prediction, count = fit_normalized(spec, bundle, iterations)
            else:
                model, prediction, count, _ = fit_anniversary(spec, bundle, iterations)
            assert count == iterations and len(prediction) == len(query) and np.isfinite(prediction).all()
            joblib.dump(dict(model=model, spec=spec, iterations=count, training_ids_sha256=manifest["training_ids_sha256"]), CACHE / f"{member}.joblib", compress=1)
            pd.DataFrame(dict(row_id=query.index, month=query.RENT_APPROVAL_DATE, actual=actual, predicted=prediction)).to_csv(path, index=False, float_format="%.17g")
            del model
            gc.collect()
        combined += weight * prediction
        members[name] = dict(weight=weight, iterations=iterations, spec=spec, source_spec_sha256=sha256(record_path),
                             prediction_sha256=sha256(path), seconds=time.perf_counter()-begun)
        print(f"TREES done {member}", flush=True)
    assert np.isclose(sum(WEIGHTS.values()), 1)
    output = OUT / "predictions_zch.csv"
    pd.DataFrame(dict(row_id=query.index, month=query.RENT_APPROVAL_DATE, actual=actual, predicted=combined)).to_csv(output, index=False, float_format="%.17g")
    (OUT / "zch_provenance.json").write_text(json.dumps(dict(members=members, source_hashes=hashes,
        context_rows=len(fitting), query_rows=len(query), validation_targets_blinded=True,
        feature_sha256=sha256(bundle_path), prediction_sha256=sha256(output)), indent=2) + "\n", encoding="utf-8")
    print("TREES complete", flush=True)


if __name__ == "__main__":
    main()

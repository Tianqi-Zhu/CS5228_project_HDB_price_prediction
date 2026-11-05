"""Archive historical development predictions for reproducible offline analysis.

This preparation step reads trusted local development artifacts. It rebuilds
feature/support definitions from raw labels using the original fitting windows,
but neither trains models nor selects weights. The resulting NPZ contains no
pickled objects; the evaluator needs only the archive and the supplied raw CSVs.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from analysis.pipeline.config import ROOT

DEFAULT = ROOT / "experiment/main/M00/inputs"
WINDOWS = {"Y": ("outer", "2024-04", 35158), "J": ("june", "2024-06", 28995)}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schema(frame):
    return [{"name": c, "dtype": str(frame[c].dtype)} for c in frame]


def same_input_codes(frame):
    # Exact tuple equality, including missing values; no hash-collision shortcut.
    return pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)[0].astype(np.int32)


def prepare(source_root=ROOT, destination=DEFAULT):
    from analysis.experiments.second_wave_anniversary import data
    from analysis.experiments.report_selection_trials import select_features
    from analysis.experiments.third_wave_training import inputs

    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(ROOT) or destination == ROOT or destination.is_relative_to(ROOT / "data"):
        raise ValueError("Archive destination must stay in this repository outside data/.")
    if destination.exists():
        raise FileExistsError("Use a new input archive directory; historical evidence is immutable.")
    raw = pd.read_csv(ROOT / "data/train.csv")
    arrays, records, windows, feature_schemas = {}, [], {}, {}
    sources = []

    def source(path, kind):
        path = Path(path)
        relative = path.relative_to(source_root).as_posix()
        if not any(s["path"] == relative for s in sources):
            sources.append({"path": relative, "sha256": sha(path), "kind": kind,
                            "origin": "local historical development artifact; not a newly trained result"})
        return relative

    def add(window, model, pred_path, result_path, columns, policy, reference=None):
        stage, start, n = WINDOWS[window]
        pred_path, result_path = source_root / pred_path, source_root / result_path
        saved = pd.read_csv(pred_path, float_precision="round_trip").sort_values("source_row")
        ids = arrays[window + "__row"]
        if saved.source_row.duplicated().any():
            raise ValueError(f"Duplicate source row: {pred_path}")
        np.testing.assert_array_equal(saved.source_row, ids)
        np.testing.assert_array_equal(saved.actual, raw.loc[ids, "MONTHLY_RENT"])
        date_column = "month" if "month" in saved else "approval_month"
        np.testing.assert_array_equal(saved[date_column], raw.loc[ids, "RENT_APPROVAL_DATE"])
        prediction = saved.prediction.to_numpy(float)
        if prediction.shape != (n,) or not np.isfinite(prediction).all():
            raise ValueError("Invalid development prediction")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        arrays[window + "__" + model] = prediction
        rec = {"window": window, "stage": stage, "model": model,
               "prediction_key": window + "__" + model,
               "original_prediction_path": source(pred_path, "predictions"),
               "original_record_path": source(result_path, "metrics/configuration"),
               "record": result, "schema": columns, "macro_policy": policy,
               "seed": result.get("spec", {}).get("seed", 42),
               "target_output": "SGD rent; all component inverse transforms already applied",
               "selection": "Historical, repeatedly examined development data; no new search in this analysis",
               "history_policy": "Fit rows before query origin; same-month labels excluded; query statistics frozen at origin",
               "compare_reference": reference}
        model_path = pred_path.with_name("model.json")
        if model_path.is_file():
            rec["blend_model"] = json.loads(model_path.read_text(encoding="utf-8"))
            source(model_path, "frozen blend weights and fitting months")
        records.append(rec)

    for window, (stage, start, n) in WINDOWS.items():
        print("BUILD frozen inputs", window, start, flush=True)
        ids = raw.index[(raw.RENT_APPROVAL_DATE >= start) & (raw.RENT_APPROVAL_DATE < "2025-04")].to_numpy()
        assert len(ids) == n
        arrays[window + "__row"] = ids
        arrays[window + "__actual"] = raw.loc[ids, "MONTHLY_RENT"].to_numpy(float)
        bundle = data(start, "2025-04")
        np.testing.assert_array_equal(bundle["row_val"], ids)
        arrays[window + "__support24"] = bundle["ann_val"].ANN_BLOCK_24_R1_N.to_numpy() > 0
        arrays[window + "__support36"] = bundle["ann_val"].ANN_BLOCK_36_R1_N.to_numpy() > 0
        fitting = raw.loc[raw.RENT_APPROVAL_DATE < start]
        windows[window] = {"stage": stage, "origin": start, "end_exclusive": "2025-04",
                           "fitting_rows": len(fitting), "query_rows": n,
                           "fit_last_month": fitting.RENT_APPROVAL_DATE.max(),
                           "rent_quantiles_fit_only": fitting.MONTHLY_RENT.quantile([.1, .9]).tolist(),
                           "raw_data_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                                               for p in [ROOT / "data/train.csv", *sorted((ROOT / "data/auxiliary").glob("*.csv"))]}}
        for level in (19, 27, 47, 62):
            x, q = select_features(bundle, level)
            feature_schemas[f"{window}_ladder_{level}"] = schema(x)
            for family in ("lgb", "cat"):
                model = f"{family}_f{level}"
                add(window, model,
                    f"output/models/00_report_and_final/ladder/{stage}/{model}/predictions.csv",
                    f"output/experiment_runs/00_report_and_final/ladder/{stage}/{model}/results.json",
                    schema(x), "no query-month macro observations",
                    f"{family}_f{19 if level == 19 else {27: 19, 47: 27, 62: 47}[level]}")
            if level == 62:
                arrays[window + "__lgb_input_group"] = same_input_codes(q)
                categorical = list(q.select_dtypes(include=["category", "object", "string"]).columns)
                cat = q.copy()
                for c in categorical:
                    cat[c] = cat[c].astype("string").fillna("__MISSING__").astype(str)
                arrays[window + "__cat_input_group"] = same_input_codes(cat.fillna(-999.))
        for model, sub in [("blend_no_new", "baseline"), ("blend_PLE", "only_ft_ple"), ("F1", "only_ft_plr")]:
            add(window, model,
                f"output/models/round_10_plr_transformer_final/ensemble/{stage}/{sub}/predictions.csv",
                f"output/experiment_runs/round_10_plr_transformer_final/ensemble/{stage}/{sub}/results.json",
                {"kind": "heterogeneous component inputs", "member_order_in": "blend_model.names",
                 "recipe": "reproduction/f1/training_recipes.json; per-component input families differ"},
                "no query-month macro observations", "blend_no_new")
        for model, spec_name in [("single_PLE", "ft_ple_128_3"), ("single_PLR", "ft_plr_128_3")]:
            path = source_root / f"output/experiment_runs/round_10_plr_transformer_final/{stage}/{spec_name}/results.json"
            spec = json.loads(path.read_text())["spec"]
            x, _q, _base, _offset = inputs(bundle, spec)
            add(window, model,
                f"output/models/round_10_plr_transformer_final/{stage}/{spec_name}/predictions.csv",
                str(path.relative_to(source_root)), schema(x), "no query-month macro observations", "single_PLE")
    # The original hyperparameter/recipe declarations are captured, not refitted.
    copied_records = {}
    for relative in ["output/experiment_runs/round_09_grouped_blending/development_candidate.json",
                     "output/experiment_runs/round_10_plr_transformer_final/selection.json"]:
        path = source_root / relative
        copied_records[relative] = json.loads(path.read_text(encoding="utf-8"))
        source(path, "historical selection")
    destination.mkdir(parents=True)
    bank = destination / "validation.npz"
    np.savez_compressed(bank, **arrays)
    metadata = {"format": 1, "windows": windows, "predictions": records,
                "feature_schemas": feature_schemas, "sources": sources,
                "historical_selections": copied_records,
                "array_sha256": sha(bank), "training_performed": False,
                "source_branch": "zch-complete-proj", "source_branch_snapshot": "048eca867b6c9ad03a0a15f5a3de9f7432d3ba2f",
                "source_attribution_limit": "Ignored artifacts are a local historical snapshot. The branch commit is context, not proof every saved CSV was generated by that exact tree.",
                "feature_builder_code_sha256": {p.relative_to(ROOT).as_posix(): sha(p)
                    for p in [Path(__file__), ROOT / "analysis/experiments/reporting/report_selection_trials.py",
                              ROOT / "analysis/experiments/optimization/round_02_anniversary_history/second_wave_anniversary.py"]},
                "support_definition": "ANN_BLOCK_24_R1_N>0 and ANN_BLOCK_36_R1_N>0, from pre-origin fitting records only",
                "group_definition": "Exact equality of actual ordered 62-feature query inputs after each learner's preprocessing; evaluated only for the 62-feature controls, never assumed to be the complete F1 input."}
    (destination / "metadata.json").write_bytes((json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print("ARCHIVED", len(records), "predictions", bank.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, default=DEFAULT)
    args = parser.parse_args()
    prepare(args.source_root, args.destination)

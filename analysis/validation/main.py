"""Execute main requirement cards using frozen development predictions.

No model fitting, hyperparameter search, API use or prediction overwriting.
F2 candidate reconstruction and recent pairing are audited separately from
the unavailable annual predictions and historical screenshot association.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.pipeline.config import ROOT
from analysis.validation.prepare_inputs import sha

INPUTS = ROOT / "experiment/main/M00/inputs"


def _ensure_optional_training_stubs():
    import types
    try:
        from sklearn.base import BaseEstimator, TransformerMixin  # noqa: F401
        from sklearn.neighbors import BallTree, NearestNeighbors  # noqa: F401
    except ImportError:
        sk = sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
        sk.__path__ = getattr(sk, "__path__", [])
        sk_base = sys.modules.setdefault("sklearn.base", types.ModuleType("sklearn.base"))
        sk_nb = sys.modules.setdefault("sklearn.neighbors", types.ModuleType("sklearn.neighbors"))
        for submod, attrs in (
            ("compose", ("ColumnTransformer",)),
            ("ensemble", ("RandomForestRegressor",)),
            ("impute", ("SimpleImputer",)),
            ("linear_model", ("Ridge",)),
            ("pipeline", ("Pipeline",)),
            ("preprocessing", ("OneHotEncoder", "StandardScaler")),
        ):
            mod = sys.modules.setdefault(f"sklearn.{submod}", types.ModuleType(f"sklearn.{submod}"))
            for attr in attrs:
                setattr(mod, attr, getattr(mod, attr, object))
            setattr(sk, submod, mod)
        class BaseEstimator: pass
        class TransformerMixin:
            def fit_transform(self, X, y=None, **kwargs):
                return self.fit(X, y, **kwargs).transform(X)
        class BallTree:
            def __init__(self, data, metric="haversine"):
                self.data = np.asarray(data, dtype=float)
                self._cache_key = None
                self._cache_dists = None
                self._cache_inv = None
            def _dists(self, X):
                arr = np.asarray(X, dtype=float)
                if self._cache_key is not X:
                    uniq, inv = np.unique(arr, axis=0, return_inverse=True)
                    dlat = uniq[:, None, 0] - self.data[None, :, 0]
                    dlon = uniq[:, None, 1] - self.data[None, :, 1]
                    a = (
                        np.sin(dlat * 0.5) ** 2
                        + np.cos(uniq[:, None, 0]) * np.cos(self.data[None, :, 0]) * (np.sin(dlon * 0.5) ** 2)
                    )
                    self._cache_dists = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                    self._cache_inv = inv
                    self._cache_key = X
                return self._cache_dists, self._cache_inv
            def query(self, X, k=1):
                dists, inv = self._dists(X)
                idx = np.argsort(dists, axis=1)[:, :k]
                top_d = np.take_along_axis(dists, idx, axis=1)
                return top_d[inv], idx[inv]
            def query_radius(self, X, r, count_only=False):
                dists, inv = self._dists(X)
                if count_only:
                    counts = (dists <= float(r)).sum(axis=1)
                    return counts[inv]
                return [np.flatnonzero(dists[i] <= float(r)) for i in inv]
        sk_base.BaseEstimator = BaseEstimator
        sk_base.TransformerMixin = TransformerMixin
        sk_nb.BallTree = BallTree
        sk_nb.NearestNeighbors = object
        sk.base = sk_base
        sk.neighbors = sk_nb
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        lgb = sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))
        lgb.LGBMRegressor = object
        lgb.early_stopping = lambda *a, **k: None
    try:
        import joblib  # noqa: F401
    except ImportError:
        sys.modules.setdefault("joblib", types.ModuleType("joblib"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def metrics(actual, prediction):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Aligned, nonempty 1-D inputs required")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite actual/prediction")
    error = prediction - actual
    return {"n": len(error), "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
            "mae_sgd": float(np.mean(abs(error))), "bias_sgd": float(error.mean()),
            "sse": float(np.sum(error ** 2))}


def grouped(actual, prediction, labels, window, model, dimension):
    labels = np.asarray(labels).astype(str)
    if labels.shape != np.asarray(actual).shape:
        raise ValueError("Group labels do not align")
    total = metrics(actual, prediction)
    rows = []
    for label in sorted(np.unique(labels)):
        selected = labels == label
        score = metrics(actual[selected], prediction[selected])
        rows.append({"window": window, "model": model, "dimension": dimension, "group": label,
                     **score, "sse_share": score["sse"] / total["sse"], "sparse_n_lt_100": score["n"] < 100,
                     "diagnostic_only_uses_query_labels": dimension == "rent_band_fit_q10_q90"})
    if sum(r["n"] for r in rows) != total["n"] or not np.isclose(sum(r["sse"] for r in rows), total["sse"], rtol=1e-12, atol=1e-6):
        raise AssertionError("Disjoint groups must reconcile exactly")
    return rows


def validate_prediction(raw, ids, archived_actual, prediction, start, end, fit_last):
    expected = raw.index[(raw.RENT_APPROVAL_DATE >= start) & (raw.RENT_APPROVAL_DATE < end)].to_numpy()
    if len(np.unique(ids)) != len(ids) or not np.array_equal(expected, ids):
        raise ValueError("Missing, duplicate or reordered source row IDs")
    if not np.array_equal(raw.loc[ids, "MONTHLY_RENT"].to_numpy(float), archived_actual):
        raise ValueError("Labels do not match the original source rows")
    if not fit_last < start:
        raise ValueError("Fitting/weight cutoff reaches query origin")
    return metrics(archived_actual, prediction)


def audit_data(raw, test):
    from analysis.pipeline.features import build_feature_frame
    frames = {"train": raw, "test": test}
    quality = []
    for name, frame in frames.items():
        transformed = build_feature_frame(frame)
        if len(frame) != len(transformed) or not frame.index.equals(transformed.index):
            raise AssertionError("Static joins changed row count/order")
        quality.append({"split": name, "rows": len(frame), "raw_duplicate_rows": int(frame.duplicated().sum()),
                        "missing_cells": int(frame.isna().sum().sum()),
                        "constant_columns": [c for c in frame if frame[c].nunique(dropna=False) == 1],
                        "negative_raw_age": int((pd.to_datetime(frame.RENT_APPROVAL_DATE).dt.year - frame.LEASE_COMMENCE_DATE < 0).sum()),
                        "joined_rows": len(transformed), "rows_deleted": 0,
                        "date_min": frame.RENT_APPROVAL_DATE.min(), "date_max": frame.RENT_APPROVAL_DATE.max(),
                        "model_age_missing": int(transformed.FLAT_AGE_YEARS.isna().sum()),
                        "missing_building_coordinates": int(transformed.LATITUDE.isna().sum()),
                        "canonical_flat_types": transformed.FLAT_TYPE_CLEAN.value_counts().to_dict()})
    assert quality[0]["rows"] == 150000 and quality[1]["rows"] == 50000
    assert quality[0]["raw_duplicate_rows"] == 425 and quality[1]["raw_duplicate_rows"] == 5508
    return {"splits": quality, "test_target_absent": "MONTHLY_RENT" not in test,
            "overlapping_test_train_month_rows": int((test.RENT_APPROVAL_DATE <= raw.RENT_APPROVAL_DATE.max()).sum()),
            "preprocessing_policy": "Keep all rows and duplicates; no label-based deletion. Negative age is missing. Categories fit on fitting rows; unseen values remain unknown.",
            "facility_time_policy": "Supplied static facility snapshots, including STATUS filtering; not certified historical openings. No strict historical availability claim.",
            "strict_historical_facility_evaluation": "pending; X08 requires verified opening dates and matched refits"}


def validate_scenario(raw, test, metadata):
    policy = json.loads((ROOT / 'docs/methods/prediction-scenario.json').read_text('utf-8'))
    for window, setting in policy['development_windows'].items():
        origin, end = setting['query_start'], setting['query_end_exclusive']
        assert setting['fit_end_exclusive'] == origin == metadata['windows'][window]['origin']
        assert int((raw.RENT_APPROVAL_DATE < origin).sum()) == setting['fit_rows']
        assert int(((raw.RENT_APPROVAL_DATE >= origin) & (raw.RENT_APPROVAL_DATE < end)).sum()) == setting['query_rows']
    full = policy['full_submission']
    assert len(raw) == full['supervised_rows'] and len(test) == full['query_rows']
    assert raw.RENT_APPROVAL_DATE.max() == full['latest_supervised_month'] == full['same_month']
    assert int((test.RENT_APPROVAL_DATE == full['same_month']).sum()) == full['same_month_test_rows']
    assert not policy['query_targets_available_to_fitting']
    assert not policy['F1_query_month_macro_observations_used']
    return policy


def run(output_root=None):
    meta = json.loads((INPUTS / "metadata.json").read_text(encoding="utf-8"))
    bank = INPUTS / "validation.npz"
    if sha(bank) != meta["array_sha256"]:
        raise ValueError("Frozen prediction archive changed")
    data_paths = [ROOT / "data/train.csv", ROOT / "data/test.csv", ROOT / "data/example-submission.csv",
                  *sorted((ROOT / "data/auxiliary").glob("*.csv"))]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in data_paths}
    from analysis.validation.final_pair import audit as pair_audit, write_evidence
    final_pair = pair_audit()
    code_paths = [*sorted(Path(__file__).parent.glob('*.py')),
                  ROOT / 'analysis/pipeline/config.py', ROOT / 'docs/methods/prediction-scenario.json']
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in code_paths}
    signature = hashlib.sha256(json.dumps([meta["array_sha256"], before, code, final_pair['source_sha256']], sort_keys=True).encode()).hexdigest()[:16]
    out = (Path(output_root) if output_root else ROOT / "output/planning_validation/main" / signature).resolve()
    if not out.is_relative_to(ROOT / "output") or out == ROOT / "output":
        raise ValueError("Generated evaluations must stay under this repository's output/.")
    if out.exists():
        raise FileExistsError("Run directory exists; use a new --output-dir to keep earlier records intact.")
    from analysis.validation.evidence_audit import audit as evidence_audit
    from analysis.validation.feature_catalog import catalog
    from analysis.validation.refit_audit import audit as refit_audit
    imported, spatial, facility_bins, cleaning = evidence_audit()
    feature_catalog, refits = catalog(), refit_audit()
    raw = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    scenario = validate_scenario(raw, test, meta)
    arrays = np.load(bank, allow_pickle=False)
    registry, recalculated, checks, group_rows, monthly, identical, deltas = [], [], [], [], [], [], []
    all_scores, support_audit = {}, {}
    for window, protocol in meta["windows"].items():
        ids, actual = arrays[window + "__row"], arrays[window + "__actual"]
        start, end = protocol["origin"], protocol["end_exclusive"]
        if {p.replace("\\", "/"): h for p, h in protocol["raw_data_sha256"].items()} != {p: h for p, h in before.items() if p != "data/test.csv" and p != "data/example-submission.csv"}:
            raise ValueError("Historical input hashes differ from current raw data")
        fit = raw.loc[raw.RENT_APPROVAL_DATE < start]
        np.testing.assert_allclose(fit.MONTHLY_RENT.quantile([.1, .9]), protocol["rent_quantiles_fit_only"], rtol=0, atol=0)
        months = raw.loc[ids, "RENT_APPROVAL_DATE"].to_numpy()
        # Independently rebuild support using only the pre-origin source. Query
        # rents are dropped, so neither labels nor future history enter groups.
        _ensure_optional_training_stubs()
        from analysis.pipeline.features import build_feature_frame
        from analysis.experiments.second_wave_anniversary import AnniversaryHistory
        history = AnniversaryHistory().fit(build_feature_frame(fit))
        support = history.transform(build_feature_frame(raw.loc[ids].drop(columns="MONTHLY_RENT")))
        for lag in (24, 36):
            np.testing.assert_array_equal(support[f"ANN_BLOCK_{lag}_R1_N"].to_numpy() > 0, arrays[f"{window}__support{lag}"])
        support_audit[window] = {"query_rows": len(ids), "fitting_rows": len(fit),
                                "history_source_last_month": fit.RENT_APPROVAL_DATE.max(),
                                "query_rent_removed_before_feature_build": True,
                                "support24_rows": int(arrays[window + "__support24"].sum()),
                                "support36_rows": int(arrays[window + "__support36"].sum()),
                                "support_matches_independent_rebuild": True}
        for rec in [r for r in meta["predictions"] if r["window"] == window]:
            name, prediction = rec["model"], arrays[rec["prediction_key"]]
            score = validate_prediction(raw, ids, actual, prediction, start, end, protocol["fit_last_month"])
            record = rec["record"]
            for source_path, recorded_hash in record.get('source_sha256', {}).items():
                normalized = source_path.replace('\\', '/')
                if normalized in before and before[normalized] != recorded_hash:
                    raise ValueError(f"Recorded source hash differs for {name}: {normalized}")
            differences = {key: abs(score[key] - record[key]) for key in ["rmse_sgd", "mae_sgd", "bias_sgd"]}
            if max(differences.values()) > 1e-6:
                raise AssertionError(f"Historical metric mismatch: {window}/{name}: {differences}")
            blend = rec.get("blend_model")
            if blend and not blend["fit_last_month"] < start:
                raise ValueError("Blend fitted on query-period labels")
            from analysis.validation.feature_catalog import resolved_record_schema
            resolved_schema = resolved_record_schema(rec, meta)
            schema_digest = hashlib.sha256(json.dumps(resolved_schema, sort_keys=True).encode()).hexdigest()
            registry.append({"window": window, "model": name, "fitting_rows": len(fit), "fit_end": protocol["fit_last_month"],
                             "query_start": start, "query_end_exclusive": end, "n": score["n"],
                             "ordered_schema_sha256": schema_digest,
                             "schema_metadata_correction": name in ('single_PLE', 'single_PLR'),
                             "schema_reference": "experiment/main/M03/feature_schemas/schemas.json",
                             "prediction_scenario": "supplied-snapshot-fixed-origin-v1",
                             "schema_kind": "heterogeneous; inspect components" if isinstance(rec["schema"], dict) else "ordered learner input",
                             "target_transform": rec["target_output"], "history_policy": rec["history_policy"],
                             "macro_policy": rec["macro_policy"], "seed": rec["seed"], "selection_stage": rec["selection"],
                             "blend_fit_last_month": blend["fit_last_month"] if blend else None,
                             "prediction_array_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                             "prediction_original_path": rec["original_prediction_path"],
                             "data_sha256": before["data/train.csv"],
                             "source_branch_context": meta["source_branch_snapshot"], "generator_commit_verified": False})
            recalculated.append({"window": window, "model": name, **score,
                                 **{"difference_" + k: v for k, v in differences.items()}})
            checks.append({"window": window, "model": name, "rows": len(ids), "duplicate_ids": 0,
                           "id_label_date_mismatches": 0, "metric_max_abs_difference": max(differences.values()),
                           "history_query_origin_checked": True, "historical_prediction_generation_reexecuted": False})
            all_scores[window, name] = score
        for family in ("lgb", "cat"):
            for level, prior in [(27, 19), (47, 27), (62, 47)]:
                schema_a = meta["feature_schemas"][f"{window}_ladder_{level}"]
                schema_b = meta["feature_schemas"][f"{window}_ladder_{prior}"]
                if len(schema_a) != level or len(schema_b) != prior or not {s['name'] for s in schema_b} < {s['name'] for s in schema_a}:
                    raise AssertionError("Feature ladder is not nested")
                a, b = all_scores[window, f"{family}_f{level}"], all_scores[window, f"{family}_f{prior}"]
                deltas.append({"task": "M03", "window": window, "candidate": f"{family}_f{level}",
                               "control": f"{family}_f{prior}", "delta_rmse_candidate_minus_control": a["rmse_sgd"] - b["rmse_sgd"],
                               "delta_mae": a["mae_sgd"] - b["mae_sgd"], "same_rows_and_within_family_settings": True})
        for candidate, control in [("single_PLR", "single_PLE"), ("blend_PLE", "blend_no_new"), ("F1", "blend_no_new")]:
            a, b = all_scores[window, candidate], all_scores[window, control]
            rec_a, rec_b = [next(r for r in meta['predictions'] if r['window'] == window and r['model'] == name)
                            for name in (candidate, control)]
            if candidate == "single_PLR" and resolved_record_schema(rec_a, meta) != resolved_record_schema(rec_b, meta):
                raise AssertionError("PLE/PLR actual ordered inputs differ")
            if candidate != "single_PLR" and rec_a['blend_model']['spec'] != rec_b['blend_model']['spec']:
                raise AssertionError("Blend inclusion comparison changed its fitting rule")
            deltas.append({"task": "M05", "window": window, "candidate": candidate, "control": control,
                           "delta_rmse_candidate_minus_control": a["rmse_sgd"] - b["rmse_sgd"],
                           "delta_mae": a["mae_sgd"] - b["mae_sgd"], "same_query_rows": True,
                           "candidate_selected_epochs": rec_a['record'].get('epochs'),
                           "control_selected_epochs": rec_b['record'].get('epochs'),
                           "qualification": "Blend rule fixed; weights were fitted earlier for each included member set. Epoch and embedding parameter budgets differ."})
        support_labels = np.array([f"24={bool(a)}|36={bool(b)}" for a, b in zip(arrays[window + "__support24"], arrays[window + "__support36"])])
        q10, q90 = protocol["rent_quantiles_fit_only"]
        bands = np.where(actual < q10, "low_below_fit_q10", np.where(actual > q90, "high_above_fit_q90", "middle_inclusive"))
        for name in ["lgb_f62", "cat_f62", "F1"]:
            prediction = arrays[window + "__" + name]
            for dimension, labels in [("month", months), ("flat_type", raw.loc[ids, "FLAT_TYPE"].str.lower().str.strip().str.replace(r"\s*-\s*", "-", regex=True)),
                                      ("support24_36", support_labels), ("rent_band_fit_q10_q90", bands)]:
                rows = grouped(actual, prediction, labels, window, name, dimension)
                group_rows.extend(rows)
                if dimension == "month":
                    monthly.extend(rows)
            if name != "F1":
                codes = arrays[window + ("__lgb_input_group" if name.startswith("lgb") else "__cat_input_group")]
                frame = pd.DataFrame({"code": codes, "actual": actual, "prediction": prediction})
                groups = frame.groupby("code", dropna=False).agg(n=("actual", "size"), label_min=("actual", "min"), label_max=("actual", "max"),
                                                                 label_mean=("actual", "mean"), prediction_min=("prediction", "min"), prediction_max=("prediction", "max"))
                group_mean = frame.code.map(groups.label_mean).to_numpy()
                within = float(np.sum((actual - group_mean) ** 2))
                identical.append({"window": window, "model": name, "exact_input_groups": len(groups),
                                  "repeated_groups": int((groups.n > 1).sum()),
                                  "rows_in_repeated_groups": int(groups.loc[groups.n > 1, "n"].sum()),
                                  "repeated_groups_with_label_conflict": int(((groups.n > 1) & (groups.label_max > groups.label_min)).sum()),
                                  "within_validation_group_sse": within,
                                  "within_validation_group_rmse": float(np.sqrt(within / len(actual))),
                                  "maximum_prediction_spread_within_equal_input": float((groups.prediction_max - groups.prediction_min).max()),
                                  "interpretation": "Descriptive conditional dispersion on this validation set, not a global error floor or F1 input-equivalence claim."})
    for item in identical:
        if item["maximum_prediction_spread_within_equal_input"] > 1e-6:
            raise AssertionError("Prediction spread contradicts archived exact-input group definition")
    quality = audit_data(raw, test)
    out.mkdir(parents=True)
    table(out / "M00/protocol_registry.csv", registry)
    table(out / "M00/metric_recalculation.csv", recalculated)
    dump(out / "M00/alignment_checks.json", {"all_passed": True, "checks": checks,
         "limitation": meta["source_attribution_limit"], "Y_J_overlap_rows": len(set(arrays["Y__row"]) & set(arrays["J__row"])),
         "Y_J_are_independent": False, "F2_eligible_for_paired_comparison": True,
         "F2_eligible_windows": final_pair['protocol']['eligible_windows'],
         "F2_identical_information_policies_verified": False})
    dump(out / "M00/source_manifest.json", {"raw_data": before, "code": code, "frozen_archive": {"path": str(bank.relative_to(ROOT)), "sha256": sha(bank)},
         "historical_source_files": meta["sources"], "source_branch_snapshot": meta["source_branch_snapshot"], "generator_commit_verified": False})
    dump(out / "M01/data_policy_audit.json", quality)
    dump(out / "M00/fresh_F1_validation_audit.json", refits)
    dump(out / "M01/imported_evidence_audit.json", imported)
    table(out / "M01/raw_adjusted_comparison.csv", spatial.to_dict('records'))
    table(out / "M01/facility_bin_controls.csv", facility_bins.to_dict('records'))
    table(out / "M01/matched_cleaning_comparison.csv", cleaning.to_dict('records'))
    dump(out / "M03/feature_schemas.json", feature_catalog)
    dump(out / "M01/prediction_scenario.json", scenario)
    table(out / "M03/feature_ladder_deltas.csv", [r for r in deltas if r["task"] == "M03"])
    table(out / "M04/error_groups.csv", group_rows)
    table(out / "M04/monthly_errors.csv", monthly)
    table(out / "M04/identical_input_diagnostics.csv", identical)
    dump(out / "M04/feature_support_audit.json", {"windows": support_audit, "support_definition": meta["support_definition"],
         "group_definition": meta["group_definition"], "group_accounting_passed": True,
         "rent_bands_are_posthoc_diagnostics_not_deployment_features": True})
    table(out / "M05/controlled_structure_deltas.csv", [r for r in deltas if r["task"] == "M05"])
    dump(out / "M05/selection_audit.json", {"historical_choices": meta["historical_selections"],
         "conclusion_scope": "Retrospective development comparison. Fixed blending rule is not identical numeric weights across different member sets. Architectural epoch/parameter budgets differ.",
         "same_27_block_separate_from_62_block": True, "new_search_performed": False})
    pending = {"M07": {"status": "partial", "missing": final_pair['manifest']['missing']},
               "M08": {"status": "partial", "completed_windows": ["J"], "missing": final_pair['protocol']['missing']}}
    write_evidence(out, final_pair)
    replay_results = []
    for script in ["verify.py", "reproduce.py"]:
        command = [sys.executable, "-X", "utf8", str(ROOT / "reproduction/f1" / script)]
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / "M09").mkdir(exist_ok=True)
        (out / "M09" / (script + ".log")).write_bytes(result.stdout.replace(b"\r\n", b"\n"))
        if result.returncode:
            raise RuntimeError(f"F1 {script} failed; see run log")
        replay_results.append({"command": ["python", "reproduction/f1/" + script], "returncode": result.returncode})
    replay = json.loads((ROOT / "output/review_f1_reproduced/verification_run.json").read_text(encoding="utf-8"))
    dump(out / "M09/verification.json", {"F1": {"status": "passed", "checks": replay_results, "replay": replay,
         "full_training_this_run": False, "previous_full_training_evidence": "review/checks/push-validation/training_run.json"}, "F2": final_pair['reconstruction']})
    plots(out, monthly, group_rows)
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in data_paths}
    if before != after:
        raise AssertionError("Raw inputs changed")
    status = {"M00": "F1 fresh Y/J generation verified; 26 old artifacts audited; F2 recent IDs/labels and full-candidate component identities checked",
              "M01": "completed for supplied facility snapshot scenario; strict historical facilities remain X08",
              "M02": "archived and previously refitted; no new training required",
              "M03": "completed frozen 16-fit paired recalculation", "M04": "completed for F1 and two 62-input controls on Y/J",
              "M05": "completed controlled recalculation; annual PLR fusion regression preserved",
              "M06": "archived three-way submission selection; not relabelled as independent test",
              "M07": "candidate composition/CSV reconstructed; ordered dictionary and screenshot association pending",
              "M08": "recent J paired scores complete; annual Y and full information-policy equivalence pending",
              "M09": "F1 and F2 component replay passed; full F2 fits/inference recorded in source audit, not rerun on this machine"}
    summary = {"run_id": signature, "tasks": status, "new_models_fitted": 0, "predictions_audited": len(checks),
               "group_rows": len(group_rows), "raw_inputs_unchanged": before == after,
               "fresh_F1_validation_models_previously_trained": 30,
               "fresh_F1_generation_audit_passed": refits['all_passed'],
               "unified_feature_schemas_verified": feature_catalog['all_passed'],
               "imported_M01_evidence_verified": imported['all_passed'],
               "prediction_scenario": "supplied-snapshot-fixed-origin-v1",
               "all_numeric_checks_passed": True, "all_main_P0_completed": False,
               "F2_component_reconstruction_passed": True, "F1_F2_paired_windows": ["J"],
               "remaining_required_inputs": pending, "results_are_development_not_independent_test": True}
    dump(out / "summary.json", summary)
    print(json.dumps({"output": str(out), **summary}, ensure_ascii=False, indent=2))
    return out


def plots(out, monthly, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for window in ("Y", "J"):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        data = pd.DataFrame([r for r in monthly if r["window"] == window])
        for name, part in data.groupby("model", sort=True):
            part = part.sort_values("group")
            ax.plot(part.group, part.rmse_sgd, marker="o", label=name)
        ax.set(ylabel="RMSE (SGD)", title=f"{window}: fixed development predictions by month")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"M04/monthly_{window}.png", dpi=160)
        plt.close(fig)
        data = pd.DataFrame([r for r in groups if r["window"] == window and r["dimension"] == "support24_36"])
        fig, ax = plt.subplots(figsize=(9, 4.2))
        data.pivot(index="group", columns="model", values="rmse_sgd").plot.bar(ax=ax)
        ax.legend(title="model", loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set(ylabel="RMSE (SGD)", title=f"{window}: pre-origin cohort support groups")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out / f"M04/support_{window}.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == "__main__":
    main()

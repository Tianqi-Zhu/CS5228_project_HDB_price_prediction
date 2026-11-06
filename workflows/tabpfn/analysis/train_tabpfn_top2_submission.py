"""Full-context inference and local Kaggle exports for equal ABCDE and weighted ABDE.

Reuse audited full-data A/B/D/E fits; fit only the missing frozen C view.
No weight fitting, model selection or Kaggle upload takes place here.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT, DATA, REPORTS
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable, write_json, hashlib_array
from analysis.compare_tabpfn_hierarchical_latent import weight_digest
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.temporal_feature_views import temporal_view, LEVEL_COLUMNS
from analysis.experiments.feature_diversified_ensemble import combine_predictions
from analysis import train_tabpfn_feature_ensemble_submission as prior
from analysis import train_tabpfn_diagscale_mse_submission as diag

OUT = REPORTS / "tabpfn_top2_final_submission"
CACHE = ROOT / ".cache/tabpfn_top2_final_submission"
EXPERIMENT = REPORTS / "tabpfn_abcde_ensemble"
RELATIVE = REPORTS / "tabpfn_temporal_view_ensembles_80_20"
FEATURES = CACHE / "features_C.joblib"
MEMBERS = list("ABCDE")
BATCH = 200
FILES = {
    "ABCDE_equal": "submission_tabpfn_ABCDE_equal_150k.csv",
    "ABDE": "submission_tabpfn_ABDE_weighted_150k.csv",
}


def verify_sources():
    s = json.loads((OUT / "settings.json").read_text())
    for group in ("source_sha256", "code_sha256"):
        for name, digest in s[group].items():
            assert sha256(ROOT / name) == digest, name
    return s


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    prior.verify_sources()
    diag.verify_sources()
    b = prior.full_base()
    weights = pd.read_csv(EXPERIMENT / "weights.csv", float_precision="round_trip")
    row = weights.loc[(weights.origin == "2024-06") & (weights.group == "ABDE")]
    assert len(row) == 1
    learned = row[MEMBERS].to_numpy(float)[0]
    assert learned[2] == 0 and (learned >= 0).all() and np.isclose(learned.sum(), 1)
    scores = pd.read_csv(EXPERIMENT / "scores.csv", float_precision="round_trip")
    scores = scores.loc[scores.origin == "2024-06"].set_index("model")
    audit = json.loads((EXPERIMENT / "independent_audit.json").read_text())
    assert audit["verified"] and audit["weights_earlier_only"]
    sources = [DATA / "train.csv", DATA / "test.csv", DATA / "example-submission.csv",
               CHECKPOINT, prior.FEATURE_PATH, EXPERIMENT / "weights.csv", EXPERIMENT / "scores.csv",
               EXPERIMENT / "independent_audit.json", EXPERIMENT / "candidate_gate.json",
               RELATIVE / "features_2024-06_relative_history.json",
               prior.OUT / "settings.json", prior.OUT / "verification.json",
               prior.OUT / "final_verification.json", prior.SUBMISSION,
               diag.OUT / "settings.json", diag.OUT / "verification.json",
               diag.OUT / "training.json", diag.OUT / "inference_settings.json",
               diag.OUT / "inference_timing.json", diag.SUBMISSION, diag.ADAPTER,
               diag.CACHE / "predictions.npy"]
    for member in ("baseline", "temporal", "spatial"):
        sources += [prior.OUT / f"predictions_{member}.csv", prior.OUT / f"timing_{member}.json"]
    codes = [Path(__file__), ROOT / "analysis/experiments/temporal_feature_views.py",
             ROOT / "analysis/experiments/feature_diversified_ensemble.py",
             Path(prior.__file__), Path(diag.__file__)]
    immutable(OUT / "settings.json", dict(
        purpose="User-requested submission candidates: two lowest observed development RMSE ensembles",
        selection_limit="Retrospective development ranking; neither candidate was the earlier-selected family. User explicitly requested preparation; original experimental gate and artifacts remain unchanged.",
        member_order=MEMBERS,
        members=dict(A="MSE-only DiagScale77; one view; fresh full-data adapter already fitted",
                     B="frozen baseline77; two views", C="frozen relative-history77; two views",
                     D="frozen seasonal84; two views", E="frozen spatial89; two views"),
        weights={"ABCDE_equal": [.2] * 5, "ABDE": learned.tolist()},
        source_development_rmse_sgd={name: float(scores.loc[name, "rmse_sgd"]) for name in FILES},
        weight_policy="Keep original fixed weights; no reoptimization on training residuals or test predictions",
        context_rows=150000, test_rows=50000, batch_size=BATCH, random_state=42,
        C_policy="Frozen V3_5_FAST; two estimators; native default estimator batching; all rows; no subsampling or adaptation",
        targets="Original rent minus unchanged train baseline; add original query baseline once per member",
        full_data_reuse="Reuse verified full150000 fits and full precision predictions for A/B/D/E; newly fit C",
        test_targets_available=False, kaggle_upload=False,
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        code_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in codes},
        packages={p: version(p) for p in ("numpy", "pandas", "torch", "tabpfn")},
    ))
    verify_sources()
    if FEATURES.exists():
        load_C()
        print("REUSED verified C feature bundle", flush=True)
        return
    expected = json.loads((RELATIVE / "features_2024-06_relative_history.json").read_text())
    views = {}
    checks = []
    for split in ("train", "query"):
        x = b[f"X_{split}"]
        z = temporal_view(x, b[f"{split}_month"], b[f"{split}_baseline"], "relative_history")
        assert list(z) == expected["feature_names"] and z.shape == x.shape
        unchanged = [c for c in x if c not in LEVEL_COLUMNS]
        pd.testing.assert_frame_equal(z[unchanged], x[unchanged])
        for column in LEVEL_COLUMNS:
            np.testing.assert_array_equal(z[column + "_REL_BASE"].isna(), x[column].isna())
            np.testing.assert_allclose(z[column + "_REL_BASE"],
                                       x[column].to_numpy(float) - b[f"{split}_baseline"],
                                       atol=0, rtol=0, equal_nan=True)
        assert not np.isinf(z.select_dtypes("number").to_numpy(float)).any()
        views[f"X_{split}"] = z
        checks.append(dict(split=split, rows=len(z), unchanged_columns=len(unchanged),
                           relative_columns=len(LEVEL_COLUMNS), exact_baseline_subtraction=True,
                           original_missingness_preserved=True))
    joblib.dump({**b, **views}, FEATURES, compress=3)
    write_json(OUT / "features_C.json", dict(
        sha256=sha256(FEATURES), feature_names=list(views["X_train"]), checks=checks,
        context_rows=150000, test_rows=50000, features=77,
        context_ids_sha256=hashlib_array(b["X_train"].index),
        test_ids_sha256=hashlib_array(b["X_query"].index),
        original_targets_and_baselines_preserved=True,
        rowwise_representation_only=True, no_new_temporal_data_read=True,
    ))
    print("PREPARED C features for150000 context and50000 test rows", flush=True)


def load_C():
    manifest = json.loads((OUT / "features_C.json").read_text())
    assert sha256(FEATURES) == manifest["sha256"]
    b = joblib.load(FEATURES)
    assert b["X_train"].shape == (150000, 77) and b["X_query"].shape == (50000, 77)
    return b


def predict_C():
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings

    verify_sources()
    assert settings.tabpfn.max_batched_estimator_rows is None
    assert settings.tabpfn.max_batched_estimator_cells is None
    path, meta_path = OUT / "predictions_C.csv", OUT / "timing_C.json"
    if path.exists() and meta_path.exists():
        assert sha256(path) == json.loads(meta_path.read_text())["predictions_sha256"]
        print("REUSED complete C inference", flush=True)
        return
    b = load_C()
    x, q = b["X_train"], b["X_query"]
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda", n_estimators=2,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        n_preprocessing_jobs=1, inference_config={"SUBSAMPLE_SAMPLES": None},
        categorical_features_indices=[i for i, c in enumerate(x) if isinstance(x[c].dtype, pd.CategoricalDtype)])
    progress = CACHE / "partial_C.npy"
    values = np.load(progress) if progress.exists() else np.full(50000, np.nan)
    assert values.shape == (50000,)
    resumed = int(np.isfinite(values).sum())

    def forbidden(*a, **kw):
        raise AssertionError("Frozen C backbone: gradients prohibited")

    started = time.perf_counter()
    with patch("torch.Tensor.backward", forbidden), patch("torch.autograd.backward", forbidden), patch("torch.autograd.grad", forbidden):
        model.fit(x, np.asarray(b["y_train"]) - b["train_baseline"])
        assert type(model) is TabPFNRegressor and model.n_train_samples_ == 150000
        assert model.n_estimators_ == 2 and model.inference_config_.SUBSAMPLE_SAMPLES is None
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None:
            assert all(len(i) == len(np.unique(i)) == 150000 for i in indices)
        for m in model.models_:
            for parameter in m.parameters():
                parameter.requires_grad_(False)
        fitted = time.perf_counter()
        before = weight_digest(model)
        versions = [p._version for m in model.models_ for p in m.parameters()]
        replay_error = None
        if np.isfinite(values[:BATCH]).all():
            replay = np.asarray(model.predict(q.iloc[:BATCH], output_type="mean"), float) + b["query_baseline"][:BATCH]
            replay_error = float(np.max(np.abs(replay - values[:BATCH])))
            np.testing.assert_allclose(replay, values[:BATCH], rtol=0, atol=1e-5)
        prediction_started = time.perf_counter()
        for i in range(0, len(q), BATCH):
            stop = min(i + BATCH, len(q))
            if np.isfinite(values[i:stop]).all():
                continue
            torch.cuda.empty_cache()
            values[i:stop] = np.asarray(model.predict(q.iloc[i:stop], output_type="mean"), float) + b["query_baseline"][i:stop]
            assert np.isfinite(values[i:stop]).all()
            temporary = progress.with_suffix(".part.npy")
            np.save(temporary, values)
            temporary.replace(progress)
            if i % 2000 == 0 or stop == len(q):
                print(f"C PREDICT {stop:,}/50,000", flush=True)
        predicted = time.perf_counter()
        after = weight_digest(model)
        assert before == after and versions == [p._version for m in model.models_ for p in m.parameters()]
    assert (values > 0).all()
    pd.DataFrame(dict(Id=q.index, month=b["query_month"], Predicted=values)).to_csv(path, index=False, float_format="%.17g")
    write_json(meta_path, dict(
        member="C", context_rows=150000, test_rows=50000, features=77, n_estimators=2,
        seed=42, batch_size=BATCH, subsampling=False, native_default_estimator_batching=True,
        fit_seconds=fitted-started, predict_seconds=predicted-prediction_started,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        frozen_digest_before=before, frozen_digest_after=after, parameters_unchanged=True,
        no_adapters=True, gradients_prohibited=True, resumed_rows=resumed,
        resume_replay_error_sgd=replay_error, features_sha256=sha256(FEATURES),
        training_context_ids_sha256=hashlib_array(x.index), test_ids_sha256=hashlib_array(q.index),
        predictions_sha256=sha256(path), baseline_added_once=True))
    print("C inference complete", flush=True)


def aligned_members(include_C=True):
    sample = pd.read_csv(DATA / "example-submission.csv")
    test = pd.read_csv(DATA / "test.csv")
    assert sample.shape == (50000, 2) and list(sample) == ["Id", "Predicted"]
    assert sample.Id.is_unique and "MONTHLY_RENT" not in test
    np.testing.assert_array_equal(sample.Id, test.index)
    f = pd.DataFrame(dict(Id=sample.Id, month=test.RENT_APPROVAL_DATE))
    a = np.load(diag.CACHE / "predictions.npy")
    a_meta = json.loads((diag.OUT / "verification.json").read_text())
    training = json.loads((diag.OUT / "training.json").read_text())
    inference = json.loads((diag.OUT / "inference_timing.json").read_text())
    assert a_meta["context_rows"] == training["training_pool_rows"] == inference["context_rows"] == 150000
    assert sha256(diag.ADAPTER) == training["adapter_sha256"] == a_meta["adapter_sha256"]
    old_a = pd.read_csv(diag.SUBMISSION)
    np.testing.assert_array_equal(old_a.Id, sample.Id)
    np.testing.assert_allclose(old_a.Predicted, a, rtol=0, atol=1e-6)
    f["A"] = a
    paths = {"B": (prior.OUT, "baseline"), "D": (prior.OUT, "temporal"), "E": (prior.OUT, "spatial")}
    if include_C:
        paths["C"] = (OUT, "C")
    for letter, (directory, name) in paths.items():
        path = directory / f"predictions_{name}.csv"
        meta = json.loads((directory / f"timing_{name}.json").read_text())
        assert sha256(path) == meta["predictions_sha256"]
        assert meta["context_rows"] == 150000 and meta["n_estimators"] == 2 and not meta["subsampling"]
        assert meta["parameters_unchanged"] and meta["frozen_digest_before"] == meta["frozen_digest_after"]
        p = pd.read_csv(path, float_precision="round_trip")
        assert len(p) == 50000 and p.Id.is_unique
        np.testing.assert_array_equal(p.Id, f.Id)
        np.testing.assert_array_equal(p.month, f.month)
        f[letter] = p.Predicted
    letters = MEMBERS if include_C else list("ABDE")
    assert np.isfinite(f[letters]).all().all() and (f[letters] > 0).all().all()
    return f


def export(include_C=True):
    s = verify_sources()
    f = aligned_members(include_C)
    results = {}
    summaries = []
    groups = FILES if include_C else {"ABDE": FILES["ABDE"]}
    for group, name in groups.items():
        weights = np.asarray(s["weights"][group])
        active = np.flatnonzero(weights > 0)
        p = f[[MEMBERS[i] for i in active]].to_numpy(float)
        values = combine_predictions(p, weights[active])
        assert (values >= p.min(axis=1)-1e-9).all() and (values <= p.max(axis=1)+1e-9).all()
        path = OUT / name
        if path.exists():
            old = pd.read_csv(path)
            np.testing.assert_array_equal(old.Id, f.Id)
            np.testing.assert_allclose(old.Predicted, values, atol=1e-6, rtol=0)
        else:
            tmp = path.with_suffix(".part.csv")
            pd.DataFrame(dict(Id=f.Id.astype(int), Predicted=values)).to_csv(tmp, index=False, float_format="%.6f")
            tmp.replace(path)
        saved = pd.read_csv(path)
        assert saved.shape == (50000, 2) and list(saved) == ["Id", "Predicted"] and saved.Id.is_unique
        np.testing.assert_array_equal(saved.Id, f.Id)
        assert np.isfinite(saved.Predicted).all() and (saved.Predicted > 0).all()
        rounding = float(np.max(np.abs(saved.Predicted-values)))
        assert rounding <= 1e-6
        f[group] = values
        monthly = pd.DataFrame(dict(month=f.month, prediction=values)).groupby("month").prediction.agg(["size", "mean", "min", "max"]).reset_index()
        monthly.insert(0, "ensemble", group)
        summaries.append(monthly)
        results[group] = dict(file=name, sha256=sha256(path), rows=50000, weights=s["weights"][group],
                              source_development_rmse_sgd=s["source_development_rmse_sgd"][group],
                              minimum_sgd=float(values.min()), maximum_sgd=float(values.max()), mean_sgd=float(values.mean()),
                              maximum_csv_rounding_error_sgd=rounding)
        print("READY", path, flush=True)
    if not include_C:
        return
    f = f[["Id", "month", *MEMBERS, *FILES]]
    f.to_csv(OUT / "predictions_members.csv", index=False, float_format="%.17g")
    pd.concat(summaries, ignore_index=True).to_csv(OUT / "test_month_summary.csv", index=False)
    verify_sources()
    write_json(OUT / "verification.json", dict(
        verified=True, submissions=results, all_members_use150000_context_rows=True,
        all50000_sample_ids_exactly_aligned=True, finite_positive_predictions=True,
        convex_hull_checks_passed=True, baseline_added_once_per_member=True,
        original_weights_preserved=True, original_submissions_and_sources_unchanged=True,
        test_labels_available=False, test_rmse_available=False, uploaded_to_kaggle=False,
        predictions_members_sha256=sha256(OUT / "predictions_members.csv")))
    report = ["# Two full-data TabPFN Kaggle submissions", "",
              "Both CSVs contain50,000 rows with exactly `Id,Predicted`, in the supplied sample/test order. All rents are finite and positive.", "",
              "| Candidate | Original development RMSE (SGD) | A/B/C/D/E weights | Upload CSV |",
              "| --- | --- | --- | --- |"]
    for group, name in FILES.items():
        report.append(f"| {group} | {s['source_development_rmse_sgd'][group]:.6f} | {s['weights'][group]} | [{name}]({name}) |")
    report += ["", "These are the two lowest observed later-development RMSEs from the previous ensemble comparison, prepared at the user's explicit request. That comparison repeatedly reused development labels; its earlier-selected family was CDE, and its independent-promotion gate failed. The original gate remains unchanged. Neither listed score measures this full-data refit or the hidden Kaggle test, and the small difference between the two candidates is uncertain.", "",
               "Every member uses all150,000 labelled training transactions. A reuses the verified fresh one-epoch full-data MSE-only DiagScale adapter and its one-view full-precision predictions. B/D/E reuse audited full-data frozen77/84/89-feature fits and two-view predictions. C is newly fitted on all150,000 rows using77 relative-history features, seed42 and two internal estimators; no context subsampling, neural-weight training or adapter is used for C. The19 historical-level fields are replaced by their value minus the original per-row temporal baseline, preserving missingness. All targets and residual baselines remain unchanged.", "",
               "All weights stay fixed at their previous values. Each model predicts residual rent and adds the original query baseline once before averaging. A predictions are read from the full-precision cache rather than its rounded upload CSV. Existing source artifacts and submissions are checked by SHA256 and preserved.", "",
               "The inherited temporal policy uses histories through February2025 for March2025 test records, and through March2025 for later test records. Added E spatial features retain their extra one-month lag. All provided labels remain in each TabPFN context. The50,000 test records span March2025-July2026;18,077 lie beyond the ten-month validated temporal range (maximum16months). The existing forecast level/uncertainty plateau at month10, while actual calendar, horizon and history age remain accurate. C only changes the rowwise representation and adds no new historical information.", "",
               "Validation checks cover every test ID, output column, member alignment, positive/finite prediction, convex combination and CSV rounding. The test file has no rent labels, so local test RMSE cannot be calculated. These files are saved locally and have not been uploaded to Kaggle.", "",
               "Reproduce/resume: `.venv/Scripts/python.exe analysis/train_tabpfn_top2_submission.py --stage all`. Source/weight hashes are in settings.json; the new C bundle is under.cache/tabpfn_top2_final_submission. Full-precision member predictions, monthly summaries, GPU timing and verification.json are retained here.", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "one", "export", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage == "one":
        predict_C()
    if args.stage == "all":
        export(include_C=False)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "TABPFN_NO_BROWSER": "1", "TABPFN_DISABLE_TELEMETRY": "1"}
        for key in ("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "TABPFN_MAX_BATCHED_ESTIMATOR_CELLS"):
            env.pop(key, None)
        with (OUT / "inference_C.log").open("a", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-u", str(Path(__file__)), "--stage", "one"],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("C inference failed; inspect reports/tabpfn_top2_final_submission/inference_C.log")
    if args.stage in ("all", "export"):
        export()


if __name__ == "__main__":
    main()

"""Check full-data API inputs, preserve provenance, then audit the Kaggle CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from analysis.config import ROOT, DATA, REPORTS
from analysis.tabpfn_api_77 import load_inputs, EXPORTS, file_hash, write_json
from analysis.compare_local_economy_landscape import immutable
from analysis.train_tabpfn_feature_ensemble_submission import full_base, FEATURE_PATH

RUN = "submission_v3.5_plus_77_150k"
OUT = REPORTS / "tabpfn_api_77" / RUN
CACHE = ROOT / ".cache/tabpfn_api_77" / RUN


def check_inputs():
    b = full_base()
    a = load_inputs(EXPORTS / "submission", "submission")
    raw, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    assert a["x"].shape == (150000, 77) and a["q"].shape == (50000, 77)
    assert "MONTHLY_RENT" not in test and a["actual"] is None
    np.testing.assert_array_equal(a["x"].index, raw.index)
    np.testing.assert_array_equal(a["q"].index, test.index)
    np.testing.assert_array_equal(b["y_train"], raw.MONTHLY_RENT)
    for key, original in [("x", "X_train"), ("q", "X_query")]:
        x, z = b[original], a[key]
        assert list(x) == list(z)
        np.testing.assert_array_equal(x.index, z.index)
        for column in x:
            np.testing.assert_array_equal(x[column].isna(), z[column].isna())
            if isinstance(x[column].dtype, pd.CategoricalDtype):
                assert isinstance(z[column].dtype, pd.CategoricalDtype)
                np.testing.assert_array_equal(x[column].cat.codes, z[column].cat.codes)
                np.testing.assert_array_equal(x[column].cat.categories, z[column].cat.categories)
                assert x[column].cat.ordered == z[column].cat.ordered
            else:
                np.testing.assert_allclose(x[column].to_numpy(float), z[column].to_numpy(float),
                                           rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(a["target"], np.asarray(b["y_train"]) - b["train_baseline"])
    np.testing.assert_array_equal(a["baseline"], b["query_baseline"])
    np.testing.assert_array_equal(a["month"], test.RENT_APPROVAL_DATE)
    return b, a


def preflight():
    b, a = check_inputs()
    quote = json.loads((OUT / "quote.json").read_text())
    usage = json.loads((OUT / "usage_before.json").read_text())
    limits = json.loads((OUT / "server_settings.json").read_text())["model_limits"]["v3.5"]
    assert quote["quotes"][0]["inputs"] == dict(train_rows=150000, raw_columns=77, test_rows=50000,
        model_version="v3.5", operation="predict", n_estimators=8, thinking_effort=None)
    assert quote["estimated_prediction_total"] <= usage["daily_token_limit"] - usage["daily_tokens_used"]
    assert quote["estimated_prediction_total"] <= usage["monthly_token_limit"] - usage["monthly_tokens_used"]
    assert len(a["x"]) <= limits["train_set_max_rows"]
    assert a["x"].size <= min(limits["train_set_max_cells"], limits["train_set_max_upload_cells"])
    assert len(a["q"]) <= limits["test_set_max_rows"] and a["q"].size <= limits["test_set_max_cells"]
    paths = [DATA / name for name in ("train.csv", "test.csv", "example-submission.csv")]
    paths += [FEATURE_PATH, ROOT / "analysis/tabpfn_api_77.py", Path(__file__),
              EXPORTS / "submission/manifest.json"]
    paths += [EXPORTS / "submission" / name for name in ("train.parquet", "query.parquet", "arrays.npz")]
    paths += [REPORTS / "tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv",
              REPORTS / "tabpfn_feature_ensemble_final_submission/submission_tabpfn_optimized_ensemble_150k.csv",
              REPORTS / "tabpfn_top2_final_submission/submission_tabpfn_ABCDE_equal_150k.csv",
              REPORTS / "tabpfn_top2_final_submission/submission_tabpfn_ABDE_weighted_150k.csv"]
    immutable(OUT / "input_audit.json", dict(
        verified=True, context_rows=150000, test_rows=50000, features=77,
        categorical_columns=list(a["x"].select_dtypes("category")),
        all_feature_values_missingness_category_codes_and_ids_exact=True,
        original_residual_targets_and_baselines_exact=True,
        full_context_fits_live_service_limits=True, subsampling_requested=False,
        quote_within_recorded_daily_and_monthly_allowance=True,
        test_labels_available=False,
        model_selection_source="https://docs.priorlabs.ai/models/selecting-model-version",
        model_selection="Official hosted Plus selector v3.5_default; eight estimators, no Thinking",
        source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}))
    print("PASS: all150000 training and50000 test rows exactly match original77-feature inputs", flush=True)


def final():
    audit = json.loads((OUT / "input_audit.json").read_text())
    for name, digest in audit["source_sha256"].items():
        assert file_hash(ROOT / name) == digest, name
    b, a = check_inputs()
    settings = json.loads((OUT / "settings.json").read_text())
    assert settings == json.loads((CACHE / "settings.json").read_text())
    assert settings["split"] == "submission" and settings["context_rows"] == 150000
    assert settings["query_rows"] == 50000 and len(settings["features"]) == 77
    assert settings["train_rows_limit"] is None and settings["query_rows_limit"] is None
    params = settings["estimator"]
    assert params["model_path"] == "v3.5_default" and params["n_estimators"] == 8
    assert params["random_state"] == 42 and params["inference_config"]["SUBSAMPLE_SAMPLES"] is None
    categories = [i for i, name in enumerate(a["x"]) if isinstance(a["x"][name].dtype, pd.CategoricalDtype)]
    assert params["categorical_features_indices"] == categories and len(categories) == 9
    model = json.loads((CACHE / "model.json").read_text())
    assert model["n_train_rows"] == 150000 and model["task"] == "regression"
    assert model["fit_status"] == "completed"
    for key, value in params.items():
        assert model["params"][key] == value, key
    assert not model["params"]["thinking_mode"]
    assert json.loads((CACHE / "progress.json").read_text())["next_row"] == 50000
    assert not (CACHE / "pending_prediction.json").exists()
    values = np.load(CACHE / "predictions.npy", allow_pickle=False)
    assert values.shape == (50000,) and np.isfinite(values).all() and (values > 0).all()
    sample = pd.read_csv(DATA / "example-submission.csv")
    path = OUT / "submission_tabpfn_api.csv"
    saved = pd.read_csv(path, float_precision="round_trip")
    assert saved.shape == sample.shape == (50000, 2) and list(saved) == list(sample) == ["Id", "Predicted"]
    assert saved.Id.is_unique and np.isfinite(saved.Predicted).all() and (saved.Predicted > 0).all()
    np.testing.assert_array_equal(saved.Id, sample.Id)
    np.testing.assert_array_equal(saved.Predicted, values)
    rows = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(rows.row_id, sample.Id)
    np.testing.assert_array_equal(rows.month, a["month"])
    np.testing.assert_array_equal(rows.predicted, values)
    frame = pd.DataFrame(dict(month=a["month"], predicted=values,
                              horizon=b["X_query"].TEMP_FORECAST_HORIZON.to_numpy()))
    frame.groupby("month").agg(rows=("predicted", "size"), mean_sgd=("predicted", "mean"),
        minimum_sgd=("predicted", "min"), maximum_sgd=("predicted", "max"),
        maximum_horizon=("horizon", "max")).to_csv(OUT / "test_month_summary.csv")
    timings = json.loads((CACHE / "last_timings.json").read_text())
    write_json(OUT / "server_timings.json", timings)
    result = dict(verified=True, submission=path.relative_to(ROOT).as_posix(), submission_sha256=file_hash(path),
        model="Hosted TabPFN-3.5 Plus", model_path="v3.5_default", n_estimators=8, seed=42,
        context_rows=150000, test_rows=50000, features=77, categorical_columns=9,
        full_training_upload_and_no_context_subsampling_requested=True,
        saved_fit_metadata_training_rows_verified=True,
        complete_predictions_and_exact_sample_ids_order_verified=True,
        all_predictions_positive_and_finite=True,
        csv_matches_full_precision_prediction_cache_exactly=True,
        original_features_targets_baselines_and_submissions_unchanged=True,
        baseline_added_once_by_verified_existing_runner=True,
        test_labels_available=False, test_rmse_available=False, uploaded_to_kaggle=False,
        minimum_sgd=float(values.min()), maximum_sgd=float(values.max()), mean_sgd=float(values.mean()),
        test_month_range=[str(frame.month.min()), str(frame.month.max())],
        beyond_validated10month_rows=int((frame.horizon > 10).sum()),
        maximum_horizon=int(frame.horizon.max()))
    write_json(OUT / "verification.json", result)
    report = f"""# TabPFN-3.5 Plus: full-data77-feature Kaggle submission

Upload [{path.name}]({path.name}):50,000 rows, exactly `Id,Predicted`, in the supplied test/sample order. Every prediction is finite and positive.

The managed Prior Labs API fits on all150,000 supplied labelled rows with the original77 features and nine explicit categorical columns. It uses `model_path=\"v3.5_default\"`, eight internal estimators, seed42, `fit_mode=\"fit_preprocessors\"`, mean regression predictions and `SUBSAMPLE_SAMPLES=None`. The official [model-selector documentation](https://docs.priorlabs.ai/models/selecting-model-version) identifies this hosted selector as TabPFN-3.5 Plus. Thinking is disabled, and no local DiagScale adapter is transferred. API fitting conditions the pretrained model on the data; it is not a new neural-weight fine-tuning run.

All200,000 feature rows were compared with the original saved full-data bundle, checking exact numeric values, category values/codes/order, intentional missingness and transaction IDs. The portable Parquet inputs bridge the project pandas3 environment and the API client's pandas2 environment without changing values. Only training features/residual labels and query features are supplied to the service. The target remains raw rent minus the original training temporal baseline; each returned residual receives the original query baseline once.

The existing forecasting policy is preserved. March2025 test histories endFebruary2025; later test histories endMarch2025. All supplied labels remain in the requested context. There are{result['beyond_validated10month_rows']:,} test rows beyond the validated ten-month local forecast range, with maximum horizon{result['maximum_horizon']}; the inherited baseline forecast plateaus at horizon10 while actual month, horizon and history age remain accurate.

The service quote was103,136 quota_v3 units, within the recorded account allowance. quote.json,usage_before.json,usage_after.json,server_settings.json andserver_timings.json record metering and execution. Settings,source hashes,input_audit.json andverification.json record provenance; saved fit and completed inference references are under.cache/tabpfn_api_77/{RUN}/. Credentials are not included in reports.

All50,000 output IDs and values were reloaded and checked against the complete cached predictions and supplied sample. Prediction range: S${values.min():,.2f}–S${values.max():,.2f}; mean S${values.mean():,.2f}. All original submissions are unchanged. The test set has no rent labels, so test RMSE is unavailable. This eight-estimator full-data configuration has no newly measured validation score; previous one-estimator validation scores must not be assigned to it. The CSV has been prepared locally and has not been uploaded to Kaggle.

Reproduce/resume using the dedicated API environment: `.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage run --split submission --model v3.5_default --n-estimators 8 --run-name {RUN}`. In this environment, unset the stale HTTP_PROXY/HTTPS_PROXY/ALL_PROXY variables for that process. Audit with `.venv/Scripts/python.exe analysis/audit_tabpfn_plus_submission.py --stage final`.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print("PASS: complete50000-row Kaggle CSV, full150000-row API fit,77 features,8 estimators", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "final"], required=True)
    args = parser.parse_args()
    preflight() if args.stage == "preflight" else final()

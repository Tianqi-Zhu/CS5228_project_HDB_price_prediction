"""Independently reconstruct every submission row and the full C feature matrices."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT, DATA
from analysis.data import sha256
from analysis.compare_local_economy_landscape import write_json, hashlib_array
from analysis.experiments.temporal_feature_views import LEVEL_COLUMNS
from analysis import train_tabpfn_top2_submission as run


def main():
    settings = run.verify_sources()
    raw = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "example-submission.csv")
    assert len(raw) == 150000 and len(test) == len(sample) == 50000
    assert "MONTHLY_RENT" not in test and list(sample) == ["Id", "Predicted"]
    np.testing.assert_array_equal(sample.Id, test.index)
    base = run.prior.full_base()
    manifest = json.loads((run.OUT / "features_C.json").read_text())
    assert sha256(run.FEATURES) == manifest["sha256"]
    c = joblib.load(run.FEATURES)
    for key in ("y_train", "train_baseline", "query_baseline", "train_month", "query_month", "query_origin"):
        np.testing.assert_array_equal(c[key], base[key])
    np.testing.assert_array_equal(c["y_train"], raw.MONTHLY_RENT)
    for split in ("train", "query"):
        x, z = base[f"X_{split}"], c[f"X_{split}"]
        assert x.shape == z.shape == (150000 if split == "train" else 50000, 77)
        np.testing.assert_array_equal(z.index, raw.index if split == "train" else test.index)
        expected_names = [name for name in x if name not in LEVEL_COLUMNS] + [name + "_REL_BASE" for name in LEVEL_COLUMNS]
        assert list(z) == expected_names == manifest["feature_names"]
        original = [name for name in x if name not in LEVEL_COLUMNS]
        pd.testing.assert_frame_equal(x[original], z[original])
        for name in LEVEL_COLUMNS:
            expected = x[name].to_numpy(float) - np.asarray(base[f"{split}_baseline"])
            np.testing.assert_allclose(z[name + "_REL_BASE"], expected, rtol=0, atol=0, equal_nan=True)
    c_timing = json.loads((run.OUT / "timing_C.json").read_text())
    assert c_timing["context_rows"] == 150000 and c_timing["test_rows"] == 50000
    assert not c_timing["subsampling"] and c_timing["n_estimators"] == 2
    assert c_timing["native_default_estimator_batching"] and c_timing["parameters_unchanged"]
    assert c_timing["frozen_digest_before"] == c_timing["frozen_digest_after"]
    assert c_timing["training_context_ids_sha256"] == hashlib_array(raw.index)
    assert c_timing["test_ids_sha256"] == hashlib_array(test.index)
    assert c_timing["features_sha256"] == sha256(run.FEATURES)
    a_settings = json.loads((run.diag.OUT / "inference_settings.json").read_text())
    assert a_settings["context_rows"] == 150000
    assert a_settings["context_ids_sha256"] == hashlib_array(raw.index)
    assert a_settings["test_ids_sha256"] == hashlib_array(test.index)
    a = np.load(run.diag.CACHE / "predictions.npy")
    a_old = pd.read_csv(run.diag.SUBMISSION)
    np.testing.assert_array_equal(a_old.Id, sample.Id)
    np.testing.assert_allclose(a_old.Predicted, a, rtol=0, atol=1e-6)
    columns = {"A": a}
    for letter, directory, name in [("B", run.prior.OUT, "baseline"), ("C", run.OUT, "C"),
                                     ("D", run.prior.OUT, "temporal"), ("E", run.prior.OUT, "spatial")]:
        path = directory / f"predictions_{name}.csv"
        meta = json.loads((directory / f"timing_{name}.json").read_text())
        assert sha256(path) == meta["predictions_sha256"]
        assert meta["training_context_ids_sha256"] == hashlib_array(raw.index)
        assert meta["test_ids_sha256"] == hashlib_array(test.index)
        assert meta["context_rows"] == 150000 and not meta["subsampling"]
        f = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(f.Id, sample.Id)
        np.testing.assert_array_equal(f.month, test.RENT_APPROVAL_DATE)
        columns[letter] = f.Predicted.to_numpy()
    original_weights = pd.read_csv(run.EXPERIMENT / "weights.csv", float_precision="round_trip")
    row = original_weights.loc[(original_weights.origin == "2024-06") & (original_weights.group == "ABDE")]
    np.testing.assert_array_equal(settings["weights"]["ABDE"], row[run.MEMBERS].to_numpy()[0])
    np.testing.assert_array_equal(settings["weights"]["ABCDE_equal"], [.2] * 5)
    p = np.column_stack([columns[letter] for letter in run.MEMBERS])
    assert np.isfinite(p).all() and (p > 0).all()
    combined = pd.read_csv(run.OUT / "predictions_members.csv", float_precision="round_trip")
    np.testing.assert_array_equal(combined.Id, sample.Id)
    np.testing.assert_array_equal(combined.month, test.RENT_APPROVAL_DATE)
    np.testing.assert_array_equal(combined[run.MEMBERS], p)
    output = {}
    for group, name in run.FILES.items():
        weights = np.asarray(settings["weights"][group])
        assert (weights >= 0).all() and np.isclose(weights.sum(), 1)
        # Independently sum weighted columns, without the production combine helper.
        prediction = sum(weights[i] * columns[letter] for i, letter in enumerate(run.MEMBERS))
        np.testing.assert_allclose(combined[group], prediction, rtol=0, atol=1e-9)
        saved = pd.read_csv(run.OUT / name)
        assert saved.shape == (50000, 2) and list(saved) == ["Id", "Predicted"] and saved.Id.is_unique
        np.testing.assert_array_equal(saved.Id, sample.Id)
        assert np.isfinite(saved.Predicted).all() and (saved.Predicted > 0).all()
        np.testing.assert_allclose(saved.Predicted, prediction, rtol=0, atol=1e-6)
        active = p[:, weights > 0]
        assert (prediction >= active.min(axis=1)-1e-9).all()
        assert (prediction <= active.max(axis=1)+1e-9).all()
        output[group] = dict(file=name, sha256=sha256(run.OUT / name), rows=50000,
                             maximum_rounding_error_sgd=float(np.max(np.abs(saved.Predicted-prediction))))
    assert sha256(run.OUT / run.FILES["ABDE"]) != sha256(run.OUT / run.FILES["ABCDE_equal"])
    monthly = pd.read_csv(run.OUT / "test_month_summary.csv")
    assert (monthly.groupby("ensemble")["size"].sum() == 50000).all()
    b_month = pd.PeriodIndex(base["query_month"], freq="M")
    origins = pd.PeriodIndex(base["query_origin"], freq="M")
    np.testing.assert_array_equal(base["X_query"].TEMP_FORECAST_HORIZON, b_month.asi8-origins.asi8)
    run.verify_sources()
    write_json(run.OUT / "independent_audit.json", dict(
        verified=True, submissions=output, all150000_training_labels_preserved=True,
        all200000_C_feature_rows_reconstructed=True, all_original_baselines_and_missingness_preserved=True,
        all50000_test_ids_and_five_member_vectors_verified=True,
        exact_original_weights_preserved=True, both_outputs_reconstructed_independently=True,
        C_full_context_and_frozen_backbone_verified=True,
        existing_sources_and_submissions_unchanged=True,
        test_targets_used=False, test_rmse_available=False,
        test_month_range=[str(b_month.min()), str(b_month.max())],
        beyond_validated10month_rows=int((b_month.asi8-origins.asi8 > 10).sum()),
        maximum_horizon=int((b_month.asi8-origins.asi8).max()),
        audit_code_sha256=sha256(Path(__file__)),
        production_code_sha256=sha256(Path(run.__file__))))
    print("PASSED: independently verified both50,000-row submissions and all200,000 C feature rows", flush=True)


if __name__ == "__main__":
    main()

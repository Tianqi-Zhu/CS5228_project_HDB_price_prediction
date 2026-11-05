import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def implementation():
    name = "experiments.tabfm_v10.benchmark"
    assert importlib.util.find_spec(name) is not None, "TabFM benchmark is not implemented yet"
    return importlib.import_module(name)


def population(ids=(1, 2, 3, 4)):
    n = len(ids)
    return {"numeric": np.arange(n * 2, dtype=float).reshape(n, 2),
            "categorical": np.array([[1, 1], [1, 2], [2, 1], [2, 2]])[:n],
            "origin": np.arange(n), "horizon": np.ones(n, dtype=int),
            "target": np.arange(n) * .1, "anchor_ksgd": np.full(n, 2.),
            "actual_sgd": 2000. + np.arange(n) * 100., "row_id": np.array(ids)}


def test_rejects_overlapping_training_and_validation_ids():
    m = implementation()
    with pytest.raises(ValueError, match="overlap"):
        m.validate_populations(population(), population((4, 5, 6, 7)))


def test_rejects_broken_anchor_residual_units():
    m = implementation()
    fit, hold = population(), population((5, 6, 7, 8))
    hold["target"][0] = 2000.
    with pytest.raises(ValueError, match="residual"):
        m.validate_populations(fit, hold)


def test_context_sampling_never_depends_on_labels():
    m = implementation()
    fit = population()
    first = m.sample_context(fit, 3, 42, "recent")
    fit["target"][:] = 1000000.
    fit["actual_sgd"][:] = -1000000.
    second = m.sample_context(fit, 3, 42, "recent")
    assert np.array_equal(first, second)
    assert len(first) == len(set(first)) == 3
    assert np.all((first >= 0) & (first < 4))


def test_categories_remain_categories_and_ids_are_not_features():
    m = implementation()
    frame = m.feature_frame(population(), ["area", "distance"], ["town", "flat_type"])
    assert list(frame.columns) == ["area", "distance", "town", "flat_type"]
    assert not np.issubdtype(frame["town"].dtype, np.number)
    assert "row_id" not in frame


def test_prediction_reconstruction_uses_each_rows_anchor():
    m = implementation()
    np.testing.assert_allclose(m.reconstruct(np.array([.2, -.3]), np.array([2., 3.])), [2200., 2700.])


def test_baseline_predictions_are_aligned_by_id_and_labels_checked(tmp_path):
    m = implementation()
    hold = population()
    order = np.array([3, 1, 0, 2])
    path = tmp_path / "baseline.npz"
    np.savez(path, row_id=hold["row_id"][order], actual_sgd=hold["actual_sgd"][order],
             horizon=hold["horizon"][order], predicted_sgd=np.array([2300., 2100., 2000., 2200.]))
    np.testing.assert_allclose(m.aligned_baseline(path, hold), hold["actual_sgd"])
    np.savez(path, row_id=hold["row_id"][order], actual_sgd=np.zeros(4),
             horizon=hold["horizon"][order], predicted_sgd=np.zeros(4))
    with pytest.raises(ValueError, match="label"):
        m.aligned_baseline(path, hold)


def test_quota_guard_stops_before_credits_and_requires_recent_evidence(tmp_path):
    m = implementation()
    path = tmp_path / "usage.jsonl"
    def write(used, timestamp="2026-11-01T08:00:00Z"):
        path.write_text(json.dumps({"timestamp": timestamp, "type": "event_msg", "payload": {
            "type": "token_count", "rate_limits": {"primary": {"used_percent": used},
                "secondary": {"used_percent": 47}}}}) + "\n")
    write(72)
    assert m.check_quota(path, now="2026-11-01T08:01:00Z")["primary_used_percent"] == 72
    write(90)
    with pytest.raises(RuntimeError, match="quota"):
        m.check_quota(path, now="2026-11-01T08:01:00Z")
    write(72)
    with pytest.raises(RuntimeError, match="stale"):
        m.check_quota(path, now="2026-11-01T09:00:00Z")


def test_never_overwrites_existing_experiment(tmp_path):
    m = implementation()
    destination = tmp_path / "run"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        m.new_run_directory(destination)

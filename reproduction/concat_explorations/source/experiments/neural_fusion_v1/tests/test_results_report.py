"""Tests for the results generator against synthetic run fixtures.

These guard reporting correctness, which the model tests cannot see: a generator that
pools seeds across different designs, or silently drops an incomplete run, produces a
wrong report while every other test stays green.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import make_results  # noqa: E402


RUN_A = {
    "run_id": "run-a", "mode": "probe", "splits": ["orig"], "seeds": [42],
    "finished": "2026-01-01T00:00:00",
    "checkpoint_verification": [{"split_id": "orig", "model": "mlp", "seed": 42,
                                 "ok": True, "max_abs_delta_sgd": 0.0}],
    "results": [
        {"run_id": "run-a", "split_id": "orig", "model": "mlp", "mode": "probe", "seed": 42,
         "n_parameters": 100, "stage_a_selected_epochs": 3, "stage_a_best_inner_rmse_sgd": 700.0,
         "total_seconds": 5.0, "validation_months": ["2024-04", "2025-03"],
         "overall": {"rmse_sgd": 500.0, "mae_sgd": 400.0, "bias_sgd": -10.0, "n": 35158},
         "by_horizon_band": [
             {"horizon_band": "1-3", "n": 10, "rmse_sgd": 510.0, "mae_sgd": 400.0, "bias_sgd": 0.0},
             {"horizon_band": "13-16", "n": 0, "rmse_sgd": None, "mae_sgd": None, "bias_sgd": None,
              "note": "empty band"}]},
        {"run_id": "run-a", "split_id": "orig", "model": "lstm_concat", "mode": "probe", "seed": 42,
         "n_parameters": 200, "stage_a_selected_epochs": 4, "stage_a_best_inner_rmse_sgd": 690.0,
         "total_seconds": 9.0, "validation_months": ["2024-04", "2025-03"],
         "overall": {"rmse_sgd": 490.0, "mae_sgd": 390.0, "bias_sgd": 5.0, "n": 35158},
         "by_horizon_band": []},
    ],
}

# same model/split/mode, DIFFERENT design and a much worse score
RUN_B = {
    "run_id": "run-b", "mode": "probe", "splits": ["orig"], "seeds": [42],
    "finished": "2026-01-02T00:00:00", "checkpoint_verification": [],
    "results": [
        {"run_id": "run-b", "split_id": "orig", "model": "mlp", "mode": "probe", "seed": 42,
         "n_parameters": 100, "stage_a_selected_epochs": 1, "stage_a_best_inner_rmse_sgd": 900.0,
         "total_seconds": 1.0, "validation_months": ["2024-04", "2025-03"],
         "overall": {"rmse_sgd": 636.0, "mae_sgd": 478.0, "bias_sgd": 399.0, "n": 35158},
         "by_horizon_band": []},
    ],
}


@pytest.fixture
def fake_report(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for payload in (RUN_A, RUN_B):
        directory = runs / payload["run_id"]
        directory.mkdir(parents=True)
        (directory / "run_summary.json").write_text(json.dumps(payload))
    crashed = runs / "crashed-run"
    crashed.mkdir()
    (crashed / "run_metadata.json").write_text(json.dumps({"mode": "probe"}))
    monkeypatch.setattr(make_results, "REPORT", tmp_path)
    return tmp_path


def test_incomplete_run_is_reported_not_dropped(fake_report):
    runs = make_results.load_runs()
    ids = [r["run_id"] for r in runs]
    assert "crashed-run" in ids
    incomplete = next(r for r in runs if r["run_id"] == "crashed-run")
    assert incomplete["incomplete"] is True
    make_results.main()
    text = (fake_report / "RESULTS.md").read_text()
    assert "crashed-run" in text and "INCOMPLETE" in text


def test_seeds_are_not_pooled_across_runs(fake_report):
    make_results.main()
    text = (fake_report / "RESULTS.md").read_text()
    seeds_section = text.split("## Seeds per model/split")[1].split("\n## ")[0]
    rows = [line for line in seeds_section.splitlines() if line.startswith("| run-")]
    mlp_rows = [row for row in rows if "| mlp |" in row]
    assert len(mlp_rows) == 2, "each run must get its own row, never one pooled row"
    for row in mlp_rows:
        assert "| 1 |" in row
        assert "single seed" in row
    assert "139.07" not in text, "a design difference must not be reported as seed spread"


def test_empty_band_stays_visible(fake_report):
    make_results.main()
    text = (fake_report / "RESULTS.md").read_text()
    assert "13-16: n=0" in text


def test_integer_seeds_render_as_text(fake_report):
    make_results.main()
    text = (fake_report / "RESULTS.md").read_text()
    assert "| orig | 42 |" in text


def test_reference_and_anchor_caveat_present(fake_report):
    make_results.main()
    text = (fake_report / "RESULTS.md").read_text()
    assert "477.2036" in text
    assert "anchor + residual" in text


def test_coverage_gap_lists_unrun_combinations(fake_report):
    make_results.main()
    text = (fake_report / "RESULTS.md").read_text()
    gaps = text.split("## Coverage gaps")[1]
    assert "NOT RUN" in gaps
    for split in ("early", "long"):
        assert split in gaps

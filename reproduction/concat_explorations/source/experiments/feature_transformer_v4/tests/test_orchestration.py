"""Orchestration rules, tested without training a model.

Three defects this pins, each caught before it could corrupt a result:

1. `main.run_trials` decided the pooling for trials 4-8 from whatever subset the caller
   asked for, so `--trials 1` froze "best of 1-3" from one trial.
2. Any existing run directory was refused, so a finished trial could only be re-run, never
   resumed — and nothing checked that a resumed trial still had the same inputs.
3. `freeze_top_two` returned dicts keyed `id` while callers read `trial_id`, so the log line
   after a successful freeze raised `KeyError`.

Nothing here trains; `execute_trial` is exercised only for its resume path, and the training
path is stubbed out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))

from feature_transformer_v4 import orchestrate as O  # noqa: E402
from feature_transformer_v4.training import TrainConfig  # noqa: E402


class StubTensors:
    """Enough of `FeatureTensors` for the orchestration layer, which only hashes them."""

    def __init__(self, tag: str = "x"):
        self.tag = tag

    def parity_signature(self) -> str:
        return self.tag


class StubState:
    def __init__(self):
        self.scaler = type("S", (), {"static_median": {}, "history_median": {},
                                     "context_median": {}, "log1p_columns": []})()
        self.vocabularies = {}
        self.numeric_names = []


def summary(trial_id: int, pooling: str, rmse: float | None, epoch: int | None = 7) -> dict:
    return {"id": trial_id, "pooling": pooling, "selected_epoch": epoch,
            "selected_rmse_sgd": rmse, "epochs_run": 10, "parameters": 1000,
            "config": TrainConfig(pooling=pooling).to_dict(), "changes": {},
            "ineligible_best": None, "seconds": 1.0}


# ------------------------------------------------------- 1. no early pooling decision
def test_pooling_decision_refuses_before_trials_1_3_are_complete():
    with pytest.raises(O.OrchestrationError, match=r"trials \[2, 3\]"):
        O.decide_pooling([summary(1, "mean", 600.0)])


def test_pooling_decision_refuses_with_only_two_of_three():
    with pytest.raises(O.OrchestrationError, match=r"trials \[3\]"):
        O.decide_pooling([summary(1, "mean", 600.0), summary(2, "cls", 610.0)])


def test_pooling_decision_picks_the_lowest_eligible_rmse():
    decision = O.decide_pooling([summary(1, "mean", 601.0), summary(2, "cls", 599.0),
                                 summary(3, "horizon_query", 605.0)])
    assert decision["pooling"] == "cls"
    assert decision["trial_id"] == 2
    assert decision["complete_trials_1_3"] == [1, 2, 3]


def test_ties_go_to_the_earlier_trial_id():
    decision = O.decide_pooling([summary(3, "horizon_query", 600.0),
                                 summary(1, "mean", 600.0), summary(2, "cls", 600.0)])
    assert decision["trial_id"] == 1


def test_a_trial_without_an_eligible_epoch_is_skipped_not_fatal():
    decision = O.decide_pooling([summary(1, "mean", None), summary(2, "cls", 590.0),
                                 summary(3, "horizon_query", 592.0)])
    assert decision["trial_id"] == 2
    assert decision["decided_from"] == [2, 3]


# ------------------------------------------------------------ 2. resume verification
def test_resume_reuses_a_completed_trial_without_training(monkeypatch, tmp_path):
    entry = {"id": 1, "pooling": "mean", "changes": {}}
    config = TrainConfig()
    # the fingerprint must be the one this module would compute, or the resume is refused
    fingerprint = O.fingerprint_inputs(StubState(), StubTensors(), StubTensors())
    existing = summary(1, "mean", 600.0)
    existing["input_fingerprint"] = fingerprint
    (tmp_path / "trial1").mkdir()
    (tmp_path / "trial1" / "summary.json").write_text(json.dumps(existing))

    def explode(*args, **kwargs):
        raise AssertionError("resume must not train")

    monkeypatch.setattr(O.S, "run_trial", explode)
    state, fit_tensors, hold_tensors = StubState(), StubTensors(), StubTensors()
    got, how = O.execute_trial(entry, config, state, fit_tensors, hold_tensors,
                               tmp_path, resume=True)
    assert how == "reused"
    assert got["selected_rmse_sgd"] == 600.0


def test_resume_refuses_when_a_config_field_changed(tmp_path):
    fingerprint = {"a_fit_parity": "abc"}
    existing = summary(1, "mean", 600.0)
    existing["config"]["learning_rate"] = 1e-3
    existing["input_fingerprint"] = fingerprint
    (tmp_path / "trial1").mkdir()
    (tmp_path / "trial1" / "summary.json").write_text(json.dumps(existing))

    with pytest.raises(O.OrchestrationError, match="learning_rate"):
        O.execute_trial({"id": 1, "pooling": "mean", "changes": {}},
                        TrainConfig(learning_rate=3e-4), StubState(), StubTensors(), StubTensors(),
                        tmp_path, resume=True)


def test_resume_refuses_when_the_data_fingerprint_changed(tmp_path):
    existing = summary(1, "mean", 600.0)
    existing["input_fingerprint"] = {"a_fit_parity": "old"}
    (tmp_path / "trial1").mkdir()
    (tmp_path / "trial1" / "summary.json").write_text(json.dumps(existing))

    with pytest.raises(O.OrchestrationError, match="a_fit_parity"):
        O.execute_trial({"id": 1, "pooling": "mean", "changes": {}}, TrainConfig(),
                        StubState(), StubTensors(), StubTensors(), tmp_path, resume=True)


def test_resume_refuses_a_summary_without_a_fingerprint(tmp_path):
    existing = summary(1, "mean", 600.0)          # no input_fingerprint at all
    (tmp_path / "trial1").mkdir()
    (tmp_path / "trial1" / "summary.json").write_text(json.dumps(existing))

    with pytest.raises(O.OrchestrationError, match="input_fingerprint"):
        O.execute_trial({"id": 1, "pooling": "mean", "changes": {}}, TrainConfig(),
                        StubState(), StubTensors(), StubTensors(), tmp_path, resume=True)


def test_resume_trains_a_trial_that_has_no_summary(monkeypatch, tmp_path):
    entry = {"id": 2, "pooling": "cls", "changes": {}}
    called = {}

    def fake_run_trial(entry_, config, state, a_fit, a_hold, run_dir, log_path):
        called["trial"] = entry_["id"]
        (run_dir).mkdir(parents=True, exist_ok=True)
        return summary(entry_["id"], config.pooling, 555.0)

    monkeypatch.setattr(O.S, "run_trial", fake_run_trial)
    got, how = O.execute_trial(entry, TrainConfig(pooling="cls"), StubState(), StubTensors(),
                               StubTensors(), tmp_path, resume=True)
    assert how == "trained" and called["trial"] == 2
    assert got["selected_rmse_sgd"] == 555.0


# ----------------------------------------------------------- 3. one candidate schema
def test_registered_candidates_use_the_documented_schema(tmp_path):
    top = O.register_top_two([summary(1, "mean", 601.0), summary(2, "cls", 599.0),
                              summary(4, "cls", 602.0)], tmp_path)
    assert [c["rank"] for c in top] == [1, 2]
    assert [c["trial_id"] for c in top] == [2, 1]
    for candidate in top:
        assert set(candidate) == set(O.CANDIDATE_KEYS), sorted(candidate)

    written = json.loads((tmp_path / "registered_candidates.json").read_text())
    assert written["registered_before_loading_B"] is True
    for candidate in written["candidates"]:
        assert "trial_id" in candidate and "id" not in candidate


def test_the_registered_file_can_be_read_by_the_key_the_cli_prints(tmp_path):
    """The concrete crash: a dict keyed `id` printed with `t['trial_id']`."""
    top = O.register_top_two([summary(1, "mean", 601.0), summary(2, "cls", 599.0)], tmp_path)
    line = f"{[(c['rank'], c['trial_id']) for c in top]}"      # what main used to do
    assert "1" in line
    registered = json.loads((tmp_path / "registered_candidates.json").read_text())
    assert [c["trial_id"] for c in registered["candidates"]] == [2, 1]


def test_registration_needs_two_eligible_trials(tmp_path):
    with pytest.raises(O.OrchestrationError, match="only 1"):
        O.register_top_two([summary(1, "mean", 600.0), summary(2, "cls", None)], tmp_path)


def test_candidate_projection_drops_trial_only_fields():
    candidate = O.candidate_of(summary(3, "horizon_query", 590.0), rank=1)
    assert set(candidate) == set(O.CANDIDATE_KEYS)
    assert candidate["trial_id"] == 3 and candidate["rank"] == 1
    assert "curve" not in candidate and "seconds" not in candidate


# --------------------------------------------------------------- search progress
def test_run_search_defers_the_decision_when_trials_are_missing(monkeypatch, tmp_path):
    """`--trials 1` alone must not freeze the pooling."""
    monkeypatch.setattr(O, "fingerprint_inputs", lambda *a, **k: {"a_fit_parity": "x"})

    def fake_execute(entry, config, state, a_fit, a_hold, run_dir, resume, **kwargs):
        summary_ = summary(entry["id"], config.pooling, 600.0 + entry["id"])
        summary_["input_fingerprint"] = {"a_fit_parity": "x"}
        (run_dir / f"trial{entry['id']}").mkdir(parents=True, exist_ok=True)
        (run_dir / f"trial{entry['id']}" / "summary.json").write_text(json.dumps(summary_))
        return summary_, "trained"

    monkeypatch.setattr(O, "execute_trial", fake_execute)
    outcome = O.run_search(tmp_path, StubState(), StubTensors(), StubTensors(), seed=42, only=[1])
    assert "best_pooling_decision" not in outcome["ledger"]
    assert outcome["ledger"]["decisions"] == []


def test_run_search_decides_once_all_three_exist_even_if_only_one_ran(monkeypatch, tmp_path):
    """A resumed invocation must be able to complete the decision from disk."""
    monkeypatch.setattr(O, "fingerprint_inputs", lambda *a, **k: {"a_fit_parity": "x"})
    for trial_id, pooling, rmse in ((1, "mean", 601.0), (2, "cls", 599.0), (3, "horizon_query", 605.0)):
        payload = summary(trial_id, pooling, rmse)
        payload["input_fingerprint"] = {"a_fit_parity": "x"}
        (tmp_path / f"trial{trial_id}").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"trial{trial_id}" / "summary.json").write_text(json.dumps(payload))

    def fake_execute(entry, config, state, a_fit, a_hold, run_dir, resume, **kwargs):
        summary_ = summary(entry["id"], config.pooling, 700.0)
        summary_["input_fingerprint"] = {"a_fit_parity": "x"}
        (run_dir / f"trial{entry['id']}").mkdir(parents=True, exist_ok=True)
        (run_dir / f"trial{entry['id']}" / "summary.json").write_text(json.dumps(summary_))
        return summary_, "trained"

    monkeypatch.setattr(O, "execute_trial", fake_execute)
    outcome = O.run_search(tmp_path, StubState(), StubTensors(), StubTensors(), seed=42, only=[8],
                           resume=True)
    decision = outcome["ledger"]["best_pooling_decision"]
    assert decision["pooling"] == "cls", "the decision must come from the trials on disk"
    assert decision["complete_trials_1_3"] == [1, 2, 3]

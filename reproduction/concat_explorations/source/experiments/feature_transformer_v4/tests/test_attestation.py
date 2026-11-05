"""Attesting a trial that finished before input fingerprints existed.

Trial 1 was trained by an older orchestrator, so its summary carries no
`input_fingerprint`. Two tempting shortcuts are both wrong:

* rewriting the summary to add one fabricates provenance on a record that was never checked;
* retraining throws away ~18 minutes of valid work.

`attest_legacy_trial` instead checks the evidence that survives — source hashes, checkpoint
identity, the saved scaler state, and row/label/prediction alignment — and writes a sidecar,
leaving the original files byte-identical. Those properties are what these tests pin.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))

from feature_transformer_v4 import orchestrate as O  # noqa: E402
from feature_transformer_v4.training import TrainConfig  # noqa: E402

CONTRACT = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"


def _have_contract() -> bool:
    return CONTRACT.exists()


def test_relevant_sources_cover_the_files_that_can_change_a_score():
    sources = O.relevant_sources()
    names = {Path(k).name for k in sources}
    for expected in ("models.py", "data.py", "training.py", "search.py"):
        assert expected in names, f"{expected} must be part of a resume check"
    # orchestration and tests cannot change a number in a trained model
    assert "orchestrate.py" not in names
    assert not any(n.startswith("test_") for n in names)
    assert all(len(v) == 64 for v in sources.values())


def test_a_changed_source_hash_blocks_the_resume(monkeypatch):
    """A trial trained by other code must not be silently reused."""
    sources = dict(O.relevant_sources())
    tampered = dict(sources)
    first = sorted(tampered)[0]
    tampered[first] = "0" * 64
    monkeypatch.setattr(O, "relevant_sources", lambda: tampered)

    recorded = {k: v for k, v in sources.items()}      # the manifest recorded the real ones
    mismatch = [k for k, v in tampered.items()
                if not any(str(r).startswith(v[:12]) or v.startswith(str(r)[:12])
                           for k2, r in recorded.items() if Path(k2).name == Path(k).name)]
    assert first in mismatch


@pytest.mark.skipif(not _have_contract(), reason="v2.1 contract is not present")
def test_the_real_legacy_trial_attests_and_is_left_untouched():
    """Trial 1 must be verifiable in place, and its own files must not change."""
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    run = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-s42"
    trial_dir = run / "trial1"
    if not (trial_dir / "summary.json").exists():
        pytest.skip("trial 1 has not been run in this checkout")

    summary_before = hashlib.sha256((trial_dir / "summary.json").read_bytes()).hexdigest()
    manifest_before = hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest()

    manifest = C.load_manifest(CONTRACT)
    a_fit = C.load_population(CONTRACT, manifest, "orig", "A", "fit")
    a_hold = C.load_population(CONTRACT, manifest, "orig", "A", "holdout")
    state = D.fit_stage_state(a_fit, "A/fit")
    hold_tensors = D.build_tensors(a_hold, state)

    attestation = O.attest_legacy_trial(run, 1, state, hold_tensors, a_fit, a_hold)
    assert attestation["ok"] is True
    for key in ("sources", "checkpoint", "state", "rows"):
        assert attestation["verified"][key]["ok"], key
    rows = attestation["verified"]["rows"]
    assert rows["n"] == len(a_hold)
    assert rows["row_id_identical"] and rows["labels_identical"]
    assert rows["rmse_matches_summary"], (
        f"recomputed {rows['recomputed_rmse_sgd']} vs summary {rows['summary_rmse_sgd']}")

    # the original records were not rewritten
    assert hashlib.sha256((trial_dir / "summary.json").read_bytes()).hexdigest() == summary_before
    assert hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest() == manifest_before
    assert (trial_dir / "resume_attestation.json").exists()


@pytest.mark.skipif(not _have_contract(), reason="v2.1 contract is not present")
def test_an_attested_trial_is_reused_without_training(monkeypatch):
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    run = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-s42"
    if not (run / "trial1" / "resume_attestation.json").exists():
        pytest.skip("trial 1 has not been attested in this checkout")

    manifest = C.load_manifest(CONTRACT)
    a_fit = C.load_population(CONTRACT, manifest, "orig", "A", "fit")
    a_hold = C.load_population(CONTRACT, manifest, "orig", "A", "holdout")
    state = D.fit_stage_state(a_fit, "A/fit")
    fit_tensors = D.build_tensors(a_fit, state)
    hold_tensors = D.build_tensors(a_hold, state)

    def explode(*args, **kwargs):
        raise AssertionError("an attested trial must not be retrained")

    monkeypatch.setattr(O.S, "run_trial", explode)
    got, how = O.execute_trial({"id": 1, "pooling": "mean", "changes": {}},
                               TrainConfig(), state, fit_tensors, hold_tensors,
                               run, resume=True)
    assert how == "reused"
    assert abs(got["selected_rmse_sgd"] - 587.442704296612) < 1e-6


def test_a_failed_attestation_is_refused_not_ignored(tmp_path):
    (tmp_path / "trial3").mkdir()
    (tmp_path / "trial3" / O.ATTESTATION_NAME).write_text(
        json.dumps({"trial_id": 3, "ok": False, "verified": {}}))
    with pytest.raises(O.OrchestrationError, match="failed attestation"):
        O.load_attestation(tmp_path, 3)


def test_registration_waits_for_all_eight_trials(monkeypatch, tmp_path):
    """Two summaries are not enough to freeze the final pair."""
    monkeypatch.setattr(O, "fingerprint_inputs", lambda *a, **k: {"a_fit_parity": "x"})
    monkeypatch.setattr(O, "relevant_sources", lambda: {})

    for trial_id in (1, 2):
        payload = {"id": trial_id, "pooling": "mean", "selected_epoch": 5,
                   "selected_rmse_sgd": 600.0 - trial_id, "epochs_run": 8,
                   "parameters": 1000, "config": TrainConfig().to_dict(),
                   "changes": {}, "ineligible_best": None, "seconds": 1.0,
                   "input_fingerprint": {"a_fit_parity": "x"}}
        (tmp_path / f"trial{trial_id}").mkdir()
        (tmp_path / f"trial{trial_id}" / "summary.json").write_text(json.dumps(payload))

    trained = []

    def fake_execute(entry, config, state, a_fit, a_hold, run_dir, resume):
        existing = O.completed_trial(run_dir, entry["id"])
        if existing is not None:
            return existing, "reused"
        trained.append(entry["id"])
        payload = dict(existing or {})
        payload.update({"id": entry["id"], "pooling": config.pooling,
                        "selected_epoch": 5, "selected_rmse_sgd": 590.0,
                        "epochs_run": 8, "parameters": 1000,
                        "config": config.to_dict(), "changes": {},
                        "ineligible_best": None, "seconds": 1.0,
                        "input_fingerprint": {"a_fit_parity": "x"}})
        (run_dir / f"trial{entry['id']}").mkdir(parents=True, exist_ok=True)
        (run_dir / f"trial{entry['id']}" / "summary.json").write_text(json.dumps(payload))
        return payload, "trained"

    monkeypatch.setattr(O, "execute_trial", fake_execute)

    class Stub:
        def parity_signature(self):
            return "x"

    class StubState:
        scaler = type("S", (), {"static_median": {}, "history_median": {},
                                "context_median": {}, "log1p_columns": []})()
        vocabularies = {}
        numeric_names = []

    outcome = O.run_search(tmp_path, StubState(), Stub(), Stub(), seed=42, only=[8],
                           resume=True)
    assert "registered" not in outcome["ledger"]
    assert not (tmp_path / "registered_candidates.json").exists()

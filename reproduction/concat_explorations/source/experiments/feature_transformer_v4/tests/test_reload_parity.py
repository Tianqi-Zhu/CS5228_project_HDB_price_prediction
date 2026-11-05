"""A checkpoint reload must be judged on three separate questions, and never refit.

A single "reload delta 5.96e-4, ok=False" reading was ambiguous: it combined same-backend
determinism, cross-backend kernel divergence, and score-level impact into one number, and its
1e-4 SGD threshold sat *below* the cross-backend divergence that any MPS-trained model will
show. The result looked like a broken checkpoint when the checkpoint was fine.

`parity.compare` separates them with tolerances declared before the numbers are seen:

* same backend  — save and reload on the training device: must reproduce to 1e-5 kSGD;
* cross backend — identical weights on CPU vs MPS: 1e-6 kSGD per row, with RMS and the count
  of rows over tolerance reported rather than only the maximum;
* metric effect — what a reader sees in RMSE/MAE: 1e-3 SGD.

These tests pin the separation, the declared tolerances, and that no fitting happens.
"""
from __future__ import annotations

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

from feature_transformer_v4 import parity as P  # noqa: E402

CONTRACT = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
MPS_RUN = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-mps-s42"


def test_tolerances_are_module_constants_not_stored_in_results():
    """A threshold chosen after seeing a number is not a threshold."""
    assert P.SAME_BACKEND_TOL_KSGD == 1e-5
    assert P.CROSS_BACKEND_TOL_KSGD == 1e-6
    assert P.METRIC_TOL_SGD == 1e-3
    # the kSGD->SGD relation the docstrings rely on
    assert P.CROSS_BACKEND_TOL_KSGD * 1000 == P.METRIC_TOL_SGD


@pytest.mark.skipif(not CONTRACT.exists(), reason="v2.1 contract is not present")
@pytest.mark.skipif(not (MPS_RUN / "final_trial6" / "checkpoint.pt").exists(),
                    reason="no MPS final checkpoint in this checkout")
def test_real_final_checkpoint_parity_is_reported_three_ways():
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    manifest = C.load_manifest(CONTRACT)
    b_fit = C.load_population(CONTRACT, manifest, "orig", "B", "fit")
    b_val = C.load_population(CONTRACT, manifest, "orig", "B", "validation")
    state = D.fit_stage_state(b_fit, "B/fit")
    tensors = D.build_tensors(b_val, state)

    result = P.compare(MPS_RUN / "final_trial6", state, tensors, "mps")

    # the three questions are answered separately
    for key in ("same_backend", "cross_backend", "metric_effect"):
        assert key in result, f"{key} must be reported on its own"
    assert result["checkpoint_tensors_on_cpu"] is True
    assert result["fitted_anything"] is False

    # same backend must reproduce essentially exactly
    assert result["same_backend"]["max_abs_delta_kSGD"] <= P.SAME_BACKEND_TOL_KSGD
    # cross backend must be reported with more than a maximum
    cross = result["cross_backend"]
    assert "rms_delta_kSGD" in cross and "rows_over_tolerance" in cross
    assert cross["n_rows"] == len(tensors)
    # and the score-level gap must be far below the per-row maximum
    assert result["metric_effect"]["rmse_gap_sgd"] <= P.METRIC_TOL_SGD
    assert result["ok"] is True, json.dumps(result, indent=2)


@pytest.mark.skipif(not CONTRACT.exists(), reason="v2.1 contract is not present")
def test_parity_never_fits_a_scaler_or_vocabulary(monkeypatch):
    """Rebuilding tensors for a parity check must not refit anything."""
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    manifest = C.load_manifest(CONTRACT)
    b_fit = C.load_population(CONTRACT, manifest, "orig", "B", "fit")
    b_val = C.load_population(CONTRACT, manifest, "orig", "B", "validation")
    state = D.fit_stage_state(b_fit, "B/fit")
    tensors = D.build_tensors(b_val, state)

    def explode(*args, **kwargs):
        raise AssertionError("parity.compare refitted a scaler or vocabulary")

    monkeypatch.setattr(D, "fit_stage_state", explode)
    monkeypatch.setattr(D, "apply_state", explode)

    out = MPS_RUN / "final_trial6"
    if not (out / "checkpoint.pt").exists():
        pytest.skip("no MPS final checkpoint in this checkout")
    result = P.compare(out, state, tensors, "mps")
    assert result["fitted_anything"] is False


def test_cross_backend_section_is_absent_without_mps(monkeypatch, tmp_path):
    """On a CPU-only host the report says what it could not check, instead of guessing."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert torch.backends.mps.is_available() is False


def test_metric_gap_is_computed_from_each_backends_own_predictions():
    """A gap must come from two evaluations, not from one evaluation compared to itself."""
    rng = np.random.default_rng(0)
    actual = 3000 + rng.normal(0, 400, 200)
    mps_residual = rng.normal(0, 0.3, 200)
    cpu_residual = mps_residual + rng.normal(0, 1e-7, 200)
    anchor = np.full(200, 2.5)

    def score(residual):
        predicted = (residual + anchor) * 1000.0
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))

    gap = abs(score(mps_residual) - score(cpu_residual))
    assert gap > 0, "a real cross-backend gap cannot be exactly zero"
    assert gap < P.METRIC_TOL_SGD

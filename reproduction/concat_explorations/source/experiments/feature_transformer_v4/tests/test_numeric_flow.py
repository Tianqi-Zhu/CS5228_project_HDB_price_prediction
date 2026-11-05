"""The numeric features must actually reach the model.

An earlier version projected each feature with a **scalar** weight, i.e. it added
``x_j * w_j`` as a constant to all 48 channels. The model is Pre-LN with a final LayerNorm,
and ``LN(c*1 + v) == LN(v)`` exactly — so every numeric value was annihilated. Measured at
the time: perturbing all 56 numeric inputs moved the mean-pooled prediction by 7.45e-9 and
the CLS prediction by 4.47e-8, and the numeric weight gradient was 1.86e-9. A per-token unit
test on `FeatureTokens` alone did **not** catch it, because the tokens were correctly
*different* — they were just identical after normalisation.

So these tests assert on the **full model output and the hidden representation**, on real
contract rows, and they demand effects far above float noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))

CONTRACT = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
SLICE = 256
# A scalar-weight projection measured 7.45e-9 (predictions) and 1.86e-9 (gradient). With a
# vector weight the same probes measure 6.0e-4 and 6.0e-3 — five to seven orders of
# magnitude higher. The floors below sit in the empty gap between the two regimes, and the
# hidden-representation floors are stated as a fraction of the measured hidden scale
# (0.195 for mean pooling, 0.82 for the token poolings).
PREDICTION_FLOOR = 1e-5     # one feature; observed 6.0e-4
HIDDEN_FLOOR = 1e-4         # one feature; observed 6.0e-3 .. 1.3e-2
ALL_FEATURES_FLOOR = 1e-3   # all 56 at once; observed 9.3e-2 .. 1.4e-1


def _have_contract() -> bool:
    return CONTRACT.exists()


pytestmark = pytest.mark.skipif(not _have_contract(),
                                reason="v2.1 contract is not present")


@pytest.fixture(scope="module")
def rig():
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D
    from feature_transformer_v4.models import build_model

    manifest = C.load_manifest(CONTRACT)
    a_fit = C.load_population(CONTRACT, manifest, "orig", "A", "fit")
    a_hold = C.load_population(CONTRACT, manifest, "orig", "A", "holdout")
    state = D.fit_stage_state(a_fit, "A/fit")
    tensors = D.build_tensors(a_hold, state)

    models = {}
    for pooling in ("mean", "cls", "horizon_query"):
        torch.manual_seed(7)
        models[pooling] = build_model(state.numeric_names, state.cardinalities(),
                                      context_dim=11, pooling=pooling)
        models[pooling].eval()
    return {"state": state, "tensors": tensors.subset(np.arange(SLICE)),
            "models": models}


def _batch(tensors) -> dict:
    return {
        "numeric": torch.from_numpy(tensors.numeric),
        "categorical": torch.from_numpy(tensors.categorical),
        "query_context": torch.from_numpy(tensors.query_context),
    }


def _hidden(model, batch):
    """The representation the head sees, before the final scalar output."""
    with torch.no_grad():
        tokens = model.feature_tokens(batch)
        if model.pooling == "mean":
            encoded = model.final_norm(model.encoder(tokens))
            return encoded.mean(dim=1)
        query = model._query_token(batch, tokens.dtype)
        sequence = torch.cat([query, tokens], dim=1)
        encoded = model.final_norm(model.encoder(sequence))
        return encoded[:, 0]


@pytest.mark.parametrize("pooling", ["mean", "cls", "horizon_query"])
def test_perturbing_one_numeric_feature_changes_the_prediction(rig, pooling):
    model = rig["models"][pooling]
    batch = _batch(rig["tensors"])
    with torch.no_grad():
        before = model(batch)

    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["numeric"][:, 5] += 1.5                 # one feature, one batch, all rows
    with torch.no_grad():
        after = model(perturbed)

    delta = float((after - before).abs().max())
    assert delta > PREDICTION_FLOOR, (
        f"{pooling}: changing a numeric feature moved the prediction by only {delta:.3e}. "
        "That is float noise: the value is not reaching the representation.")


@pytest.mark.parametrize("pooling", ["mean", "cls", "horizon_query"])
def test_perturbing_one_numeric_feature_changes_the_hidden_representation(rig, pooling):
    model = rig["models"][pooling]
    batch = _batch(rig["tensors"])
    hidden_before = _hidden(model, batch)

    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["numeric"][:, 5] += 1.5
    hidden_after = _hidden(model, perturbed)

    delta = float((hidden_after - hidden_before).abs().max())
    assert delta > HIDDEN_FLOOR, f"{pooling}: hidden representation barely moved ({delta:.3e})"


def test_perturbing_all_numeric_features_is_far_above_noise(rig):
    """The exact probe that exposed the bug: change every numeric input at once."""
    model = rig["models"]["mean"]
    batch = _batch(rig["tensors"])
    with torch.no_grad():
        before = model(batch)
    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["numeric"] += 2.0
    with torch.no_grad():
        after = model(perturbed)
    delta = float((after - before).abs().max())
    assert delta > ALL_FEATURES_FLOOR, (
        f"all-numeric perturbation moved predictions by only {delta:.3e}")


def test_numeric_weight_gradient_is_not_float_noise(rig):
    """A scalar-weight projection produced a 1.86e-9 gradient; a vector one must not."""
    model = rig["models"]["cls"]
    model.train()
    batch = _batch(rig["tensors"])
    target = torch.from_numpy(rig["tensors"].target)
    loss = torch.nn.functional.mse_loss(model(batch), target)
    loss.backward()
    grad = model.numeric.weight.grad
    assert grad is not None
    assert float(grad.abs().max()) > 1e-6, (
        f"numeric weight gradient max {float(grad.abs().max()):.3e} is float noise; "
        "the numeric inputs are not influencing the loss")
    assert float(grad.abs().mean()) > 1e-8


def test_weight_is_a_vector_per_feature_not_a_scalar():
    from feature_transformer_v4.models import FeatureTokens

    tokens = FeatureTokens(["a", "b", "c"], d_model=48)
    assert tokens.weight.shape == (3, 48), "weight must be (features, d_model)"
    assert tokens.bias.shape == (3, 48)
    # rows must not be constant across the width, or LayerNorm removes the whole signal
    row_spread = tokens.weight.detach().std(dim=1)
    assert float(row_spread.min()) > 1e-3, "a constant row would be erased by LayerNorm"
    # and each row must have a different direction, or features are not distinguishable
    normalised = torch.nn.functional.normalize(tokens.weight.detach(), dim=1)
    gram = normalised @ normalised.T
    off_diagonal = gram - torch.diag(torch.diag(gram))
    assert float(off_diagonal.abs().max()) < 0.999, "two features share the same direction"


def test_initial_weight_is_in_the_zero_mean_subspace():
    """LayerNorm projects out the constant direction, so start where it is preserved."""
    from feature_transformer_v4.models import FeatureTokens

    tokens = FeatureTokens([f"f{i}" for i in range(10)], d_model=48)
    row_means = tokens.weight.detach().mean(dim=1)
    assert float(row_means.abs().max()) < 1e-6, "initial weight rows must be centred"
    assert float(tokens.bias.detach().mean(dim=1).abs().max()) < 1e-6


def test_equal_values_in_different_features_stay_distinguishable(rig):
    from feature_transformer_v4.models import FeatureTokens

    tokens = FeatureTokens(["a", "b", "c"], d_model=48)
    identical = torch.ones(1, 3)
    out = tokens(identical)[0]
    assert not torch.allclose(out[0], out[1])
    assert torch.unique(out, dim=0).shape[0] == 3


def test_feature_sentinel_only_its_own_token_changes(rig):
    """Changing x_j must move token j and no other initial token."""
    from feature_transformer_v4.models import FeatureTokens

    torch.manual_seed(3)
    tokens = FeatureTokens([f"f{i}" for i in range(56)], d_model=48)
    base = torch.zeros(1, 56)
    moved = base.clone()
    moved[0, 17] = 3.0
    with torch.no_grad():
        before = tokens(base)[0]
        after = tokens(moved)[0]
    changed = (after - before).abs().max(dim=1).values > 0
    assert int(changed.sum()) == 1 and bool(changed[17]), (
        f"expected only token 17 to move; moved {torch.nonzero(changed).flatten().tolist()}")

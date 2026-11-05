"""End-to-end source -> wrapper mask regression.

The reviewer's audit (`docs/reviews/2026-10-26-macro-time-mask-audit.json`) measured the defect
at a synthetic window whose per-block validity was
`[F, F, T, F, F, F, T]` — three DBS blocks with only the third observed, four COE blocks with
only the last observed. The source returned a row-level flag, the wrapper saw "something was
observed", and **all seven** macro tokens were left visible, so perturbing the four empty ones
changed the prediction by **7.68 SGD**.

The tests here drive the real `BlockEndpointSource` and the real wrapper. Testing
`build_encoder_mask` alone cannot catch this: the helper was always correct, the information
never reached it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.macro_time_tokens_v5 import model as M
from experiments.macro_time_tokens_v5 import sources as S
from experiments.macro_time_tokens_v5 import tokens as T


# ----------------------------------------------------------------- deterministic fakes
class _FakeMacroBatch:
    """The only key the sources read is `macro_row`."""

    def __init__(self, rows):
        self._rows = torch.as_tensor(rows, dtype=torch.long)

    def __getitem__(self, key):
        if key != "macro_row":
            raise KeyError(key)
        return self._rows


def _fake_window_tensors(dbs_mask_row, coe_mask_row):
    """A one-window `WindowTensors` matching the audit's shape."""
    import numpy as np
    from experiments.macro_encoders_v3.api import WindowTensors

    dbs_mask = torch.as_tensor(dbs_mask_row, dtype=torch.float32).unsqueeze(0)
    coe_mask = torch.as_tensor(coe_mask_row, dtype=torch.float32).unsqueeze(0)
    n_channels = 24
    return WindowTensors(
        dbs_seq=torch.zeros(1, 63, 7),
        dbs_mask=dbs_mask,
        coe_seq=torch.zeros(1, 12, n_channels),
        coe_mask=coe_mask,
        coe_channel_mask=torch.zeros(1, 12, n_channels),
        window_cutoff_month=np.asarray([202401], dtype=np.int32),
        dbs_asof_date=np.asarray([20240131], dtype=np.int32),
        coe_month_ids=np.zeros((1, 12), dtype=np.int32),
        policy="origin_asof",
        contract_id="test",
    )


def _audit_shaped_source():
    """The audit's window: only the newest DBS block and the newest COE block are observed."""
    dbs_mask = [0.0] * 42 + [1.0] * 21          # blocks 1,2 empty; block 3 observed
    coe_mask = [0.0] * 9 + [1.0] * 3            # quarters 1-3 empty; quarter 4 observed
    tensors = _fake_window_tensors(dbs_mask, coe_mask)
    source = S.BlockEndpointSource(tensors, row_window_index=[0, 0, 0],
                                   contract_id="test", policy="origin_asof")
    return source, _FakeMacroBatch([0, 0, 0])


# ------------------------------------------------------------------------- the contract
def test_source_returns_per_token_validity_not_a_row_flag():
    """The source must say which of the seven tokens are real."""
    source, batch = _audit_shaped_source()
    tokens, valid = source(batch)

    assert tokens.shape == (3, 7, T.TOKEN_DIM)
    assert valid.shape == (3, 7), f"expected per-token validity (3, 7), got {tuple(valid.shape)}"
    # the audit's expectation, verbatim
    assert valid[0].tolist() == [False, False, True, False, False, False, True]


def test_summary2_source_reports_each_source_separately():
    """`summary2` must mask per source too, not per row."""
    dbs_mask = [0.0] * 63                       # no DBS observation at all
    coe_mask = [0.0] * 9 + [1.0] * 3            # COE observed
    tensors = _fake_window_tensors(dbs_mask, coe_mask)
    source = S.SummarySource(tensors, row_window_index=[0], contract_id="test",
                             policy="origin_asof")
    tokens, valid = source(_FakeMacroBatch([0]))
    assert tokens.shape == (1, 2, 48)
    assert valid.shape == (1, 2)
    assert valid[0].tolist() == [False, True]


# ------------------------------------------------------------------- wrapper placement
class _StubBase(torch.nn.Module):
    """Minimal stand-in exposing the surface the wrapper drives."""

    def __init__(self, d_model=48, n_features=4):
        super().__init__()
        self.pooling = "cls"
        self.d_model = d_model
        self.n_features = n_features
        self.type_embedding = torch.nn.Embedding(1 + n_features + 8, d_model)
        self.n_types = self.type_embedding.num_embeddings
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model, 2, 2 * d_model, batch_first=True,
                                             norm_first=True), 1)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.head_token = torch.nn.Parameter(torch.randn(d_model))
        self.query_type_id = 0
        self.head = torch.nn.Linear(d_model, 1)
        self._projection = torch.nn.Linear(1, d_model)

    def feature_tokens(self, batch):
        x = batch["numeric"]
        tokens = self._projection(x.unsqueeze(-1))
        types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
        return tokens + self.type_embedding(types)[None]

    def _query_token(self, batch, dtype):
        token = self.head_token.expand(batch["numeric"].shape[0], 1, self.d_model)
        type_id = torch.full((1,), self.query_type_id, dtype=torch.long, device=token.device)
        return token + self.type_embedding(type_id)[None]


class _StubSource(torch.nn.Module):
    def __init__(self, n_tokens=7, valid_pattern=(False, False, True, False, False, False, True)):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = 48
        self.valid_pattern = valid_pattern
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, batch):
        rows = batch["numeric"].shape[0]
        tokens = self.scale * torch.ones(rows, self.n_tokens, self.d_model)
        valid = torch.as_tensor(self.valid_pattern).unsqueeze(0).expand(rows, -1)
        return tokens, valid


def _wrapper(valid_pattern):
    base = _StubBase()
    source = _StubSource(valid_pattern=valid_pattern)
    return M.MacroBlocksOnFeatureTransformer(base, source, "blocks7")


def test_a_single_invalid_token_perturbation_leaves_the_output_unchanged():
    """Isolate one token: change token 0 only, which is invalid in the audit pattern."""

    class PerTokenSource(torch.nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.n_tokens = len(pattern)
            self.d_model = 48
            self.bias = torch.nn.Parameter(torch.zeros(self.n_tokens, self.d_model))
            self.register_buffer("valid_pattern", torch.as_tensor(pattern))

        def forward(self, batch):
            rows = batch["numeric"].shape[0]
            tokens = self.bias.unsqueeze(0).expand(rows, -1, -1)
            valid = self.valid_pattern.unsqueeze(0).expand(rows, -1)
            return tokens, valid

    torch.manual_seed(1)
    pattern = (False, False, True, False, False, False, True)
    base = _StubBase()
    source = PerTokenSource(pattern)
    model = M.MacroBlocksOnFeatureTransformer(base, source, "blocks7").eval()
    batch = {"numeric": torch.randn(3, 4)}

    with torch.no_grad():
        before = model(batch)
        source.bias[0, 0] += 10.0          # invalid token 0
        source.bias[3, 1] += 10.0          # invalid token 3
        after = model(batch)
    assert torch.equal(before, after), (
        "perturbing invalid macro tokens changed the prediction: "
        f"max delta {(before - after).abs().max().item():.3e}")

    with torch.no_grad():
        before = model(batch)
        source.bias[2, 0] += 10.0          # valid token 2
        after = model(batch)
    assert not torch.equal(before, after), "perturbing a valid token did nothing"


def test_valid_tokens_report_their_gradient_and_invalid_ones_too():
    """Masking is about attention visibility, not about freezing the parameters."""
    torch.manual_seed(2)
    pattern = (False, False, True, False, False, False, True)
    base = _StubBase()
    source = _StubSource(valid_pattern=pattern)
    model = M.MacroBlocksOnFeatureTransformer(base, source, "blocks7")
    model({"numeric": torch.randn(2, 4)}).sum().backward()
    assert source.scale.grad is not None


def test_wrapper_rejects_a_row_level_flag():
    """Guarding the API: a (B,) flag must not be accepted as per-token validity."""
    torch.manual_seed(3)
    base = _StubBase()

    class RowLevelSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_tokens = 7
            self.d_model = 48
            self.bias = torch.nn.Parameter(torch.zeros(7, 48))

        def forward(self, batch):
            rows = batch["numeric"].shape[0]
            return (self.bias.unsqueeze(0).expand(rows, -1, -1),
                    torch.ones(rows, dtype=torch.bool))

    model = M.MacroBlocksOnFeatureTransformer(base, RowLevelSource(), "blocks7")
    with pytest.raises(T.MacroTimeTokenError, match="per-token"):
        model({"numeric": torch.randn(2, 4)})


# --------------------------------------------------------------- initialisation parity
def test_shared_initial_state_is_bitwise_equal_on_every_paired_tensor():
    summary_state, block_state, parity = S.shared_initial_state(42)
    assert parity["n_shared_tensors"] == 14
    assert parity["all_shared_bitwise_equal"], [
        r for r in parity["rows"] if not r["bitwise_equal"]]
    for row in parity["rows"]:
        assert row["bitwise_equal"], row


def test_rebuilding_from_the_same_seed_reproduces_the_pairing():
    _, _, first = S.shared_initial_state(7)
    _, _, second = S.shared_initial_state(7)
    assert first["summary_sha"] == second["summary_sha"]
    assert first["block_dbs_sha"] == second["block_dbs_sha"]
    assert first["block_coe_sha"] == second["block_coe_sha"]


def test_a_different_seed_produces_a_different_pairing():
    _, _, a = S.shared_initial_state(7)
    _, _, b = S.shared_initial_state(8)
    assert a["summary_sha"] != b["summary_sha"]


def test_no_live_parameter_is_shared_between_the_two_arms():
    """Copies, not tied tensors: training one arm must not move the other's weights."""
    summary_state, block_state, _ = S.shared_initial_state(42)
    before = block_state["dbs"]["rnn.weight_ih_l0"].clone()
    summary_state["dbs.lstm.weight_ih_l0"].add_(1.0)
    assert torch.equal(before, block_state["dbs"]["rnn.weight_ih_l0"])

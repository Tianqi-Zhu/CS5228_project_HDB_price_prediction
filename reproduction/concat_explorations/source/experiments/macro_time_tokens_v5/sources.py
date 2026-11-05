"""Token sources for the two v5 arms, over the frozen v3 macro contract.

`summary2` reads the producer's existing encoders (one token per source). `blocks7` reads the
v5 block-endpoint encoders (3 DBS + 4 COE). Both consume the *same* window tensors, so the two
arms differ only in how the window is summarised — never in the data, the scaling or the
policy.

Both sources return a **per-token validity mask**, not a row-level flag. An earlier version
returned one boolean per row ("did this window have anything at all"), so a row with three
observed DBS blocks and one observed COE month looked fully valid and all seven tokens entered
attention — including the four that were pure padding. The reviewer's audit measured a 7.68 SGD
prediction change from perturbing exactly those invalid tokens. Validity is therefore per token
here, and the wrapper masks positionally.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import torch
from torch import nn

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

from experiments.macro_time_tokens_v5 import tokens as T

D_MODEL = 48


class _WindowSource(nn.Module):
    """Shared plumbing: select this batch's windows, move them, encode, scatter to rows."""

    n_tokens: int
    d_model: int = D_MODEL

    def __init__(self, tensors, row_window_index, contract_id: str, policy: str,
                 window_meta: dict | None = None):
        super().__init__()
        self.register_buffer("_nothing", torch.zeros(0), persistent=False)
        self._tensors = tensors
        self._row_window = np.asarray(row_window_index, dtype=np.int64)
        self.contract_id = contract_id
        self.policy = policy
        self.window_meta = window_meta or {}

    def _select(self, batch, device):
        from experiments.macro_encoders_v3.api import (
            WindowTensors, batch_indices, select_window_tensors)

        rows = batch["macro_row"].to("cpu").numpy()
        unique, inverse = batch_indices(self._row_window, rows, self._tensors.n_windows)
        subset = select_window_tensors(self._tensors, unique)
        moved = WindowTensors(
            dbs_seq=subset.dbs_seq.to(device), dbs_mask=subset.dbs_mask.to(device),
            coe_seq=subset.coe_seq.to(device), coe_mask=subset.coe_mask.to(device),
            coe_channel_mask=subset.coe_channel_mask.to(device),
            window_cutoff_month=subset.window_cutoff_month,
            dbs_asof_date=subset.dbs_asof_date, coe_month_ids=subset.coe_month_ids,
            policy=subset.policy, contract_id=subset.contract_id)
        index = torch.as_tensor(np.asarray(inverse, dtype=np.int64), device=device)
        return moved, index

    def describe(self) -> dict:
        return {"type": type(self).__name__, "tokens": self.n_tokens,
                "d_model": self.d_model, "contract_id": self.contract_id,
                "policy": self.policy, **self.window_meta}


class SummarySource(_WindowSource):
    """The reference: producer encoders, one token per source."""

    n_tokens = 2

    def __init__(self, tensors, row_window_index, contract_id, policy, window_meta=None,
                 summary_state=None):
        super().__init__(tensors, row_window_index, contract_id, policy, window_meta)
        from experiments.macro_encoders_v3 import encoders as macro_encoders
        self.encoders = macro_encoders.build_encoders()
        if summary_state is not None:
            self.encoders.load_state_dict(summary_state)
        self.d_model = int(self.encoders.dbs.token_dim)

    def forward(self, batch):
        from experiments.macro_encoders_v3.api import encode_windows

        device = next(self.encoders.parameters()).device
        windows, index = self._select(batch, device)
        per_window = encode_windows(self.encoders, windows)
        dbs = per_window["dbs_token"].index_select(0, index)
        coe = per_window["coe_token"].index_select(0, index)
        tokens = torch.stack([dbs, coe], dim=1)          # (B, 2, d)
        # per source, from the producer's own missing flag (1 = window had no observation)
        valid = torch.stack([
            (1.0 - per_window["dbs_missing"]).index_select(0, index).squeeze(-1) > 0.5,
            (1.0 - per_window["coe_missing"]).index_select(0, index).squeeze(-1) > 0.5,
        ], dim=1)                                         # (B, 2)
        return tokens, valid


class BlockEndpointSource(_WindowSource):
    """Seven tokens: 3 DBS 21-row block ends + 4 COE 3-month block ends."""

    n_tokens = 7

    def __init__(self, tensors, row_window_index, contract_id, policy, window_meta=None,
                 dbs_state=None, coe_state=None):
        super().__init__(tensors, row_window_index, contract_id, policy, window_meta)
        self.dbs = T.build_dbs_block_encoder()
        self.coe = T.build_coe_block_encoder()
        if dbs_state is not None:
            self.dbs.load_state_dict(dbs_state)
        if coe_state is not None:
            self.coe.load_state_dict(coe_state)
        self.d_model = T.TOKEN_DIM
        self.source_embedding = nn.Embedding(2, T.TOKEN_DIM)
        self.age_embedding = nn.Embedding(4, T.TOKEN_DIM)   # shared across sources, 0=newest

    def forward(self, batch):
        device = next(self.dbs.parameters()).device
        windows, index = self._select(batch, device)

        dbs = self.dbs(windows.dbs_seq, windows.dbs_mask)
        coe = self.coe(windows.coe_seq, windows.coe_mask)

        dbs_tokens = dbs["tokens"].index_select(0, index)      # (B, 3, 48)
        coe_tokens = coe["tokens"].index_select(0, index)      # (B, 4, 48)
        dbs_age = dbs["age"].index_select(0, index)
        coe_age = coe["age"].index_select(0, index)

        # source identity and calendar age, so maturity survives compaction and the model can
        # tell the two sources' tokens apart
        dbs_tokens = dbs_tokens + self.source_embedding(
            torch.zeros_like(dbs_age)) + self.age_embedding(dbs_age)
        coe_tokens = coe_tokens + self.source_embedding(
            torch.ones_like(coe_age)) + self.age_embedding(coe_age)

        tokens = torch.cat([dbs_tokens, coe_tokens], dim=1)
        # per block: 3 DBS blocks then 4 COE blocks, in token order
        valid = torch.cat([
            dbs["valid"].index_select(0, index),
            coe["valid"].index_select(0, index),
        ], dim=1)                                             # (B, 7)
        return tokens, valid

    def describe(self) -> dict:
        base = super().describe()
        base.update({
            "blocks": {"dbs": {"rows": T.DBS_BLOCK_ROWS, "count": 3},
                       "coe": {"months": T.COE_BLOCK_MONTHS, "count": 4}},
            "age_embedding": {"rows": 4, "newest": 0, "shared_across_sources": True},
            "source_embedding": {"rows": 2},
            "validity": "per token (per block); invalid blocks are masked out of attention",
            "causal": "each block-end token is the recurrent state after exactly that block's "
                      "last observed step; later steps cannot affect it",
        })
        return base


# ------------------------------------------------------------------ paired initialisation
def shared_initial_state(seed: int):
    """One seed, one set of producer encoders, and a block arm seeded *from* them.

    The plan asks for identical initial values on every parameter the two arms genuinely share,
    with no live parameter tying. What they share is the recurrent core and its projection:
    `summary2` reads a whole-window LSTM/GRU summary, `blocks7` reads block-end states of the
    *same* recurrent dynamics. Starting the block RNN from the producer's weights makes the
    comparison a comparison of representations, not of initialisation.

    Shapes are checked rather than assumed. DBS pairs as `dbs.lstm.*` -> `dbs.rnn.*`, COE as
    `coe.gru.*` -> `coe.rnn.*`; `projection` and `empty` pair under the same names.

    Returns ``(summary_state, block_state, parity)``. ``parity["rows"]`` records a bitwise
    comparison per shared tensor, so the claim is measured, not asserted from the mapping.
    """
    import random as _random

    from experiments.macro_encoders_v3 import encoders as macro_encoders

    torch.manual_seed(seed)
    _random.seed(seed)
    producer = macro_encoders.build_encoders()
    summary_state = {k: v.detach().clone() for k, v in producer.state_dict().items()}

    # same seed for the block encoders, so the arm-specific parts (age_embedding) are
    # reproducible too; the shared parts are then overwritten from the producer
    torch.manual_seed(seed)
    dbs = T.build_dbs_block_encoder()
    coe = T.build_coe_block_encoder()

    def copy_shared(prefix, cell_name, target_module):
        source_state = producer.state_dict()
        target_state = {k: v.detach().clone() for k, v in target_module.state_dict().items()}
        pairs = []
        for key in target_state:
            if key.startswith("rnn."):
                source_key = f"{prefix}.{cell_name}.{key.split('.', 1)[1]}"
            elif key.startswith(("projection.", "empty")):
                source_key = f"{prefix}.{key}"
            else:
                continue                      # age_embedding is arm-specific
            if source_key not in source_state:
                raise T.MacroTimeTokenError(f"no pairing for {key}: {source_key} absent")
            if source_state[source_key].shape != target_state[key].shape:
                raise T.MacroTimeTokenError(
                    f"shape mismatch for {key}: {tuple(target_state[key].shape)} vs "
                    f"{source_key} {tuple(source_state[source_key].shape)}")
            target_state[key] = source_state[source_key].detach().clone()
            pairs.append((key, source_key))
        target_module.load_state_dict(target_state)
        return pairs, {k: v.detach().clone() for k, v in target_module.state_dict().items()}

    dbs_pairs, dbs_state = copy_shared("dbs", "lstm", dbs)
    coe_pairs, coe_state = copy_shared("coe", "gru", coe)
    block_state = {"dbs": dbs_state, "coe": coe_state}

    rows = []
    for block_key, producer_key in dbs_pairs:
        rows.append({"arm": "dbs", "block": block_key, "producer": producer_key,
                     "shape": list(block_state["dbs"][block_key].shape),
                     "bitwise_equal": bool(torch.equal(block_state["dbs"][block_key],
                                                       summary_state[producer_key]))})
    for block_key, producer_key in coe_pairs:
        rows.append({"arm": "coe", "block": block_key, "producer": producer_key,
                     "shape": list(block_state["coe"][block_key].shape),
                     "bitwise_equal": bool(torch.equal(block_state["coe"][block_key],
                                                       summary_state[producer_key]))})

    parity = {
        "seed": seed,
        "pairing": {"dbs": "dbs.lstm.* -> dbs.rnn.*", "coe": "coe.gru.* -> coe.rnn.*",
                    "projection": "same name", "empty": "same name",
                    "not_paired": ["age_embedding (arm-specific, reproducible from seed)",
                                   "source_embedding (block arm only)"]},
        "rows": rows,
        "n_shared_tensors": len(rows),
        "all_shared_bitwise_equal": all(r["bitwise_equal"] for r in rows),
        "summary_sha": digest_state(summary_state),
        "block_dbs_sha": digest_state(block_state["dbs"]),
        "block_coe_sha": digest_state(block_state["coe"]),
    }
    return summary_state, block_state, parity


def digest_state(state: dict) -> str:
    """Digest of a state dict, for parity auditing."""
    import hashlib

    hasher = hashlib.sha256()
    for key in sorted(state):
        hasher.update(key.encode())
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]

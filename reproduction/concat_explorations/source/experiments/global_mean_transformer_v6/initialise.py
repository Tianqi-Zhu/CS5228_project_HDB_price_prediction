"""Seeded initialisation for the 57-column model, copied from the 56-column one.

The plan is explicit that this must copy **initial** parameters, never the trained trial6
weights: using a trained checkpoint as the starting point of a "same budget" comparison makes
the comparison meaningless. So the procedure is

1. build a 56-column model from the frozen seed and configuration,
2. snapshot its state dict,
3. build the 57-column model from the same seed,
4. overwrite every parameter the two share,
5. leave only the genuinely new parameters at their fresh values.

## The two index details that are easy to get wrong

Model layout is `[CLS, 60 feature tokens]` with type rows: CLS = 61, features = 1..60. Adding a
57th numeric feature makes it 62 tokens and moves CLS to type row **62**, while the new feature
takes row **61**.

- Numeric weight/bias/identity: copy rows 0..55, leave **row 56** (the new column) fresh.
- Feature type rows 1..60: copy unchanged — the existing features keep their identities.
- CLS: type row 61 must be copied to row **62**, not left at 61, or the query token silently
  inherits the new feature's embedding.
- Type row 61 (the new feature): independently initialised.

Everything else — categorical embeddings, encoder blocks, head, head_token, query projection —
is copied as-is, because none of it changes shape.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

NEW_COLUMN_COUNT = 57
OLD_COLUMN_COUNT = 56


class InitPairingError(RuntimeError):
    pass


def digest_state(state: dict) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state):
        hasher.update(key.encode())
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def build_enlarged_initial(seed: int, build_model, numeric_names, cardinalities,
                           context_dim, config) -> tuple[object, object, dict]:
    """Return `(base56_model, enlarged57_model, parity)`.

    `build_model(numeric_names, cardinalities, context_dim=..., config=...)` is the frozen
    factory; it is called twice with the same seed.
    """
    if len(numeric_names) != NEW_COLUMN_COUNT:
        raise InitPairingError(f"expected {NEW_COLUMN_COUNT} numeric names, "
                               f"got {len(numeric_names)}")

    # --- the 56-column model, from the frozen seed
    torch.manual_seed(seed)
    base = build_model(numeric_names[:-1], cardinalities, context_dim=context_dim,
                       config=config)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    # --- the 57-column model, same seed, then overwrite the shared parts
    torch.manual_seed(seed)
    enlarged = build_model(numeric_names, cardinalities, context_dim=context_dim,
                           config=config)
    target = {k: v.detach().clone() for k, v in enlarged.state_dict().items()}

    rows = []

    def copy(key, note, source_key=None, source_slice=None, target_slice=None):
        source_key = source_key or key
        if source_key not in base_state:
            raise InitPairingError(f"{key}: no counterpart {source_key} in the 56-column model")
        if key not in target:
            raise InitPairingError(f"{key}: absent from the 57-column model")
        source = base_state[source_key]
        destination = target[key]
        if source_slice is not None:
            if source_slice == slice(0, OLD_COLUMN_COUNT):
                if destination.shape[0] != NEW_COLUMN_COUNT:
                    raise InitPairingError(
                        f"{key}: target has {destination.shape[0]} rows, expected "
                        f"{NEW_COLUMN_COUNT}")
                if source.shape[0] != OLD_COLUMN_COUNT:
                    raise InitPairingError(f"{key}: source has {source.shape[0]} rows")
                destination[source_slice] = source
                rows.append({"param": key, "copied": "rows 0..55", "note": note,
                             "target_rows": list(destination.shape),
                             "fresh_rows": "row 56 (new column)"})
                return
            else:
                raise InitPairingError(f"unsupported slice {source_slice}")
        if destination.shape != source.shape:
            raise InitPairingError(
                f"{key}: shape {tuple(destination.shape)} vs {source_key} "
                f"{tuple(source.shape)}")
        destination.copy_(source)
        rows.append({"param": key, "copied": "all", "note": note,
                     "shape": list(destination.shape)})

    # numeric: three (56, 48) tables -> (57, 48)
    for key in ("numeric.weight", "numeric.bias", "numeric.identity"):
        copy(key, "existing 56 columns keep their vectors; row 56 stays fresh",
             source_slice=slice(0, OLD_COLUMN_COUNT))

    # categorical embeddings, encoder, head, query projection: unchanged shapes
    for key in list(target):
        if key.startswith(("numeric.", "type_embedding.")):
            continue
        if key not in base_state:
            continue
        copy(key, "unchanged shape; copied verbatim")

    # type embedding: rows 1..60 and CLS 61 -> 62
    type_key = "type_embedding.weight"
    if type_key not in base_state or type_key not in target:
        raise InitPairingError(f"{type_key} missing; cannot move the CLS row")
    source_types = base_state[type_key]
    destination_types = target[type_key]
    if source_types.shape[0] + 1 != destination_types.shape[0]:
        raise InitPairingError(
            f"type embedding grew from {source_types.shape[0]} to "
            f"{destination_types.shape[0]}; expected exactly one new row")
    cls_row = source_types.shape[0] - 1                     # 61 in the 56-column model
    destination_types[0] = source_types[0]                  # row 0 is unused/reserved
    destination_types[1:1 + 60] = source_types[1:1 + 60]    # the 60 feature identities
    destination_types[cls_row + 1] = source_types[cls_row]  # CLS moves 61 -> 62
    # row 61 now belongs to the new feature and keeps its fresh initialisation
    rows.append({"param": type_key, "copied": "rows 0..60 and CLS 61->62",
                 "note": "feature identities unchanged; CLS moved; new feature row 61 fresh",
                 "fresh_row": cls_row})

    enlarged.load_state_dict(target)
    enlarged_state = {k: v.detach().clone() for k, v in enlarged.state_dict().items()}

    # --- measure the pairing rather than assert it from the procedure
    checks = []
    for key in ("numeric.weight", "numeric.bias", "numeric.identity"):
        checks.append({"param": key, "scope": "rows 0..55",
                       "bitwise_equal": bool(torch.equal(
                           enlarged_state[key][:OLD_COLUMN_COUNT], base_state[key]))})
        checks.append({"param": key, "scope": "row 56 is fresh",
                       "nonzero": bool(enlarged_state[key][OLD_COLUMN_COUNT].abs().sum() > 0)})
    checks.append({"param": type_key, "scope": "rows 1..60",
                   "bitwise_equal": bool(torch.equal(
                       enlarged_state[type_key][1:61], base_state[type_key][1:61]))})
    checks.append({"param": type_key, "scope": "CLS 61 -> 62",
                   "bitwise_equal": bool(torch.equal(
                       enlarged_state[type_key][cls_row + 1], base_state[type_key][cls_row]))})
    checks.append({"param": type_key, "scope": "row 61 differs from old CLS",
                   "bitwise_equal": bool(torch.equal(
                       enlarged_state[type_key][cls_row], base_state[type_key][cls_row]))})
    for key in list(base_state):
        if key.startswith(("numeric.", "type_embedding.")):
            continue
        checks.append({"param": key, "scope": "all",
                       "bitwise_equal": bool(torch.equal(enlarged_state[key],
                                                         base_state[key]))})

    parity = {
        "seed": seed,
        "source": "freshly seeded initial parameters, NOT the trained trial6 checkpoint",
        "numeric_rows": {"old": OLD_COLUMN_COUNT, "new": NEW_COLUMN_COUNT,
                         "copied": "0..55", "fresh": NEW_COLUMN_COUNT - 1},
        "type_rows": {"cls_old": cls_row, "cls_new": cls_row + 1,
                      "features": "1..60 unchanged", "new_feature": cls_row},
        "rows": rows,
        "checks": checks,
        # every "bitwise_equal" check must hold; the "fresh"/"nonzero" ones assert the
        # opposite, and "row 61 differs from old CLS" is the one place equality is the failure
        "all_checks_pass": all(
            (not c["bitwise_equal"]) if c["scope"] == "row 61 differs from old CLS"
            else (c["bitwise_equal"] if "bitwise_equal" in c else c["nonzero"])
            for c in checks),
        "base56_sha": digest_state(base_state),
        "enlarged57_sha": digest_state(enlarged_state),
        "base56_parameters": int(sum(v.numel() for v in base_state.values())),
        "enlarged57_parameters": int(sum(v.numel() for v in enlarged_state.values())),
    }
    return base, enlarged, parity

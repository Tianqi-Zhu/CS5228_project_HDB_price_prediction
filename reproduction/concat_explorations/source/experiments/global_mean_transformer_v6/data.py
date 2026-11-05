"""Read the shared v6 table. Do not build a second one.

`reports/global_mean_comparison_v6/data/shared_table_origA.npz` is written **once** by the
Hermes1 producer and is the only authority for the numeric matrix, the categorical codes, the
target, the anchor and the new-feature scaling. Both model classes read those same bytes, so
"the two models saw the same numbers" is a property of the file rather than a claim about two
independent pipelines.

An earlier draft of this module re-derived the matrix (its own median-centring, its own schema)
from `StageArrays`. That was wrong on two counts: it could drift from the tree's inputs, and it
duplicated ownership of a shared artifact. Nothing here fits or transforms anything.

`FeatureTensors` is constructed directly from the shared arrays — no manual flattening, because
the producer already concatenated static/history/context in the frozen order.

## What the new column does and does not support

`grm_latest_log` is one value per origin month, shared by every row with that origin:

| split | rows | origins | distinct values |
|---|---|---|---|
| A/fit | 52,310 | 26 | 26 |
| A/holdout | 35,882 | 1 | 1 |

A/holdout is **one origin spanning twelve target months**, not one calendar month. The column is
constant there because every row shares that origin's market state.

That does not make the feature uninformative. A market-wide level can still shift the common
price level and interact with horizon and property attributes — the holdout rows differ in
horizon, flat type, town and so on, so a constant input can carry a real, shared adjustment.
What a single-origin holdout cannot do is *identify* the feature's contribution by permuting
rows, because there is nothing to permute. Establishing that this is a market signal (rather
than a level correction that happens to fit one origin) needs several origins. The comparison
requested here is still the right one; the caveat is about how strongly its outcome can be read.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = ROOT / "reports" / "global_mean_comparison_v6" / "data"
SHARED_NPZ = SHARED_DIR / "shared_table_origA.npz"
SHARED_MANIFEST = SHARED_DIR / "shared_table_origA.manifest.json"
BASE_STATE = SHARED_DIR / "base_feature_state.json"
EXTRA_STATE = SHARED_DIR / "extra_feature_state.json"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"

NEW_COLUMN = "grm_latest_log"
BASE_NUMERIC = 56
TOTAL_NUMERIC = 57


class SharedTableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SharedTable:
    """The producer's frozen arrays for one split."""

    split: str
    numeric: np.ndarray
    categorical: np.ndarray
    anchor_ksgd: np.ndarray
    target: np.ndarray
    actual_sgd: np.ndarray
    row_id: np.ndarray
    horizon: np.ndarray
    origin: np.ndarray
    new_raw: np.ndarray
    new_scaled: np.ndarray

    def __len__(self) -> int:
        return int(self.numeric.shape[0])


@dataclass(frozen=True)
class SharedBundle:
    fit: SharedTable
    holdout: SharedTable
    numeric_order: list[str]
    categorical_order: list[str]
    vocabularies: dict[str, int]
    extra_state: dict          # the manifest's new_feature_state
    manifest: dict
    # The producer's frozen base feature state, loaded from
    # `base_feature_state.json` and attached here so consumers never have to guess a file
    # name or a manifest key. This is what a checkpoint must carry as `feature_state`.
    base_state: dict

    @property
    def numeric_names(self) -> list[str]:
        return list(self.numeric_order)


def load_shared_bundle() -> SharedBundle:
    if not SHARED_NPZ.exists() or not SHARED_MANIFEST.exists():
        raise SharedTableError(
            f"the shared v6 table is missing at {SHARED_NPZ}; it is owned by the Hermes1 "
            f"producer and must be written before the neural arm can run")
    manifest = json.loads(SHARED_MANIFEST.read_text())
    payload = np.load(SHARED_NPZ)

    def table(split: str) -> SharedTable:
        return SharedTable(
            split=split,
            numeric=payload[f"{split}__numeric"].astype(np.float32),
            categorical=payload[f"{split}__categorical"].astype(np.int64),
            anchor_ksgd=payload[f"{split}__anchor_ksgd"].astype(np.float64),
            target=payload[f"{split}__target"].astype(np.float32),
            actual_sgd=payload[f"{split}__actual_sgd"].astype(np.float64),
            row_id=payload[f"{split}__row_id"].astype(np.int64),
            horizon=payload[f"{split}__horizon"].astype(np.int32),
            origin=payload[f"{split}__origin"].astype(np.int32),
            new_raw=payload[f"{split}__new_feature_raw"].astype(np.float64),
            new_scaled=payload[f"{split}__new_feature_scaled"].astype(np.float64),
        )

    base_state = load_base_state()
    # fail here rather than at checkpoint time: the state must be reconstructible
    _assert_stage_state_rebuilds(base_state)
    bundle = SharedBundle(fit=table("fit"), holdout=table("holdout"),
                          numeric_order=list(manifest["numeric_order"]),
                          categorical_order=list(manifest["categorical_order"]),
                          vocabularies=dict(manifest["vocabularies"]),
                          extra_state=dict(manifest.get("new_feature_state", {})),
                          manifest=manifest, base_state=base_state)

    for part in (bundle.fit, bundle.holdout):
        if part.numeric.shape[1] != TOTAL_NUMERIC:
            raise SharedTableError(f"{part.split}: {part.numeric.shape[1]} numeric columns, "
                                   f"expected {TOTAL_NUMERIC}")
        if part.categorical.shape[1] != len(bundle.categorical_order):
            raise SharedTableError(f"{part.split}: categorical width disagrees with the "
                                   f"manifest")
        for field in ("numeric", "categorical", "anchor_ksgd", "target", "actual_sgd"):
            if not np.isfinite(np.asarray(getattr(part, field))).all():
                raise SharedTableError(f"{part.split}: non-finite values in {field}")
    if len(bundle.numeric_order) != TOTAL_NUMERIC:
        raise SharedTableError(f"manifest lists {len(bundle.numeric_order)} numeric columns")
    if bundle.numeric_order[-1] != manifest["new_feature"]:
        raise SharedTableError("the appended column is not the manifest's new feature")
    return bundle


def to_feature_tensors(table: SharedTable, query_context: np.ndarray):
    """Wrap the shared arrays as `FeatureTensors` without touching them.

    `FeatureTensors` is the frozen container the model and the batching code expect; building it
    from these arrays is a rename, not a transformation. `query_context` is the one field the
    shared table does not carry (the producer owns the 57 learned columns, not the 11 context
    columns the horizon-query pooling reads), so it is supplied by the caller and checked for
    width here.
    """
    from feature_transformer_v4.data import FeatureTensors

    context = np.asarray(query_context, dtype=np.float32)
    if context.shape != (len(table), 11):
        raise SharedTableError(f"{table.split}: query_context has shape "
                               f"{tuple(context.shape)}, expected ({len(table)}, 11)")
    return FeatureTensors(numeric=table.numeric, categorical=table.categorical,
                          query_context=context, target=table.target,
                          actual_sgd=table.actual_sgd,
                          anchor_ksgd=table.anchor_ksgd, row_id=table.row_id,
                          horizon=table.horizon)


def load_base_state() -> dict:
    """Read the producer's frozen base feature state and reshape it for `StageState.from_dict`.

    The producer writes the scaler keys flat (static_median, history_std, ...) because that is
    how `ScalerState.to_dict` flattens them; `StageState.from_dict` expects the same nesting.
    This is a structural rename, not a re-derivation: no statistic is recomputed.
    """
    if not BASE_STATE.exists():
        raise SharedTableError(f"missing {BASE_STATE}")
    payload = json.loads(BASE_STATE.read_text())
    scaler_keys = ("static_columns", "static_median", "static_mean", "static_std",
                   "history_columns", "history_median", "history_mean", "history_std",
                   "context_columns", "context_median", "context_mean", "context_std",
                   "log1p_columns", "clip_std", "fitted_population", "fitted_rows",
                   "anchor_handling", "target_scale")
    missing = [k for k in scaler_keys if k not in payload]
    if missing:
        raise SharedTableError(f"{BASE_STATE} lacks scaler fields: {missing}")
    record = {
        "scaler": {k: payload[k] for k in scaler_keys},
        "vocabularies": {name: [int(c) for c in codes]
                         for name, codes in payload["vocabularies"].items()},
        "categorical_names": list(payload["categorical_order"]),
        "fitted_population": payload["fitted_population"],
        "n_rows": int(payload.get("fitted_rows") or payload.get("n_rows") or 0),
    }
    if record["n_rows"] == 0:
        raise SharedTableError(f"{BASE_STATE} has no fitted_rows")
    record["_provenance"] = {
        "source_checkpoint": payload.get("source_checkpoint"),
        "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
        "selected_epoch": payload.get("selected_epoch"),
    }
    return record


def stage_state_from_record(record: dict):
    """Rebuild `StageState` from the producer's record. The provenance keys are dropped."""
    import feature_transformer_v4.data as feature_data

    payload = {k: v for k, v in record.items() if not k.startswith("_")}
    try:
        return feature_data.StageState.from_dict(payload)
    except Exception as error:            # noqa: BLE001 - re-raised with the field names
        raise SharedTableError(
            f"the base feature state cannot be rebuilt into a StageState: {error}") from error


def _assert_stage_state_rebuilds(record: dict) -> None:
    state = stage_state_from_record(record)
    if len(state.numeric_names) != BASE_NUMERIC:
        raise SharedTableError(f"base feature state has {len(state.numeric_names)} numeric "
                               f"columns, expected {BASE_NUMERIC}")
    if list(state.categorical_names) != ["TOWN_CLEAN", "FLAT_TYPE_CLEAN",
                                         "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN"]:
        raise SharedTableError(f"unexpected categorical order: {state.categorical_names}")
    if state.fitted_population != "A/fit":
        raise SharedTableError(f"base feature state was fitted on "
                               f"{state.fitted_population!r}, expected 'A/fit'")


def load_extra_state() -> dict:
    """The producer's standardisation of the appended column."""
    if not EXTRA_STATE.exists():
        raise SharedTableError(f"missing {EXTRA_STATE}")
    return json.loads(EXTRA_STATE.read_text())


def load_query_context(feature_state: dict):
    """Rebuild the 11 context columns from the frozen `StageState` — never refitted.

    Reuses the state the existing 581.0164 baseline was trained with, so the context pathway is
    identical between that baseline and this comparison.
    """
    import feature_transformer_v4.data as feature_data
    from neural_fusion_v2 import contract as contract

    contract_dir = ROOT / "reports" / "fusion_contract_v2" / CONTRACT_ID
    manifest = contract.load_manifest(contract_dir)
    state = feature_data.StageState.from_dict(feature_state)
    out = {}
    for name in ("fit", "holdout"):
        population = contract.load_population(contract_dir, manifest, "orig", "A", name)
        transformed = state.scaler.transform(population.arrays)
        out[name] = (np.asarray(transformed["context"], dtype=np.float32),
                     np.asarray(population.arrays.context, dtype=np.float64),
                     np.asarray(population.arrays.row_id, dtype=np.int64))
    return out, state


def bundle_feature_tensors(bundle: SharedBundle, contexts: dict):
    """`(fit_tensors, holdout_tensors)` from the shared arrays plus the frozen context."""
    for split in ("fit", "holdout"):
        context, _, row_ids = contexts[split]
        table = getattr(bundle, split)
        if not np.array_equal(row_ids, table.row_id):
            raise SharedTableError(f"{split}: row ids differ between the contract context and "
                                   f"the shared table")
    return (to_feature_tensors(bundle.fit, contexts["fit"][0]),
            to_feature_tensors(bundle.holdout, contexts["holdout"][0]))

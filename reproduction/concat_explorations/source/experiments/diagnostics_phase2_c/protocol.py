"""Same-protocol controls on the frozen v2.1 contract, orig/A only.

The point of this package is that **every method sees exactly the information the v4
Transformer sees**, so a comparison across ridge / MLP / LGB / Transformer is a comparison of
model classes, not of feature sets:

    numeric   the 56 transformed columns, in the contract's order, from the same fit-only
              scaler (`neural_fusion_v2.preprocessing.ScalerState`), including its log1p
              handling of the eight count columns
    categorical  the four categorical codes, unknown = 0, vocabularies fitted on A/fit only
    anchor    the additive baseline; every method predicts the same residual
              `actual/1000 - anchor_ksgd` and is scored as `(residual + anchor) * 1000`

No row ids, no target-month numbers as features, no outer data. Stage A only: fit on A/fit,
choose an epoch/penalty on A/holdout, report the holdout score. That is an inner-only
comparison, and it is labelled as such.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_fusion_v2 import contract as C
from neural_fusion_v2.preprocessing import TARGET_SCALE

CONTRACT_ROOT = _Path(__file__).resolve().parents[2] / "reports" / "fusion_contract_v2"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"


@dataclass(frozen=True)
class Table:
    """One protocol's inputs: 56 numeric, 4 categorical codes, anchor, target, ids."""

    numeric: np.ndarray
    categorical: np.ndarray
    anchor_ksgd: np.ndarray
    target: np.ndarray
    actual_sgd: np.ndarray
    row_id: np.ndarray
    horizon: np.ndarray
    origin: np.ndarray
    numeric_names: list[str]
    categorical_names: list[str]
    vocabularies: dict[str, list[int]]

    def __len__(self) -> int:
        return len(self.target)

    def design_matrix(self, one_hot: bool = True,
                      with_anchor_feature: bool = False) -> tuple[np.ndarray, list[str]]:
        """Exactly the information v4 receives: 56 numeric columns + 4 categoricals.

        `anchor_rent_sgd` is **not** among the 56. v4 never reads it as an input; the anchor
        enters only as the additive baseline added back after prediction, and its siblings
        (`anchor_psm_sgd`, `anchor_n`, `anchor_level_code`, `anchor_missing`) are already
        columns 52-55. Appending the anchor level here would hand the controls a feature the
        Transformer does not have.

        `with_anchor_feature=True` is a separate, clearly labelled augmentation arm for the
        "does giving the linear model the baseline level help" question. It must never be
        mixed into the main protocol comparison.
        """
        blocks = [self.numeric.astype(np.float64)]
        names = list(self.numeric_names)
        if one_hot:
            for i, name in enumerate(self.categorical_names):
                size = len(self.vocabularies[name])
                block = np.zeros((len(self), size), dtype=np.float64)
                block[np.arange(len(self)), self.categorical[:, i]] = 1.0
                # drop one level per group: the full indicator block sums to 1 in every row,
                # so it is exactly collinear with the intercept and makes the design rank
                # deficient (measured condition number 2.7e16). Dropping the first level is
                # the standard reference-coding fix and does not remove information.
                if size > 1:
                    block = block[:, 1:]
                    kept = range(1, size)
                else:
                    kept = range(size)
                blocks.append(block)
                names += [f"{name}__{code}" for code in kept]
        else:
            blocks.append(self.categorical.astype(np.float64))
            names += [f"{name}__code" for name in self.categorical_names]
        if with_anchor_feature:
            blocks.append(self.anchor_ksgd.reshape(-1, 1))
            names.append("anchor_ksgd__augmentation_only")
        return np.concatenate(blocks, axis=1), names

    def describe(self) -> dict:
        return {"n": len(self), "n_numeric": self.numeric.shape[1],
                "n_categorical": self.categorical.shape[1],
                "numeric_names": list(self.numeric_names),
                "categorical_names": list(self.categorical_names),
                "vocabularies": {k: len(v) for k, v in self.vocabularies.items()}}


def load_tables(contract_dir: _Path | None = None) -> tuple[Table, Table]:
    """A/fit and A/holdout tables, both transformed by the fit-only state."""
    from feature_transformer_v4.data import StageState, build_tensors, fit_stage_state

    directory = contract_dir or (CONTRACT_ROOT / CONTRACT_ID)
    manifest = C.load_manifest(directory)
    a_fit = C.load_population(directory, manifest, "orig", "A", "fit")
    a_hold = C.load_population(directory, manifest, "orig", "A", "holdout")

    state = fit_stage_state(a_fit, "A/fit")
    return (table_of(a_fit, state), table_of(a_hold, state))


def table_of(population, state) -> Table:
    from feature_transformer_v4.data import build_tensors

    tensors = build_tensors(population, state)
    arrays = population.arrays
    return Table(numeric=tensors.numeric, categorical=tensors.categorical,
                 anchor_ksgd=tensors.anchor_ksgd, target=tensors.target,
                 actual_sgd=tensors.actual_sgd, row_id=tensors.row_id,
                 horizon=tensors.horizon, origin=np.asarray(arrays.origin, dtype=np.int32),
                 numeric_names=list(state.numeric_names),
                 categorical_names=list(state.categorical_names),
                 vocabularies={k: list(v) for k, v in state.vocabularies.items()})


def parity(table_a: Table, table_b: Table) -> dict:
    """The two tables must describe the same rows, in one protocol."""
    checks = {
        "numeric_width_same": table_a.numeric.shape[1] == table_b.numeric.shape[1] == 56,
        "numeric_names_same": table_a.numeric_names == table_b.numeric_names,
        "vocabularies_same": table_a.vocabularies == table_b.vocabularies,
        "no_row_id_overlap": not bool(set(table_a.row_id.tolist())
                                      & set(table_b.row_id.tolist())),
        "ids_sorted": bool(np.all(np.diff(table_a.row_id) > 0)
                           and np.all(np.diff(table_b.row_id) > 0)),
    }
    checks["ok"] = all(checks.values())
    return checks


def score(table: Table, residual_prediction: np.ndarray) -> dict:
    """Residual (kSGD) -> SGD prediction -> RMSE/MAE/bias."""
    predicted = (np.asarray(residual_prediction, dtype=np.float64) + table.anchor_ksgd) \
        * TARGET_SCALE
    residual = predicted - table.actual_sgd
    return {"rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
            "mae_sgd": float(np.mean(np.abs(residual))),
            "bias_sgd": float(np.mean(residual)),
            "n": int(len(predicted))}


def anchor_only(table: Table) -> dict:
    """The mandatory floor: predict zero residual, i.e. the anchor itself."""
    return score(table, np.zeros(len(table), dtype=np.float64))


def export_predictions(path: _Path, table: Table, prediction: np.ndarray,
                       method: str) -> None:
    """Every method exports the same columns, so a reviewer can recompute from them."""
    import pandas as pd

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "SOURCE_ROW_ID": table.row_id, "origin_month_index": table.origin,
        "horizon_months": table.horizon, "actual_rent_sgd": table.actual_sgd,
        "predicted_rent_sgd": prediction, "method": method,
    }).to_csv(path, index=False)

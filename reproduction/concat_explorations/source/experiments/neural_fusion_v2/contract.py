"""Reader for the producer's real contract layout (fusion-contract-v2.0.0).

Layout produced by the contract owner — this package adapts to it and does not ask for
a different one:

    <contract>/manifest.json
    <contract>/samples/<split>/stageA.csv      incl. stage, population
    <contract>/samples/<split>/stageB.csv
    <contract>/features/<split>/stageA.npz     static, context, history, sequence,
    <contract>/features/<split>/stageB.npz     sequence_mask, target_sgd, row_id,
                                               horizon, origin
    <contract>/features/<split>/encoding.json  categorical vocabularies + medians

Column order comes from ``manifest.splits[split].feature_order``. Nothing here
hard-codes a column order; the only required column *name* is the anchor
(``anchor_rent_sgd``), whose existence is asserted rather than assumed.

Populations: stage A has ``fit`` / ``holdout`` / ``validation``; stage B has ``fit`` /
``validation``.

* A.fit     -> fit stage-A preprocessing state and parameters
* A.holdout -> early stopping only (its origins are strictly after the inner cutoff)
* B.fit     -> refit for the chosen epoch count using stage-B state
* B.validation -> outer evaluation

There is deliberately **no** path that carves an inner holdout out of B.fit. The
producer defines the A/B boundary; this reader will not invent a second one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ANCHOR_SGD = "anchor_rent_sgd"
N_CATEGORICAL = 4          # first four static columns are integer codes (verified below)
SAMPLE_COLUMNS = ["SOURCE_ROW_ID", "stage", "population", "target_month_index",
                  "origin_month_index", "horizon_months", "actual_rent_sgd"]
ARRAY_KEYS = ["static", "context", "history", "sequence", "sequence_mask",
              "target_sgd", "row_id", "horizon", "origin"]


class ContractNotReady(RuntimeError):
    """The contract, or a required artifact, does not exist yet."""


class ContractInvalid(ValueError):
    """The contract exists but violates an invariant this consumer relies on."""


@dataclass(frozen=True)
class StageArrays:
    split: str
    stage: str
    static: np.ndarray
    context: np.ndarray
    history: np.ndarray
    sequence: np.ndarray
    sequence_mask: np.ndarray
    target_sgd: np.ndarray
    row_id: np.ndarray
    horizon: np.ndarray
    origin: np.ndarray
    static_order: list[str]
    context_order: list[str]
    history_order: list[str]
    sequence_order: list[str]

    @property
    def categorical_columns(self) -> list[str]:
        return self.static_order[:N_CATEGORICAL]

    @property
    def numeric_columns(self) -> list[str]:
        return self.static_order[N_CATEGORICAL:]

    @property
    def anchor_index(self) -> int:
        return self.context_order.index(ANCHOR_SGD)

    @property
    def n_levels(self) -> int:
        return int(self.sequence.shape[1])

    @property
    def n_months(self) -> int:
        return int(self.sequence.shape[2])

    @property
    def n_channels(self) -> int:
        return int(self.sequence.shape[3])


@dataclass(frozen=True)
class Population:
    """One (split, stage, population) slice, row-aligned with its stage arrays."""

    split: str
    stage: str
    population: str
    samples: pd.DataFrame
    arrays: StageArrays

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def anchor_sgd(self) -> np.ndarray:
        return self.arrays.context[:, self.arrays.anchor_index].astype(np.float64)

    @property
    def actual_sgd(self) -> np.ndarray:
        return self.samples["actual_rent_sgd"].to_numpy(dtype=np.float64)

    def subset(self, index: np.ndarray) -> "Population":
        arrays = self.arrays
        return Population(
            split=self.split, stage=self.stage, population=self.population,
            samples=self.samples.iloc[index].reset_index(drop=True),
            arrays=StageArrays(
                split=self.split, stage=self.stage,
                static=arrays.static[index], context=arrays.context[index],
                history=arrays.history[index], sequence=arrays.sequence[index],
                sequence_mask=arrays.sequence_mask[index], target_sgd=arrays.target_sgd[index],
                row_id=arrays.row_id[index], horizon=arrays.horizon[index],
                origin=arrays.origin[index], static_order=arrays.static_order,
                context_order=arrays.context_order, history_order=arrays.history_order,
                sequence_order=arrays.sequence_order),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(contract_dir: Path) -> dict:
    path = Path(contract_dir) / "manifest.json"
    if not path.exists():
        raise ContractNotReady(f"contract manifest missing: {path}")
    payload = json.loads(path.read_text())
    for key in ("contract_id", "contract_version", "splits"):
        if key not in payload:
            raise ContractInvalid(f"manifest missing {key!r}: {path}")
    return payload


def split_spec(manifest: dict, split: str) -> dict:
    if split not in manifest["splits"]:
        raise ContractInvalid(f"split {split!r} not in manifest.splits ({sorted(manifest['splits'])})")
    spec = manifest["splits"][split].get("split")
    if not isinstance(spec, dict):
        raise ContractInvalid(f"splits.{split}.split missing")
    for key in ("cutoff_index", "val_start_index", "val_end_index", "inner_cutoff_index"):
        if key not in spec:
            raise ContractInvalid(f"splits.{split}.split.{key} missing")
    if "feature_order" not in manifest["splits"][split]:
        raise ContractInvalid(f"splits.{split}.feature_order missing")
    return spec


def _stage_meta(manifest: dict, split: str, stage: str) -> dict:
    stages = manifest["splits"][split].get("stages", {})
    if stage not in stages:
        raise ContractInvalid(f"splits.{split}.stages.{stage} missing ({sorted(stages)})")
    return stages[stage]


def load_stage_arrays(contract_dir: Path, manifest: dict, split: str, stage: str) -> StageArrays:
    contract_dir = Path(contract_dir)
    meta = _stage_meta(manifest, split, stage)
    path = contract_dir / "features" / split / f"stage{stage}.npz"
    if not path.exists():
        raise ContractNotReady(f"stage arrays missing: {path}")
    stored = np.load(path, allow_pickle=False)
    missing = [k for k in ARRAY_KEYS if k not in stored.files]
    if missing:
        raise ContractInvalid(f"{path.name} missing arrays {missing}; has {sorted(stored.files)}")

    order = manifest["splits"][split]["feature_order"]
    arrays = StageArrays(
        split=split, stage=stage,
        static=stored["static"], context=stored["context"], history=stored["history"],
        sequence=stored["sequence"], sequence_mask=stored["sequence_mask"],
        target_sgd=stored["target_sgd"], row_id=stored["row_id"],
        horizon=stored["horizon"], origin=stored["origin"],
        static_order=list(order["static"]), context_order=list(order["context"]),
        history_order=list(order["history"]), sequence_order=list(order["sequence"]),
    )
    _validate_arrays(arrays, path, meta)
    return arrays


def _validate_arrays(arrays: StageArrays, path: Path, meta: dict) -> None:
    n = arrays.static.shape[0]
    for name in ("context", "history", "sequence", "sequence_mask", "target_sgd",
                 "row_id", "horizon", "origin"):
        if len(getattr(arrays, name)) != n:
            raise ContractInvalid(f"{path.name}: {name} row count != static's {n}")
    if arrays.static.shape[1] != len(arrays.static_order):
        raise ContractInvalid(
            f"{path.name}: static width {arrays.static.shape[1]} != feature_order.static "
            f"{len(arrays.static_order)}")
    if arrays.context.shape[1] != len(arrays.context_order):
        raise ContractInvalid(f"{path.name}: context width != feature_order.context")
    if arrays.history.shape[1] != len(arrays.history_order):
        raise ContractInvalid(f"{path.name}: history width != feature_order.history")
    if ANCHOR_SGD not in arrays.context_order:
        raise ContractInvalid(f"{path.name}: context has no {ANCHOR_SGD!r} column")
    if arrays.sequence.ndim != 4:
        raise ContractInvalid(
            f"{path.name}: sequence must be (n, levels, months, channels); got {arrays.sequence.shape}")
    if arrays.sequence_mask.shape != arrays.sequence.shape[:3]:
        raise ContractInvalid(
            f"{path.name}: sequence_mask {arrays.sequence_mask.shape} != sequence leading dims "
            f"{arrays.sequence.shape[:3]}")
    padded = arrays.sequence_mask == 0
    if padded.any():
        values = arrays.sequence[padded]
        if not np.isfinite(values).all():
            raise ContractInvalid(f"{path.name}: padded sequence positions are non-finite")
        if np.abs(values).max(initial=0.0) > 0:
            raise ContractInvalid(f"{path.name}: padded sequence positions are non-zero")
    if not np.isin(arrays.sequence_mask, (0.0, 1.0)).all():
        raise ContractInvalid(f"{path.name}: sequence_mask must be 0/1")
    if not np.isfinite(arrays.context).all():
        raise ContractInvalid(f"{path.name}: non-finite context values")
    if not np.isfinite(arrays.target_sgd).all():
        raise ContractInvalid(f"{path.name}: non-finite target values")
    if (arrays.horizon < 1).any():
        raise ContractInvalid(f"{path.name}: horizon must be >= 1")
    declared = meta.get("arrays", {}).get("n_rows")
    if declared is not None and int(declared) != n:
        raise ContractInvalid(f"{path.name}: {n} rows, manifest arrays.n_rows={declared}")
    codes = arrays.static[:, :N_CATEGORICAL]
    if not np.allclose(codes, np.round(codes)):
        raise ContractInvalid(
            f"{path.name}: first {N_CATEGORICAL} static columns are not integer category codes")
    if (codes < 0).any():
        raise ContractInvalid(f"{path.name}: negative category codes")


def load_population(contract_dir: Path, manifest: dict, split: str, stage: str,
                    population: str) -> Population:
    contract_dir = Path(contract_dir)
    csv_path = contract_dir / "samples" / split / f"stage{stage}.csv"
    if not csv_path.exists():
        raise ContractNotReady(f"stage sample table missing: {csv_path}")
    samples = pd.read_csv(csv_path)
    missing = [c for c in SAMPLE_COLUMNS if c not in samples.columns]
    if missing:
        raise ContractInvalid(f"{csv_path.name} missing columns {missing}")
    if len(samples) != samples.shape[0]:
        raise ContractInvalid(f"{csv_path.name}: unreadable rows")

    arrays = load_stage_arrays(contract_dir, manifest, split, stage)
    if len(samples) != len(arrays.static):
        raise ContractInvalid(
            f"{split}/{stage}: csv {len(samples)} rows vs arrays {len(arrays.static)}")

    # Row order: the producer promises CSV order == array order. Verify, do not assume.
    if not np.array_equal(samples["SOURCE_ROW_ID"].to_numpy(), arrays.row_id):
        raise ContractInvalid(f"{split}/{stage}: csv SOURCE_ROW_ID not row-aligned with npz row_id")
    if not np.allclose(samples["actual_rent_sgd"].to_numpy(), arrays.target_sgd, atol=1e-3):
        raise ContractInvalid(f"{split}/{stage}: csv actual_rent_sgd != npz target_sgd")
    if not np.array_equal(samples["horizon_months"].to_numpy(), arrays.horizon):
        raise ContractInvalid(f"{split}/{stage}: csv horizon != npz horizon")
    if not np.array_equal(samples["origin_month_index"].to_numpy(), arrays.origin):
        raise ContractInvalid(f"{split}/{stage}: csv origin != npz origin")
    expected_origin = samples["target_month_index"].to_numpy() - samples["horizon_months"].to_numpy()
    if not np.array_equal(expected_origin, samples["origin_month_index"].to_numpy()):
        raise ContractInvalid(f"{split}/{stage}: origin != target_month - horizon")

    available = sorted(samples["population"].unique())
    if population not in available:
        raise ContractInvalid(f"{split}/{stage}: population {population!r} absent (has {available})")
    mask = (samples["population"] == population).to_numpy()
    index = np.flatnonzero(mask)
    return Population(
        split=split, stage=stage, population=population,
        samples=samples.loc[mask].reset_index(drop=True),
        arrays=StageArrays(
            split=split, stage=stage,
            static=arrays.static[index], context=arrays.context[index],
            history=arrays.history[index], sequence=arrays.sequence[index],
            sequence_mask=arrays.sequence_mask[index], target_sgd=arrays.target_sgd[index],
            row_id=arrays.row_id[index], horizon=arrays.horizon[index], origin=arrays.origin[index],
            static_order=arrays.static_order, context_order=arrays.context_order,
            history_order=arrays.history_order, sequence_order=arrays.sequence_order),
    )


def load_encoding(contract_dir: Path, manifest: dict, split: str, stage: str) -> dict:
    """Load the encoding fitted for one stage.

    v2.1 exports ``encodingA.json`` and ``encodingB.json`` per split: A sees only A/fit
    rows, B only B/fit rows, so the vocabularies and medians legitimately differ. The
    older single ``encoding.json`` is not accepted — silently reusing one stage's
    vocabulary for the other would make the two stages share fitted state.
    """
    directory = Path(contract_dir) / "features" / split
    path = directory / f"encoding{stage}.json"
    if not path.exists():
        raise ContractNotReady(
            f"stage-{stage} encoding missing: {path}\n"
            "v2.1 exports encodingA.json / encodingB.json per split; each stage must use "
            "its own fitted vocabulary and medians.")
    payload = json.loads(path.read_text())
    declared = manifest["splits"][split].get("stages", {}).get(stage, {}).get("encoding", {})
    if declared.get("n_fit_rows") is not None:
        if int(payload.get("n_fit_rows", -1)) != int(declared["n_fit_rows"]):
            raise ContractInvalid(
                f"{path.name}: n_fit_rows {payload.get('n_fit_rows')} != manifest "
                f"{declared['n_fit_rows']}")
    return payload


def encoding_cardinalities(encoding: dict, arrays: StageArrays) -> list[int]:
    """Vocabulary sizes from encoding<stage>.json, including the unknown slot 0."""
    vocab = encoding.get("categorical_vocabularies", {})
    unknown = int(encoding.get("unknown_code", 0))
    sizes = []
    for i, column in enumerate(arrays.categorical_columns):
        observed = int(arrays.static[:, i].max()) + 1
        declared = len(vocab.get(column, {})) if isinstance(vocab.get(column), dict) else len(vocab.get(column, []))
        if declared == 0:
            raise ContractInvalid(f"encoding has no vocabulary for categorical column {column!r}")
        if observed > declared:
            raise ContractInvalid(
                f"{column}: category code {observed - 1} observed but vocabulary has {declared} "
                f"entries (unknown slot {unknown})")
        sizes.append(declared)
    return sizes


def validate_boundaries(manifest: dict, split: str, a_fit: Population, a_holdout: Population,
                        b_fit: Population, b_validation: Population) -> dict:
    """Check the A/B information boundaries for the v2.1 population set.

    v2.1 defines exactly four populations: A/fit, A/holdout, B/fit, B/validation. Stage A
    has **no** validation population by design — A exists only to fit state and to pick an
    epoch count, so an A/validation would be an unused copy of B/validation.

    Blocking (``ContractInvalid``): anything that would leak a label or break alignment.
    Deviations (returned, non-blocking): a contract that departs from the design
    requirement; the caller must downgrade ``result_status`` and never treat those numbers
    as Gate 1 results.
    """
    spec = split_spec(manifest, split)
    inner_cutoff = int(spec["inner_cutoff_index"])
    cutoff = int(spec["cutoff_index"])
    val_start, val_end = int(spec["val_start_index"]), int(spec["val_end_index"])
    deviations: list[dict] = []

    # --- blocking: fitting populations carry no post-cutoff labels ---
    if (a_fit.samples["target_month_index"] > inner_cutoff).any():
        raise ContractInvalid(
            f"A/fit: target month beyond inner cutoff {inner_cutoff}; labels would leak")
    if (a_fit.arrays.origin > inner_cutoff).any():
        raise ContractInvalid(f"A/fit: origin beyond inner cutoff {inner_cutoff}")
    if (b_fit.samples["target_month_index"] > cutoff).any():
        raise ContractInvalid(f"B/fit: target month beyond cutoff {cutoff}; labels would leak")
    if (b_fit.arrays.origin > cutoff).any():
        raise ContractInvalid(f"B/fit: origin beyond cutoff {cutoff}")

    # --- blocking: A/holdout must be strictly after A/fit's labels ---
    fit_max_target = int(a_fit.samples["target_month_index"].max())
    if (a_holdout.samples["target_month_index"] <= fit_max_target).any():
        raise ContractInvalid(
            f"A/holdout: target months overlap A/fit's last target month {fit_max_target}")

    # --- blocking: B/validation is one frozen origin at the split cutoff ---
    origins = np.unique(b_validation.arrays.origin)
    if len(origins) != 1:
        raise ContractInvalid(
            f"B/validation: must use one frozen origin; found {len(origins)}")
    if int(origins[0]) != cutoff:
        raise ContractInvalid(
            f"B/validation: origin {int(origins[0])} != cutoff {cutoff}")
    months = b_validation.samples["target_month_index"].to_numpy()
    if months.min() < val_start or months.max() > val_end:
        raise ContractInvalid(
            f"B/validation: target months outside [{val_start}, {val_end}]")

    # --- deviation: A/holdout should be evaluated from the inner cutoff ---
    holdout_origins = np.unique(a_holdout.arrays.origin)
    n_months = int(spec["n_validation_months"])
    if len(holdout_origins) != 1 or int(holdout_origins[0]) != inner_cutoff:
        deviations.append({
            "requirement": "A/holdout evaluated from the single inner cutoff",
            "expected_origin": inner_cutoff,
            "found_origins": [int(x) for x in holdout_origins[:8]],
            "n_distinct_origins": int(len(holdout_origins)),
            "impact": ("the early-stopping signal mixes origins, so it does not match the "
                       "one-origin multi-month-ahead task"),
            "action": "not a Gate 1 result",
        })
    holdout_horizons = np.unique(a_holdout.samples["horizon_months"].to_numpy())
    if len(holdout_horizons) and int(holdout_horizons.max()) > n_months:
        deviations.append({
            "requirement": "A/holdout horizon bounded by the inner window",
            "expected_max_horizon": n_months,
            "found_max_horizon": int(holdout_horizons.max()),
            "impact": "holdout horizons exceed the inner window they are meant to represent",
            "action": "not a Gate 1 result",
        })

    return {
        "split": split, "inner_cutoff_index": inner_cutoff, "cutoff_index": cutoff,
        "val_start_index": val_start, "val_end_index": val_end,
        "populations": {"A/fit": len(a_fit), "A/holdout": len(a_holdout),
                        "B/fit": len(b_fit), "B/validation": len(b_validation)},
        "a_fit_origin_range": [int(a_fit.arrays.origin.min()), int(a_fit.arrays.origin.max())],
        "a_fit_target_range": [int(a_fit.samples["target_month_index"].min()),
                               int(a_fit.samples["target_month_index"].max())],
        "a_holdout_origin": [int(x) for x in holdout_origins],
        "a_holdout_target_range": [int(a_holdout.samples["target_month_index"].min()),
                                   int(a_holdout.samples["target_month_index"].max())],
        "a_holdout_horizon_range": [int(holdout_horizons.min()), int(holdout_horizons.max())],
        "b_fit_origin_range": [int(b_fit.arrays.origin.min()), int(b_fit.arrays.origin.max())],
        "b_validation_origin": int(origins[0]),
        "stage_a_has_validation": False,
        "deviations": deviations,
        "blocking_checks": "passed",
    }


def contract_fingerprint(contract_dir: Path) -> dict:
    directory = Path(contract_dir)
    return {str(p.relative_to(directory)): sha256_file(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}

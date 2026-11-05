"""Export the aligned Gate-1 artifacts for both the tree and the neural controls.

Outputs (one directory per ``contract_id``)::

    <reports>/fusion_contract_v2/<contract_id>/
        manifest.json                  contract id, hashes, counts, stage summary
        samples/<split>/<stage>.csv    one row per supervised sample
        features/<split>/<stage>.npz   scalars, context, history, sequence, mask
        feature_order.json             exact column order per block
        fixtures/synthetic_*.npz       tiny synthetic fixture for consumers

Design notes
------------
* Thousands of rows, so CSVs carry the sample table and per-row scalars; arrays
  are stored in ``.npz`` keyed by split/stage and row order matches the CSV.  Row
  order is ``SOURCE_ROW_ID``-sorted inside each (split, stage, population).
* Only rows and boundaries are exported.  No model-specific transformation is
  applied here: category vocabularies and numeric medians are exported as data so
  each consumer applies its own documented encoding (native categoricals for the
  tree, embeddings for the network) to identical inputs.
* Stage A and stage B never share fitted state.  The panel and the reference
  statistics for stage A are built with ``fit_end = cutoff - INNER_HOLDOUT_MONTHS``;
  for stage B with ``fit_end = cutoff``.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v2 import contract, panel as panel_module

REQUIRED_SOURCE_COLUMNS = (
    "SOURCE_ROW_ID", "RENT_APPROVAL_DATE", "APPROVAL_MONTH_INDEX", "MONTHLY_RENT",
    "TOWN_CLEAN", "STREET_CLEAN", "BLOCK_CLEAN", "SUBZONE", "PLANNING_AREA",
    "REGION", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN",
    "FLOOR_AREA_SQM", "YEARS_SINCE_LEASE_COMMENCEMENT", "MAX_FLOOR",
    "BUILDING_AGE_YEARS", "LATITUDE", "LONGITUDE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash() -> dict:
    """Hashes of every module in this package, recorded at launch."""
    out = {}
    for path in sorted(contract.PACKAGE_DIR.glob("*.py")):
        out[path.name] = sha256_file(path)
    return out


def load_source(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or contract.SOURCE)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"source missing columns: {missing}")
    if not frame[contract.ID_COL].is_unique:
        raise ValueError("SOURCE_ROW_ID is not unique")
    if not np.array_equal(frame[contract.ID_COL].to_numpy(), np.arange(len(frame))):
        raise ValueError("SOURCE_ROW_ID does not match physical row order")

    dates = pd.to_datetime(frame[contract.DATE_COL], format="%Y-%m", errors="coerce")
    if dates.isna().any():
        raise ValueError("unparseable RENT_APPROVAL_DATE")
    derived = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    if not np.array_equal(derived.to_numpy(), frame[contract.MONTH_COL].to_numpy()):
        raise ValueError("APPROVAL_MONTH_INDEX disagrees with RENT_APPROVAL_DATE")
    if frame[contract.TARGET].isna().any():
        raise ValueError("missing target")

    frame = frame.copy()
    frame["MONTH_INDEX"] = frame[contract.MONTH_COL].astype(np.int64)
    frame["BLOCK_ID"] = (
        frame["TOWN_CLEAN"].fillna("__missing__").astype(str)
        .str.cat(frame["STREET_CLEAN"].fillna("__missing__").astype(str), sep="|")
        .str.cat(frame["BLOCK_CLEAN"].fillna("__missing__").astype(str), sep="|")
    )
    for column in contract.STATIC_FLAG_COLUMNS:
        base = column.replace("__missing", "")
        frame[column] = frame[base].isna().astype(float)
    return frame


# ---------------------------------------------------------------- samples
def build_samples(frame: pd.DataFrame, split: dict, stage: str) -> pd.DataFrame:
    """Supervised sample table for one split and stage.

    Training population: fitting-month rows get ``horizon`` drawn once from a
    frozen RNG and ``origin = target_month - h``; rows with a negative origin are
    dropped (no history to read).
    Validation population: every row in ``val_start..val_end``, ``origin = cutoff``.
    Inner holdout population (stage A only): fitting rows in
    ``(inner_cutoff, cutoff]`` with a sampled horizon as well.
    """
    if stage not in contract.STAGES:
        raise ValueError(stage)
    cutoff = split["cutoff_index"]
    inner = split.get("inner_cutoff_index")
    if inner is None:
        inner = cutoff - contract.INNER_HOLDOUT_MONTHS
    if inner < 0:
        raise ValueError(f"fitting period shorter than the {contract.INNER_HOLDOUT_MONTHS}-month inner holdout")
    month = frame["MONTH_INDEX"].to_numpy()

    rng = np.random.default_rng(contract.SAMPLE_SEED)
    # Draw for every row once so that the same target row always receives the
    # same horizon regardless of which population it lands in.
    horizons = rng.integers(1, contract.MAX_HORIZON + 1, size=len(frame))

    rows = []
    if stage == "A":
        fit_population = month <= inner
        holdout_population = (month > inner) & (month <= cutoff)
        specs = [("fit", fit_population), ("holdout", holdout_population)]
    else:
        fit_population = month <= cutoff
        specs = [("fit", fit_population)]

    for population, mask in specs:
        selected = frame.loc[mask]
        target_month = selected["MONTH_INDEX"].to_numpy()
        if population == "holdout":
            # The inner holdout is an early-stopping *evaluation* set.  Its
            # history boundary must be the frozen inner snapshot, exactly as
            # outer validation is frozen at the cutoff -- otherwise a sampled
            # horizon would let a holdout row read months inside its own
            # evaluation window and its context h/origin would disagree with the
            # history it was actually given.
            horizon = target_month - inner
            origin = np.full(len(selected), inner, dtype=np.int64)
            if (horizon < 1).any():
                raise AssertionError("inner holdout row at or before the inner cutoff")
            keep = np.ones(len(selected), dtype=bool)
        else:
            horizon = horizons[mask]
            origin = target_month - horizon
            keep = origin >= 0
        rows.append(pd.DataFrame({
            contract.ID_COL: selected[contract.ID_COL].to_numpy()[keep],
            "split_id": split["id"],
            "stage": stage,
            "population": population,
            "target_month_index": target_month[keep],
            "origin_month_index": origin[keep],
            "horizon_months": horizon[keep],
            "actual_rent_sgd": selected[contract.TARGET].to_numpy(dtype=float)[keep],
            "approval_month": selected[contract.DATE_COL].to_numpy()[keep],
        }))

    # Outer validation rows appear only in stage B.  Including them in stage A
    # would expose the outer holdout to anything that fits or selects there.
    valid_mask = (month >= split["val_start_index"]) & (month <= split["val_end_index"])
    if stage == "A":
        valid_mask = np.zeros(len(frame), dtype=bool)
    if valid_mask.any():
        selected = frame.loc[valid_mask]
        target_month = selected["MONTH_INDEX"].to_numpy()
        rows.append(pd.DataFrame({
            contract.ID_COL: selected[contract.ID_COL].to_numpy(),
            "split_id": split["id"],
            "stage": stage,
            "population": "validation",
            "target_month_index": target_month,
            "origin_month_index": np.full(len(selected), cutoff, dtype=np.int64),
            "horizon_months": target_month - cutoff,
            "actual_rent_sgd": selected[contract.TARGET].to_numpy(dtype=float),
            "approval_month": selected[contract.DATE_COL].to_numpy(),
        }))

    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values([contract.ID_COL, "population"]).reset_index(drop=True)

    # Invariants that make the table usable as a comparison contract.
    if table["origin_month_index"].to_numpy().max() > cutoff:
        raise AssertionError("origin exceeds the cutoff")
    if (table["horizon_months"] <= 0).any():
        raise AssertionError("non-positive horizon")
    if (table["horizon_months"] > contract.MAX_HORIZON).any():
        raise AssertionError("horizon exceeds the maximum")
    validation = table[table["population"] == "validation"]
    if len(validation) and not (validation["horizon_months"] ==
                                validation["target_month_index"] - cutoff).all():
        raise AssertionError("validation horizon != target - cutoff")
    return table


# ---------------------------------------------------------------- features
def _codes_for(frame: pd.DataFrame, panel: panel_module.Panel) -> dict[str, np.ndarray]:
    """Map each row's level key onto the panel vocabulary (UNSEEN when new)."""
    out = {}
    for level, key_columns in panel_module.LEVEL_KEYS.items():
        series = panel_module._key_series(frame, key_columns)
        mapping = {name: index for index, name in enumerate(panel.level_maps[level])}
        out[level] = np.array([mapping.get(value, panel_module.UNSEEN) for value in series],
                              dtype=np.int64)
    return out


@dataclass
class Export:
    arrays: dict[str, np.ndarray]
    feature_order: dict[str, list]
    encoding: dict | None = None

    def save(self, directory: Path, name: str) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.npz"
        np.savez_compressed(path, **self.arrays)
        return {
            "path": str(path.relative_to(contract.ROOT)),
            "sha256": sha256_file(path),
            "n_rows": int(len(next(iter(self.arrays.values())))),
            "keys": sorted(self.arrays),
        }


UNKNOWN_CODE = 0


def fit_vocabularies(frame: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Category vocabularies and numeric medians from the stage's fit rows only.

    Codes are ``1..K`` in sorted string order with ``0`` reserved for
    unknown/unseen, so a category that appears only in holdout or validation data
    maps to ``0`` and the vocabulary never depends on which rows are scored.
    """
    fit_ids = table.loc[table["population"] == "fit", contract.ID_COL].to_numpy()
    if len(fit_ids) == 0:
        raise ValueError("stage has no fit population")
    fit_rows = frame.loc[frame[contract.ID_COL].isin(fit_ids)] \
        .set_index(contract.ID_COL).loc[fit_ids].reset_index()

    vocabularies: dict[str, dict[str, int]] = {}
    for column in contract.CATEGORICAL_COLUMNS:
        values = sorted(fit_rows[column].fillna("__missing__").astype(str).unique().tolist())
        mapping = {str(value): index + 1 for index, value in enumerate(values)}
        mapping["<unknown>"] = UNKNOWN_CODE
        vocabularies[column] = mapping

    medians = {}
    for column in list(contract.STATIC_NUMERIC_COLUMNS) + list(contract.STATIC_FLAG_COLUMNS):
        median = float(pd.to_numeric(fit_rows[column], errors="coerce").median())
        if not np.isfinite(median):
            raise ValueError(f"no finite fit-population median for {column}")
        medians[column] = median

    return {
        "contract_version": contract.CONTRACT_VERSION,
        "categorical_vocabularies": vocabularies,
        "unknown_code": UNKNOWN_CODE,
        "numeric_medians": medians,
        "categorical_columns": list(contract.CATEGORICAL_COLUMNS),
        "fitted_on": "population == 'fit' of this stage only",
        "n_fit_rows": int(len(fit_rows)),
    }


def apply_vocabularies(rows: pd.DataFrame, encoding: dict) -> np.ndarray:
    """Map every row's categories through the frozen vocabulary.

    Unseen values become ``UNKNOWN_CODE``.  The mapping is asserted to round-trip
    so the exported codes cannot silently disagree with the saved JSON.
    """
    matrix = np.zeros((len(rows), len(contract.CATEGORICAL_COLUMNS)), dtype=np.float64)
    vocabularies = encoding["categorical_vocabularies"]
    for index, column in enumerate(contract.CATEGORICAL_COLUMNS):
        mapping = vocabularies[column]
        values = rows[column].fillna("__missing__").astype(str)
        matrix[:, index] = values.map(lambda value: mapping.get(value, UNKNOWN_CODE)).to_numpy()
        # round-trip: every emitted code must exist in the saved mapping
        emitted = set(np.unique(matrix[:, index]).astype(int).tolist())
        allowed = set(int(code) for code in mapping.values())
        if not emitted <= allowed:
            raise AssertionError(f"{column}: emitted codes {sorted(emitted - allowed)} "
                                 f"are absent from the frozen vocabulary")
    return matrix


def build_features(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    panel: panel_module.Panel,
) -> Export:
    """Scalars, context, history and sequence arrays for the sample table."""
    rows = frame.loc[frame[contract.ID_COL].isin(table[contract.ID_COL])].copy()
    rows = rows.set_index(contract.ID_COL).loc[table[contract.ID_COL].to_numpy()].reset_index()
    if not np.array_equal(rows[contract.ID_COL].to_numpy(), table[contract.ID_COL].to_numpy()):
        raise AssertionError("row alignment broke between the sample table and the source")

    origin = table["origin_month_index"].to_numpy(dtype=np.int64)
    # Stage-A holdout rows can have an origin beyond the stage-A panel; their
    # boundary is clamped so inner-holdout labels never enter stage-A state.
    boundary = np.minimum(origin, panel.fit_end)

    codes = _codes_for(rows, panel)
    stats = panel.stats(boundary, codes)

    arrays: dict[str, np.ndarray] = {}
    feature_order: dict[str, list] = {}

    # static block (target-independent).  Categories use the stage's frozen
    # vocabulary, fitted on fit rows only -- never pd.factorize over the whole
    # table, which would let holdout/validation rows change the codes.
    encoding = fit_vocabularies(frame, table)
    categorical_matrix = apply_vocabularies(rows, encoding)
    static_columns = list(contract.static_columns())
    static = np.empty((len(rows), len(static_columns)), dtype=np.float64)
    categorical_index = 0
    for index, column in enumerate(static_columns):
        if column in contract.CATEGORICAL_COLUMNS:
            static[:, index] = categorical_matrix[:, categorical_index]
            categorical_index += 1
        else:
            static[:, index] = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)
    arrays["static"] = static.astype(np.float32)
    feature_order["static"] = static_columns

    target_month = table["target_month_index"].to_numpy(dtype=np.int64)
    horizon = table["horizon_months"].to_numpy(dtype=np.int64)
    context = np.column_stack([
        horizon.astype(np.float64),
        target_month.astype(np.float64),
        origin.astype(np.float64),
        np.sin(2 * np.pi * np.mod(target_month, 12) / 12),
        np.cos(2 * np.pi * np.mod(target_month, 12) / 12),
        np.sin(2 * np.pi * np.mod(origin, 12) / 12),
        np.cos(2 * np.pi * np.mod(origin, 12) / 12),
    ])
    for index, column in enumerate(contract.ANCHOR_COLUMNS):
        context = np.column_stack([context, stats[column]])
    arrays["context"] = context.astype(np.float32)
    feature_order["context"] = list(contract.CONTEXT_COLUMNS) + list(contract.ANCHOR_COLUMNS)

    history = np.column_stack([stats[column] for column in contract.history_columns()])
    arrays["history"] = history.astype(np.float32)
    feature_order["history"] = list(contract.history_columns())

    sequence, mask = panel.sequences(boundary, codes)
    arrays["sequence"] = sequence
    arrays["sequence_mask"] = mask
    feature_order["sequence"] = [f"{level}::{channel}"
                                 for level in contract.HISTORY_LEVELS
                                 for channel in contract.PANEL_CHANNELS]

    arrays["target_sgd"] = table["actual_rent_sgd"].to_numpy(dtype=np.float32)
    arrays["row_id"] = table[contract.ID_COL].to_numpy(dtype=np.int64)
    arrays["horizon"] = horizon.astype(np.int32)
    arrays["origin"] = origin.astype(np.int32)
    export = Export(arrays=arrays, feature_order=feature_order)
    export.encoding = encoding
    return export


# ---------------------------------------------------------------- fixtures
def synthetic_fixture(seed: int = 7, months: int = 9, per_month: int = 4) -> pd.DataFrame:
    """Small deterministic frame with exact, hand-checkable rents.

    Rents increase by 100 per month and depend on flat type, so a consumer can
    verify an origin boundary by arithmetic instead of trusting a helper.
    """
    rows = []
    row_id = 0
    for month in range(months):
        for index in range(per_month):
            flat_type = ("2 room", "3 room", "4 room", "5 room")[index % 4]
            base = {"2 room": 2000.0, "3 room": 2600.0,
                    "4 room": 3200.0, "5 room": 3900.0}[flat_type]
            rent = base + 100.0 * month
            rows.append({
                "SOURCE_ROW_ID": row_id,
                "RENT_APPROVAL_DATE": contract.month_label(month),
                "APPROVAL_MONTH_INDEX": month,
                "MONTHLY_RENT": rent,
                "TOWN_CLEAN": f"TOWN{index % 2}",
                "STREET_CLEAN": f"STREET{index % 2}",
                "BLOCK_CLEAN": str(100 + index % 2),
                "SUBZONE": "SUB1", "PLANNING_AREA": "PA1", "REGION": "R1",
                "FLAT_TYPE_CLEAN": flat_type,
                "FLAT_MODEL_CLEAN": "Improved",
                "FURNISHED_CLEAN": "yes",
                "FLOOR_AREA_SQM": 60.0 + 10.0 * (index % 4),
                "YEARS_SINCE_LEASE_COMMENCEMENT": 10.0,
                "MAX_FLOOR": 12.0,
                "BUILDING_AGE_YEARS": 20.0,
                "LATITUDE": 1.30, "LONGITUDE": 103.80,
                "NEAREST_OPEN_MRT_KM": 0.5, "OPEN_MRT_COUNT_500M": 1.0,
                "OPEN_MRT_COUNT_1000M": 2.0, "OPEN_MRT_COUNT_2000M": 4.0,
                "NEAREST_MALL_KM": 0.6, "MALL_COUNT_500M": 1.0,
                "MALL_COUNT_1000M": 2.0, "MALL_COUNT_2000M": 5.0,
                "NEAREST_PRIMARY_SCHOOL_KM": 0.4, "PRIMARY_SCHOOL_COUNT_500M": 1.0,
                "PRIMARY_SCHOOL_COUNT_1000M": 2.0, "PRIMARY_SCHOOL_COUNT_2000M": 3.0,
                "DISTANCE_TO_CBD_KM": 8.0,
            })
            row_id += 1
    frame = pd.DataFrame(rows)
    frame["MONTH_INDEX"] = frame["APPROVAL_MONTH_INDEX"].astype(np.int64)
    frame["BLOCK_ID"] = (
        frame["TOWN_CLEAN"].str.cat(frame["STREET_CLEAN"], sep="|")
        .str.cat(frame["BLOCK_CLEAN"], sep="|"))
    for column in contract.STATIC_FLAG_COLUMNS:
        frame[column] = 0.0
    return frame


def save_fixture(directory: Path) -> Path:
    frame = synthetic_fixture()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "synthetic_fixture.csv"
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- driver
def export_split(frame: pd.DataFrame, split_id: str, output: Path,
                 source_hash: str, hashes: dict) -> dict:
    split = contract.split_spec(split_id)
    summary: dict = {
        "split": {key: value for key, value in split.items()},
        "stages": {},
        "feature_order": {},
        "artifacts": {},
    }
    (output / "samples" / split_id).mkdir(parents=True, exist_ok=True)
    (output / "features" / split_id).mkdir(parents=True, exist_ok=True)

    for stage in contract.STAGES:
        fit_end = split["inner_cutoff_index"] if stage == "A" else split["cutoff_index"]
        panel = panel_module.Panel.build(frame, fit_end)
        table = build_samples(frame, split, stage)
        export = build_features(frame, table, panel)

        sample_path = output / "samples" / split_id / f"stage{stage}.csv"
        table.to_csv(sample_path, index=False)
        artifact = export.save(output / "features" / split_id, f"stage{stage}")

        encoding_path = output / "features" / split_id / f"encoding{stage}.json"
        encoding_path.write_text(json.dumps(export.encoding, indent=2) + "\n")

        if export.encoding is None:  # pragma: no cover - defensive
            raise AssertionError("build_features produced no encoding")
        summary["stages"][stage] = {
            "fit_end_month": contract.month_label(fit_end),
            "fit_end_index": fit_end,
            "panel_levels": {level: size for level, size in panel.sizes.items()},
            "populations": table.groupby("population").size().to_dict(),
            "n_rows": len(table),
            "horizon_range": [int(table["horizon_months"].min()), int(table["horizon_months"].max())],
            "row_id_sha256": sha256_text(",".join(map(str, table[contract.ID_COL].tolist()))),
            "target_sha256": sha256_text(",".join(f"{value:.6f}" for value in table["actual_rent_sgd"])),
            "sample_csv": str(sample_path.relative_to(contract.ROOT)),
            "sample_csv_sha256": sha256_file(sample_path),
            "arrays": artifact,
            "encoding": {
                "path": str(encoding_path.relative_to(contract.ROOT)),
                "sha256": sha256_file(encoding_path),
                "n_fit_rows": export.encoding["n_fit_rows"],
            },
        }
        summary["feature_order"].update(export.feature_order)

    return summary


def main(split_ids: tuple[str, ...] = tuple(contract.SPLITS), create_fixture: bool = True) -> Path:
    frame = load_source()
    source_hash = sha256_file(contract.SOURCE)
    hashes = code_hash()
    contract_id = f"{contract.CONTRACT_VERSION}__{source_hash[:12]}"
    output = contract.REPORT_ROOT / contract_id
    output.mkdir(parents=True, exist_ok=True)

    if create_fixture:
        save_fixture(output / "fixtures")

    manifest = {
        "contract_version": contract.CONTRACT_VERSION,
        "contract_id": contract_id,
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "source": str(contract.SOURCE.relative_to(contract.ROOT)),
        "source_sha256": source_hash,
        "code_hashes": hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_source_rows": len(frame),
        "contract_identity": contract.contract_identity(),
        "arms": {arm: contract.arm_scalar_columns(arm) for arm in contract.ARM_INPUTS},
        "splits": {},
    }
    for split_id in split_ids:
        print(f"exporting split {split_id} ...", flush=True)
        manifest["splits"][split_id] = export_split(frame, split_id, output, source_hash, hashes)

    if sha256_file(contract.SOURCE) != source_hash:
        raise RuntimeError("source data changed during export")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output.relative_to(contract.ROOT)}", flush=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=list(contract.SPLITS),
                        help="split to export (repeatable); default all")
    parser.add_argument("--no-fixture", action="store_true")
    args = parser.parse_args()
    main(tuple(args.split) if args.split else tuple(contract.SPLITS),
         create_fixture=not args.no_fixture)

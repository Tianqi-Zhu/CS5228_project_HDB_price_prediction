"""Producer-shaped fixture for consumer integration tests.

Unlike the earlier fixture, this writes the **producer's real layout** — manifest with
``splits[split] = {split, stages: {A: {...}, B: {...}}, feature_order}``,
``samples/<split>/stageA.csv``, ``features/<split>/stageA.npz``, ``encoding.json`` — so
the consumer is exercised against the schema it will actually meet.

It is still labelled ``contract_kind: synthetic_fixture``: history values are generated,
not derived from rental data. Its purpose is to prove the consumer's boundary handling,
not to produce scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["global", "flat_type", "town_ft", "block_ft"]
CHANNELS = ["log_mean_rent", "log_mean_psm", "log_count", "month_sin", "month_cos", "observed"]
HISTORY_STATS = ["hist_rent", "hist_psm", "hist_n", "hist_months", "hist_recency", "hist_exact"]

STATIC_ORDER = ["TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN",
                "FLOOR_AREA_SQM", "YEARS_SINCE_LEASE_COMMENCEMENT", "LATITUDE", "LONGITUDE",
                "MAX_FLOOR", "BUILDING_AGE_YEARS", "NEAREST_OPEN_MRT_KM"]
CONTEXT_ORDER = ["horizon_months", "target_month_index", "origin_month_index",
                 "target_month_sin", "target_month_cos", "anchor_rent_sgd", "anchor_n",
                 "anchor_missing"]

SPLIT_SPECS = {
    "orig": {"cutoff_index": 38, "val_start_index": 39, "val_end_index": 50,
             "inner_cutoff_index": 26, "n_validation_months": 12},
    "early": {"cutoff_index": 26, "val_start_index": 27, "val_end_index": 38,
              "inner_cutoff_index": 14, "n_validation_months": 12},
}


def _history_order() -> list[str]:
    return [f"{stat}__{level}" for level in LEVELS for stat in HISTORY_STATS]


def _sequence_order() -> list[str]:
    return [f"{level}::{channel}" for level in LEVELS for channel in CHANNELS]


def _make_arrays(rng, spec, n_fit, n_holdout, n_valid, *, holdout_single_origin: bool):
    """Build (samples, arrays) for one stage, honouring the split's month boundaries."""
    inner_cutoff = spec["inner_cutoff_index"]
    cutoff = spec["cutoff_index"]
    val_start, val_end = spec["val_start_index"], spec["val_end_index"]

    # --- A/fit: origins inside the inner window, targets <= inner cutoff
    fit_targets = rng.integers(1, inner_cutoff + 1, size=n_fit)
    fit_horizons = rng.integers(1, 17, size=n_fit)
    fit_origins = fit_targets - fit_horizons
    keep = fit_origins >= 0
    fit_targets, fit_horizons, fit_origins = fit_targets[keep], fit_horizons[keep], fit_origins[keep]

    # --- A/holdout: targets just after the inner cutoff
    if holdout_single_origin:
        hold_origins = np.full(n_holdout, inner_cutoff)
        hold_horizons = rng.integers(1, 13, size=n_holdout)
        hold_targets = hold_origins + hold_horizons
    else:
        hold_targets = rng.integers(inner_cutoff + 1, cutoff + 1, size=n_holdout)
        hold_horizons = rng.integers(1, 17, size=n_holdout)
        hold_origins = np.maximum(hold_targets - hold_horizons, 0)

    # --- validation: one frozen origin at the split cutoff
    val_origins = np.full(n_valid, cutoff)
    val_horizons = rng.integers(1, val_end - val_start + 2, size=n_valid)
    val_targets = val_origins + val_horizons

    blocks = {
        "fit": (fit_origins, fit_horizons, fit_targets),
        "holdout": (hold_origins, hold_horizons, hold_targets),
        "validation": (val_origins, val_horizons, val_targets),
    }
    return blocks


def _populate(rng, blocks, populations, n_levels=len(LEVELS), n_months=12,
              channels=len(CHANNELS)):
    rows, arrays = [], {key: [] for key in
                        ("static", "context", "history", "sequence", "sequence_mask",
                         "target_sgd", "row_id", "horizon", "origin")}
    next_row = 1000
    for population in populations:
        origins, horizons, targets = blocks[population]
        n = len(origins)
        anchor_sgd = rng.uniform(1200.0, 4300.0, size=n)

        target = anchor_sgd + rng.normal(0, 140, n)
        static = np.empty((n, len(STATIC_ORDER)), dtype=np.float32)
        static[:, 0] = rng.integers(0, 26, n)
        static[:, 1] = rng.integers(0, 6, n)
        static[:, 2] = rng.integers(0, 20, n)
        static[:, 3] = 0
        static[:, 4] = rng.uniform(31, 215, n)
        static[:, 5] = rng.uniform(0, 60, n)
        static[:, 5][::37] = np.nan
        static[:, 6] = rng.uniform(1.2, 1.5, n)
        static[:, 7] = rng.uniform(103.6, 104.0, n)
        static[:, 8] = rng.uniform(3, 50, n)
        static[:, 8][::53] = np.nan
        static[:, 9] = rng.uniform(0, 76, n)
        static[:, 9][::41] = np.nan
        static[:, 10] = rng.uniform(0.05, 2.5, n)

        context = np.empty((n, len(CONTEXT_ORDER)), dtype=np.float32)
        context[:, 0] = horizons
        context[:, 1] = targets
        context[:, 2] = origins
        context[:, 3] = np.sin(2 * np.pi * targets % 12 / 12)
        context[:, 4] = np.cos(2 * np.pi * targets % 12 / 12)
        context[:, 5] = anchor_sgd
        context[:, 6] = rng.integers(1, 400, n)
        context[:, 7] = rng.uniform(size=n) < 0.01

        history = rng.normal(size=(n, len(_history_order()))).astype(np.float32)
        # deliberately large-scale count columns, like the producer's hist_n
        for j, name in enumerate(_history_order()):
            if name.startswith("hist_n__"):
                history[:, j] = rng.uniform(1, 90000, n)
            if name == "hist_recency__block_ft":
                history[::11, j] = np.nan

        sequence = rng.normal(size=(n, n_levels, n_months, channels)).astype(np.float32)
        mask = np.ones((n, n_levels, n_months), dtype=np.float32)
        mask[::7, :, :3] = 0.0
        sequence[mask == 0] = 0.0

        row_ids = np.arange(next_row, next_row + n)
        next_row += n

        rows.append(pd.DataFrame({
            "SOURCE_ROW_ID": row_ids, "split_id": "orig", "stage": "A",
            "population": population,
            "target_month_index": targets, "origin_month_index": origins,
            "horizon_months": horizons, "actual_rent_sgd": target,
            "approval_month": ["2024-01"] * n,
        }))
        for key, value in (("static", static), ("context", context), ("history", history),
                           ("sequence", sequence), ("sequence_mask", mask),
                           ("target_sgd", target.astype(np.float32)), ("row_id", row_ids),
                           ("horizon", horizons.astype(np.int32)),
                           ("origin", origins.astype(np.int32))):
            arrays[key].append(value)

    samples = pd.concat(rows, ignore_index=True)
    combined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    return samples, combined


def write_producer_fixture(root: Path, *, contract_id: str = "producer-fixture",
                           seed: int = 11, n_fit: int = 900, n_holdout: int = 500,
                           n_valid: int = 700, split: str = "orig",
                           holdout_single_origin: bool = True) -> Path:
    """Write a fixture in the producer's real directory layout."""
    rng = np.random.default_rng(seed)
    spec = SPLIT_SPECS[split]
    directory = Path(root) / contract_id
    (directory / "samples" / split).mkdir(parents=True, exist_ok=True)
    (directory / "features" / split).mkdir(parents=True, exist_ok=True)

    stage_a_blocks = _make_arrays(rng, spec, n_fit, n_holdout, n_valid,
                                  holdout_single_origin=holdout_single_origin)
    # v2.1: stage A has fit + holdout only; validation belongs to stage B
    a_blocks = {k: stage_a_blocks[k] for k in ("fit", "holdout")}
    samples, arrays = _populate(rng, a_blocks, ["fit", "holdout"])
    samples.to_csv(directory / "samples" / split / "stageA.csv", index=False)
    np.savez(directory / "features" / split / "stageA.npz", **arrays)

    # stage B: fit reuses the A/fit rows plus the holdout rows (labels now inside the
    # fitting period); validation is the same frozen-origin block.
    b_fit_origins = np.concatenate([stage_a_blocks["fit"][0], stage_a_blocks["holdout"][0]])
    b_fit_horizons = np.concatenate([stage_a_blocks["fit"][1], stage_a_blocks["holdout"][1]])
    b_fit_targets = b_fit_origins + b_fit_horizons
    stage_b_blocks = {
        "fit": (b_fit_origins, b_fit_horizons, b_fit_targets),
        "validation": stage_a_blocks["validation"],
    }
    b_samples, b_arrays = _populate(rng, stage_b_blocks, ["fit", "validation"])
    b_samples["stage"] = "B"
    b_samples.to_csv(directory / "samples" / split / "stageB.csv", index=False)
    np.savez(directory / "features" / split / "stageB.npz", **b_arrays)

    def _encoding(stage: str, n_fit_rows: int, n_models: int) -> dict:
        vocab = {
            "TOWN_CLEAN": {f"town{i}": i + 1 for i in range(26)},
            "FLAT_TYPE_CLEAN": {f"ft{i}": i + 1 for i in range(6)},
            "FLAT_MODEL_CLEAN": {f"model{i}": i + 1 for i in range(n_models)},
            "FURNISHED_CLEAN": {"yes": 1},
        }
        for column in vocab:
            vocab[column]["<unknown>"] = 0
        return {
            "contract_version": "producer-fixture-1",
            "categorical_vocabularies": vocab,
            "unknown_code": 0,
            "numeric_medians": {"FLOOR_AREA_SQM": 92.0},
            "categorical_columns": STATIC_ORDER[:4],
            "fitted_on": f"population == 'fit' of stage {stage} only",
            "n_fit_rows": int(n_fit_rows),
        }

    n_a_fit = int((samples.population == "fit").sum())
    n_b_fit = int((b_samples.population == "fit").sum())
    (directory / "features" / split / "encodingA.json").write_text(
        json.dumps(_encoding("A", n_a_fit, 20), indent=2))
    (directory / "features" / split / "encodingB.json").write_text(
        json.dumps(_encoding("B", n_b_fit, 20), indent=2))

    n_a = int(len(samples))
    n_b = int(len(b_samples))
    manifest = {
        "contract_version": "producer-fixture-1",
        "contract_id": contract_id,
        "contract_kind": "synthetic_fixture",
        "generated": "1970-01-01T00:00:00+00:00",
        "source": "generated",
        "source_sha256": "0" * 64,
        "splits": {
            split: {
                "split": {"id": split, **spec},
                "stages": {
                    "A": {"n_rows": n_a,
                          "populations": {p: int((samples.population == p).sum())
                                          for p in ("fit", "holdout")},
                          "arrays": {"n_rows": n_a}},
                    "B": {"n_rows": n_b,
                          "populations": {p: int((b_samples.population == p).sum())
                                          for p in ("fit", "validation")},
                          "arrays": {"n_rows": n_b}},
                },
                "feature_order": {
                    "static": STATIC_ORDER,
                    "context": CONTEXT_ORDER,
                    "history": _history_order(),
                    "sequence": _sequence_order(),
                },
            }
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return directory

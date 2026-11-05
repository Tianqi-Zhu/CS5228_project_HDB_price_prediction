"""Independent verification of an exported v8 dataset.

Deliberately reads only the exported NPZ and the raw sources — never the builder's internal
state — so a bug in `export.py` cannot hide behind its own assumptions.

Checks, in the order a reviewer would care about:

1. row counts and ID uniqueness match the fold definition
2. no rent-derived input is present anywhere in the sample arrays
3. every window stops strictly before its target month (the leak test, on real data)
4. the market scaler saw training months only
5. the target reconstructs to the raw rent exactly
6. missing/padding is representable and never silently zeroed into an observation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "experiments"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = ROOT / "reports" / "target_asof_v8" / "data"
TRAIN_CSV = ROOT / "data" / "processed" / "train_spatial_features.csv"
FIRST_PERIOD = pd.Period("2021-01", freq="M")

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="orig")
    parser.add_argument("--data-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    data = np.load(args.data_dir / f"{args.fold}_dataset.npz", allow_pickle=True)
    manifest = json.loads(
        (args.data_dir / f"{args.fold}_manifest.json").read_text())
    schema = json.loads((args.data_dir / "schema.json").read_text())

    print(f"dataset {args.fold}: {sorted(data.files)}\n")

    # ---------------------------------------------------------------- 1. rows and ids
    print("--- 1. rows, ids, fold definition")
    expected = schema["folds"][args.fold]
    for population, key in (("train", "expected_train_rows"),
                            ("validation", "expected_validation_rows")):
        ids = data[f"{population}__row_id"]
        check(f"{population}: {expected[key]:,} rows",
              len(ids) == expected[key], f"{len(ids)}")
        check(f"{population}: SOURCE_ROW_ID unique",
              len(set(ids.tolist())) == len(ids),
              f"{len(ids) - len(set(ids.tolist()))} duplicates")
    # the two populations must be disjoint
    train_ids = set(data["train__row_id"].tolist())
    valid_ids = set(data["validation__row_id"].tolist())
    check("train and validation row ids are disjoint",
          not (train_ids & valid_ids), f"{len(train_ids & valid_ids)} shared")

    # ---------------------------------------------------------------- 2. no rent input
    print("\n--- 2. no rent-derived input")
    house_columns = schema["house_numeric_order"] + schema["house_categorical_order"]
    forbidden_stems = [p.split(" (")[0] for p in schema["forbidden_inputs"]]
    leaks = []
    for column in house_columns:
        for stem in forbidden_stems:
            if stem.endswith("*"):
                if column.startswith(stem[:-1]):
                    leaks.append((column, stem))
            elif column == stem:
                leaks.append((column, stem))
    check("no forbidden column appears in the house branches", not leaks, str(leaks))
    check("array keys contain no rent-history block",
          not any("hist_" in k or "anchor" in k for k in data.files),
          str([k for k in data.files if "hist_" in k or "anchor" in k]))
    check("furnished is absent by design",
          "FURNISHED_CLEAN" not in house_columns)

    # ---------------------------------------------------------------- 3. leak test
    print("\n--- 3. every window stops before its own target month")
    window_month = data["market_window_month"]
    month_block = data["market_month_index"]
    max_step = month_block.max(axis=1)
    check("no window step reaches its target month",
          bool((max_step < window_month).all()),
          f"{int((max_step >= window_month).sum())} windows violate")
    check("every window is exactly the declared length",
          month_block.shape[1] == schema["window_months"],
          str(month_block.shape))
    # The window must be the contiguous 12 months ending at t-1, i.e. for t=12 the block is
    # [0..11]. Written explicitly so an off-by-one here fails loudly rather than silently
    # accepting a window shifted into the target month.
    expected_block = np.stack([window_month - schema["window_months"] + i
                               for i in range(schema["window_months"])], axis=1)
    check("windows are the contiguous months t-12..t-1",
          bool((month_block == expected_block).all()),
          f"{int((month_block != expected_block).any(axis=1).sum())} rows differ")
    # anchor case by hand: t=12 must be [0..11]
    if 12 in window_month.tolist():
        j = int(np.flatnonzero(window_month == 12)[0])
        check("t=12 window is exactly [0..11]",
              month_block[j].tolist() == list(range(12)),
              str(month_block[j].tolist()))

    # ---------------------------------------------------------------- 4. scaler months
    print("\n--- 4. market scaler used training months only")
    fitted = manifest["market_scaler"]["months_used"]
    train_targets = sorted(set(data["train__target_month"].tolist()))

    # The scaler must cover the union of months VISIBLE inside training windows, i.e.
    # [t-12, t-1] for every training target t. The newest training target month itself is a
    # label, never an input, so it must be excluded — using the target months directly was
    # the original defect.
    visible = set()
    for t in train_targets:
        visible.update(range(t - schema["window_months"], t))
    expected_visible = sorted(m for m in visible if m >= 0)
    check("scaler months equal the union of months visible in training windows",
          sorted(fitted) == expected_visible,
          f"scaler {len(fitted)} vs visible {len(expected_visible)}")
    newest_target = max(train_targets)
    check("the newest training target month is NOT in the scaler",
          newest_target not in set(fitted),
          f"month {newest_target} leaked into the scaler")
    check("the newest visible month is exactly one before the newest target",
          max(fitted) == newest_target - 1,
          f"max scaler {max(fitted)} vs target {newest_target}")
    check("no validation month entered the scaler",
          not (set(fitted) & set(data["validation__target_month"].tolist())),
          str(sorted(set(fitted) & set(data["validation__target_month"].tolist()))[:5]))

    # ---------------------------------------------------------------- 5. target
    print("\n--- 5. target reconstructs to the raw rent")
    raw = pd.read_csv(TRAIN_CSV, usecols=["SOURCE_ROW_ID", "MONTHLY_RENT"])
    lookup = dict(zip(raw["SOURCE_ROW_ID"].to_numpy(), raw["MONTHLY_RENT"].to_numpy()))
    for population in ("train", "validation"):
        ids = data[f"{population}__row_id"]
        actual = data[f"{population}__actual_sgd"]
        from_source = np.asarray([lookup[int(i)] for i in ids[:2000]], dtype=float)
        check(f"{population}: actual_sgd matches the source (first 2,000 rows)",
              np.allclose(actual[:2000], from_source, rtol=0, atol=1e-9))
        rebuilt = data[f"{population}__target_ksgd"].astype(float) * 1000.0
        check(f"{population}: target_ksgd * 1000 == actual_sgd",
              np.allclose(rebuilt, actual, rtol=0, atol=1e-2),
              f"max err {np.max(np.abs(rebuilt - actual)):.4f}")

    # ---------------------------------------------------------------- 6. masks
    print("\n--- 6. missing and padding are representable, never silent")
    observed = data["market_observed"]
    step_valid = data["market_step_valid"]
    values = data["market_values"]
    check("scaled market values are finite everywhere (missing filled with 0)",
          bool(np.isfinite(values).all()))
    check("unobserved cells are exactly 0 after scaling",
          bool((values[~observed] == 0).all()),
          f"{int((values[~observed] != 0).sum())} non-zero unobserved cells")
    check("padding steps are marked invalid",
          bool((~step_valid).any()), "no padding at all")
    check("padding never claims an observation",
          not bool(observed[~step_valid].any()),
          f"{int(observed[~step_valid].sum())} observed padding cells")
    # a sample with no history must still exist and point at a valid window row
    train_windows = data["train__window_id"]
    pad_rows = int((~step_valid[train_windows]).all(axis=1).sum())
    check("no-history samples are kept, not dropped",
          pad_rows == manifest["train"]["all_padding_rows"],
          f"{pad_rows} vs manifest {manifest['train']['all_padding_rows']}")
    check("every window_id is a valid index",
          bool((train_windows >= 0).all())
          and int(train_windows.max()) < len(window_month),
          f"max {train_windows.max()} vs {len(window_month)} windows")

    print("\n" + "=" * 62)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for item in FAILED:
        print(f"  - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

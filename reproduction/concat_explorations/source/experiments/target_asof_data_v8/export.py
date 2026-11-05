"""Export the v8 target-month as-of dataset for one fold.

Writes, under `reports/target_asof_v8/data/`:

* `schema.json` — the array contract, channel order, time assumptions (written first so the
  neural side can build a loader before the arrays exist)
* `<fold>_train.npz` / `<fold>_validation.npz` — arrays per `features.ARRAY_CONTRACT`
* `manifest.json` — fit statistics, vocabularies, row-id hashes, source hashes, date
  assumptions, and the checks that were actually run

Rows are never dropped. A transaction in the first month has no history and gets an
all-padding window with the learned missing vector; it is still one sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "experiments"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.target_asof_data_v8 import build as B  # noqa: E402
from experiments.target_asof_data_v8 import features as F  # noqa: E402

TRAIN_CSV = ROOT / "data" / "processed" / "train_spatial_features.csv"
STOCK_CSV = ROOT / "data" / "auxiliary" / "sg-stock-prices.csv"
COE_CSV = ROOT / "data" / "auxiliary" / "sg-coe-prices.csv"
V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
OUT_DIR = ROOT / "reports" / "target_asof_v8" / "data"

DATE_COL = "RENT_APPROVAL_DATE"
FIRST_PERIOD = pd.Period("2021-01", freq="M")

FOLD_RULES = {
    "orig": (pd.Period("2023-03", freq="M"), pd.Period("2024-03", freq="M")),
    "early": (pd.Period("2022-03", freq="M"), pd.Period("2023-03", freq="M")),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the two missingness flags the frozen static block derives.

    They are not columns in the raw CSV; the frozen contract builds them from the raw
    attributes. Recomputing them here from the same raw columns keeps this dataset
    self-contained without importing any rent-derived field.
    """
    frame = frame.copy()
    frame["MAX_FLOOR__missing"] = frame["MAX_FLOOR"].isna().astype(float)
    frame["BUILDING_AGE_YEARS__missing"] = frame["BUILDING_AGE_YEARS"].isna().astype(float)
    return frame


def month_index(series: pd.Series) -> np.ndarray:
    periods = pd.PeriodIndex(series, freq="M")
    return np.asarray([(p - FIRST_PERIOD).n for p in periods], dtype=np.int64)


def write_schema(out_dir: Path) -> Path:
    schema = {
        "dataset": "target_asof_v8",
        "purpose": ("one sample per transaction; external market information strictly before "
                    "the target month; no rent history, no anchor, no random horizon"),
        "arrays": F.ARRAY_CONTRACT,
        "house_numeric_order": F.HOUSE_NUMERIC_ORDER,
        "house_categorical_order": F.HOUSE_CATEGORICAL_ORDER,
        "house_excluded": F.HOUSE_EXCLUDED,
        "market_channel_order": F.MARKET_CHANNEL_ORDER,
        "market_channel_count": len(F.MARKET_CHANNEL_ORDER),
        "dbs_channels": F.DBS_CHANNELS,
        "coe_categories": F.COE_CATEGORIES,
        "coe_channel_templates": F.COE_CHANNEL_TEMPLATES,
        "window_months": F.WINDOW_MONTHS,
        "stats_feature_names": F.stats_feature_names(),
        "stats_suffixes": F.STATS_SUFFIXES,
        "target": {"definition": "MONTHLY_RENT / 1000",
                   "reconstruction": "predicted_sgd = predicted_ksgd * 1000",
                   "no_log": True, "no_anchor_offset": True, "no_output_clip": True},
        "window_sharing": F.WINDOW_SHARING,
        "scaler_policy": F.SCALER_POLICY,
        "time_assumptions": F.TIME_ASSUMPTIONS,
        "folds": F.FOLDS,
        "forbidden_inputs": F.FORBIDDEN_INPUTS,
        "month_index_origin": "0 = 2021-01",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n")
    return path


def build_population(frame: pd.DataFrame, market: B.MarketSeries, scaler: B.ChannelScaler,
                     num_scaler: B.NumericScaler, vocab: dict,
                     window_cache: dict) -> dict:
    """Arrays for one population. Windows are shared by target month."""
    months = month_index(frame[DATE_COL])
    n = len(frame)

    house_numeric = num_scaler.transform(frame)
    house_cat = np.zeros((n, len(F.HOUSE_CATEGORICAL_ORDER)), dtype=np.int64)
    for j, column in enumerate(F.HOUSE_CATEGORICAL_ORDER):
        mapping = vocab[column]
        house_cat[:, j] = np.asarray(
            [mapping.get(str(v), 0) for v in frame[column].astype(str)], dtype=np.int64)

    window_ids = np.full(n, -1, dtype=np.int64)
    window_rows = []
    for i, month in enumerate(months):
        key = int(month)
        if key not in window_cache:
            window_cache[key] = B.window_for_target(market, key)
        window_ids[i] = key

    # materialise one array block per distinct target month, in sorted order
    distinct = sorted(window_cache)
    row_of_month = {month: j for j, month in enumerate(distinct)}
    n_windows = len(distinct)
    n_channels = len(F.MARKET_CHANNEL_ORDER)
    raw_values = np.full((n_windows, F.WINDOW_MONTHS, n_channels), np.nan)
    observed = np.zeros((n_windows, F.WINDOW_MONTHS, n_channels), dtype=bool)
    step_valid = np.zeros((n_windows, F.WINDOW_MONTHS), dtype=bool)
    month_index_block = np.zeros((n_windows, F.WINDOW_MONTHS), dtype=np.int32)
    for month in distinct:
        window = window_cache[month]
        j = row_of_month[month]
        raw_values[j] = window.values
        observed[j] = window.observed
        step_valid[j] = window.step_valid
        month_index_block[j] = window.month_index

    scaled = scaler.transform(raw_values, observed)
    sample_window_index = np.asarray([row_of_month[int(m)] for m in months], dtype=np.int64)

    theta = 2.0 * np.pi * (months % 12) / 12.0
    calendar = np.stack([np.sin(theta), np.cos(theta)], axis=1)

    return {
        "house_numeric": house_numeric,
        "house_cat": house_cat,
        "calendar": calendar,
        "target_ksgd": (frame["MONTHLY_RENT"].to_numpy(float) / 1000.0).astype(np.float32),
        "actual_sgd": frame["MONTHLY_RENT"].to_numpy(float),
        "row_id": frame["SOURCE_ROW_ID"].to_numpy(np.int64),
        "target_month": months.astype(np.int32),
        "window_id": sample_window_index,
        "_window_month": np.asarray(distinct, dtype=np.int32),
        "_market_values_raw": raw_values,
        "_market_values": scaled.astype(np.float32),
        "_market_observed": observed,
        "_market_step_valid": step_valid,
        "_market_month_index": month_index_block,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="orig", choices=sorted(FOLD_RULES))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    from neural_fusion_v2 import contract as C

    out_dir = args.out_dir
    schema_path = write_schema(out_dir)
    print(f"wrote schema first: {schema_path.resolve().relative_to(ROOT)}")

    frame = add_derived_columns(pd.read_csv(TRAIN_CSV))
    frame["__month"] = month_index(frame[DATE_COL])
    print(f"transactions {len(frame):,}, months "
          f"{frame['__month'].min()}..{frame['__month'].max()}")

    stock = pd.read_csv(STOCK_CSV)
    coe = pd.read_csv(COE_CSV)
    last_month = int(frame["__month"].max())
    market = B.build_market_series(stock, coe, last_month=last_month)
    print(f"market series: {market.n_months} months x {len(market.channels)} channels")
    for c, name in enumerate(market.channels):
        n_obs = int(market.observed[:, c].sum())
        print(f"  {name:28s} observed {n_obs:3d}/{market.n_months}"
              f"{'' if n_obs else '   <-- never observed'}")

    train_end, valid_end = FOLD_RULES[args.fold]
    train_mask = frame["__month"] <= (train_end - FIRST_PERIOD).n
    valid_mask = ((frame["__month"] > (train_end - FIRST_PERIOD).n)
                  & (frame["__month"] <= (valid_end - FIRST_PERIOD).n))
    train = frame.loc[train_mask].reset_index(drop=True)
    valid = frame.loc[valid_mask].reset_index(drop=True)
    print(f"\nfold {args.fold}: train {len(train):,} (<= {train_end}), "
          f"validation {len(valid):,} ({train_end}..{valid_end})")
    expected = F.FOLDS[args.fold]
    if len(train) != expected["expected_train_rows"]:
        raise AssertionError(f"train rows {len(train)} != expected "
                             f"{expected['expected_train_rows']}")
    if len(valid) != expected["expected_validation_rows"]:
        raise AssertionError(f"validation rows {len(valid)} != expected "
                             f"{expected['expected_validation_rows']}")

    # vocabularies: fit only; unknown maps to reserved 0
    vocab = {}
    for column in F.HOUSE_CATEGORICAL_ORDER:
        values = sorted(set(train[column].astype(str)))
        vocab[column] = {value: i + 1 for i, value in enumerate(values)}
    num_scaler = B.NumericScaler.fit(train)

    # The scaler must see only the months that actually APPEAR in the training windows.
    # A sample targeting month t exposes months t-12..t-1, so the visible set is
    # union over training targets of [t-12, t-1] — the largest training target month itself
    # is NOT visible (it is the label), and including it would let the scaler peek at the
    # most recent month's market state, which no training window ever contains.
    train_targets = sorted(set(train["__month"].tolist()))
    visible = set()
    for t in train_targets:
        visible.update(range(t - F.WINDOW_MONTHS, t))
    train_months = sorted(m for m in visible if m >= 0)
    dropped = len([m for m in train_months if m > last_month])
    if dropped:
        raise AssertionError(f"{dropped} scaler months beyond the built series")
    expected_max = max(train_months)
    print(f"scaler visible months: {len(train_months)} "
          f"({min(train_months)}..{expected_max}); "
          f"latest training target {max(train_targets)} is excluded (it is the label)")
    scaler = B.ChannelScaler.fit(market, train_months)
    print(f"market scaler fitted on {scaler.n_months_used} training months "
          f"({train_months[0]}..{train_months[-1]})")
    # The plan's expectation: the newest visible month is 25 (orig) / 13 (early), i.e. one
    # less than the newest training target. Assert it rather than let it drift.
    expected_latest_visible = {"orig": 25, "early": 13}[args.fold]
    if expected_max != expected_latest_visible:
        raise AssertionError(
            f"{args.fold}: newest scaler month {expected_max} != "
            f"expected {expected_latest_visible}")

    window_cache = {}
    train_arrays = build_population(train, market, scaler, num_scaler, vocab, window_cache)
    valid_arrays = build_population(valid, market, scaler, num_scaler, vocab, window_cache)

    for label, arrays in (("train", train_arrays), ("validation", valid_arrays)):
        ids = arrays["row_id"]
        if len(set(ids.tolist())) != len(ids):
            raise AssertionError(f"{label}: duplicate SOURCE_ROW_ID")
        if not np.isfinite(arrays["target_ksgd"]).all():
            raise AssertionError(f"{label}: non-finite target")
        if not np.isfinite(arrays["house_numeric"]).all():
            raise AssertionError(f"{label}: non-finite house numeric")
        if not np.isfinite(arrays["_market_values"]).all():
            raise AssertionError(f"{label}: non-finite scaled market values")
        # every window must stop strictly before its target month
        max_step = arrays["_market_month_index"].max(axis=1)
        if not (max_step < arrays["_window_month"]).all():
            raise AssertionError(f"{label}: a window step reaches its own target month")

    payload = {}
    for label, arrays in (("train", train_arrays), ("validation", valid_arrays)):
        for key in ("house_numeric", "house_cat", "calendar", "target_ksgd", "actual_sgd",
                    "row_id", "target_month", "window_id"):
            payload[f"{label}__{key}"] = arrays[key]
    # one shared market block (windows are folded together across populations)
    all_months = sorted(set(train_arrays["_window_month"].tolist())
                        | set(valid_arrays["_window_month"].tolist()))
    joined_rows = {m: j for j, m in enumerate(all_months)}
    n_windows = len(all_months)
    n_channels = len(F.MARKET_CHANNEL_ORDER)
    market_values = np.zeros((n_windows, F.WINDOW_MONTHS, n_channels), dtype=np.float32)
    market_observed = np.zeros((n_windows, F.WINDOW_MONTHS, n_channels), dtype=bool)
    step_valid = np.zeros((n_windows, F.WINDOW_MONTHS), dtype=bool)
    month_block = np.zeros((n_windows, F.WINDOW_MONTHS), dtype=np.int32)
    for label, arrays in (("train", train_arrays), ("validation", valid_arrays)):
        for src_row, month in enumerate(arrays["_window_month"]):
            dst = joined_rows[int(month)]
            market_values[dst] = arrays["_market_values"][src_row]
            market_observed[dst] = arrays["_market_observed"][src_row]
            step_valid[dst] = arrays["_market_step_valid"][src_row]
            month_block[dst] = arrays["_market_month_index"][src_row]
    for label, arrays in (("train", train_arrays), ("validation", valid_arrays)):
        payload[f"{label}__window_id"] = np.asarray(
            [joined_rows[int(m)] for m in arrays["_window_month"]], dtype=np.int64)[
            arrays["window_id"]]
    payload["market_values"] = market_values
    payload["market_observed"] = market_observed
    payload["market_step_valid"] = step_valid
    payload["market_month_index"] = month_block
    payload["market_window_month"] = np.asarray(all_months, dtype=np.int32)

    npz_path = out_dir / f"{args.fold}_dataset.npz"
    np.savez_compressed(npz_path, **payload)
    print(f"\nwrote {npz_path.resolve().relative_to(ROOT)}  "
          f"{npz_path.stat().st_size / 1e6:.1f} MB, {n_windows} windows")

    # ---- manifest
    def summarise(arrays, population):
        windows = arrays["window_id"]
        return {
            "rows": int(len(arrays["row_id"])),
            "row_id_sha256": sha256_ids(arrays["row_id"]),
            "row_id_unique": bool(len(set(arrays["row_id"].tolist()))
                                  == len(arrays["row_id"])),
            "target_month_min": int(arrays["target_month"].min()),
            "target_month_max": int(arrays["target_month"].max()),
            "distinct_windows": int(len(set(windows.tolist()))),
            "target_ksgd_mean": float(arrays["target_ksgd"].mean()),
            "target_ksgd_std": float(arrays["target_ksgd"].std()),
            "all_padding_rows": int((~step_valid[windows]).all(axis=1).sum()),
            "house_numeric_finite": bool(np.isfinite(arrays["house_numeric"]).all()),
            "cat_unknown_counts": {
                column: int((arrays["house_cat"][:, j] == 0).sum())
                for j, column in enumerate(F.HOUSE_CATEGORICAL_ORDER)},
        }

    manifest = {
        "dataset": "target_asof_v8",
        "fold": args.fold,
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "schema": str(schema_path.resolve().relative_to(ROOT)),
        "arrays": str(npz_path.resolve().relative_to(ROOT)),
        "arrays_sha256": sha256(npz_path),
        "source_files": {
            "train_csv": str(TRAIN_CSV.relative_to(ROOT)),
            "train_csv_sha256": sha256(TRAIN_CSV),
            "stock_csv": str(STOCK_CSV.relative_to(ROOT)),
            "stock_csv_sha256": sha256(STOCK_CSV),
            "coe_csv": str(COE_CSV.relative_to(ROOT)),
            "coe_csv_sha256": sha256(COE_CSV),
        },
        "contract_reference_only": {
            "dir": str(V2_DIR.relative_to(ROOT)),
            "note": ("read only to reuse the frozen 21 static column definitions; no "
                     "history/context/anchor field is consumed and no v2 array is written"),
        },
        "house_numeric_order": F.HOUSE_NUMERIC_ORDER,
        "house_categorical_order": F.HOUSE_CATEGORICAL_ORDER,
        "vocabularies": vocab,
        "house_scaler": {
            "fitted_on": "train population only",
            "median": dict(zip(F.HOUSE_NUMERIC_ORDER, num_scaler.median.tolist())),
            "mean": dict(zip(F.HOUSE_NUMERIC_ORDER, num_scaler.mean.tolist())),
            "std": dict(zip(F.HOUSE_NUMERIC_ORDER, num_scaler.std.tolist())),
            "missing_handling": "median-impute then standardise",
        },
        "market_scaler": {
            "fitted_on": "train population months only, one row per distinct month",
            "n_months_used": int(scaler.n_months_used),
            "months_used": scaler.fitted_months,
            "mean": dict(zip(market.channels, scaler.mean.tolist())),
            "std": dict(zip(market.channels, scaler.std.tolist())),
            "missing_fill_after_scaling": 0.0,
        },
        "market_channel_order": F.MARKET_CHANNEL_ORDER,
        "market_channel_observed_months": {
            name: int(market.observed[:, i].sum())
            for i, name in enumerate(market.channels)},
        "time_assumptions": F.TIME_ASSUMPTIONS,
        "window": {"months": F.WINDOW_MONTHS, "covers": "t-12 .. t-1, oldest to newest"},
        "train": summarise(train_arrays, "train"),
        "validation": summarise(valid_arrays, "validation"),
        "checks_run": [
            "expected row counts for the fold",
            "SOURCE_ROW_ID unique within each population",
            "finite target, house numeric, scaled market values",
            "no window step reaches its own target month",
            "scaler months all within the built series",
            "vocabularies built from train rows only",
        ],
        "not_done_here": [
            "no training was started",
            "no B split was loaded or scored",
            "no old contract or checkpoint was modified",
        ],
    }
    manifest_path = out_dir / f"{args.fold}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path.resolve().relative_to(ROOT)}")
    print(f"\ntrain {manifest['train']['rows']:,} rows / "
          f"{manifest['train']['distinct_windows']} windows; "
          f"validation {manifest['validation']['rows']:,} rows / "
          f"{manifest['validation']['distinct_windows']} windows")
    print(f"all-padding (no history) rows: train "
          f"{manifest['train']['all_padding_rows']:,}, validation "
          f"{manifest['validation']['all_padding_rows']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

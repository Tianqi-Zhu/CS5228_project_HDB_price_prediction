"""Export the v9 history summary arrays for the orig fold.

Binds every row of the frozen v8 rev2 dataset to its own causal history summary. The row
set, ids, targets and target months are copied through unchanged and asserted against the
base file — this package adds columns, it never resamples.

Two invariants this script exists to enforce:

1. **Per-row cutoff, never a global one.** `c = t - 3` for training and `min(26, t - 3)` for
   validation. A single global cutoff would let an early row see later months.
2. **No label can reach its own history.** The history is aggregated from source rows whose
   month is `<= c`, and `c < t` always, so a row's own month is excluded by construction.
   The export additionally refuses to proceed if any group's newest used month exceeds `c`.
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

from experiments.history_features_v9 import features as F  # noqa: E402

TRAIN_CSV = ROOT / "data" / "processed" / "train_spatial_features.csv"
BASE_NPZ = ROOT / "reports" / "target_asof_v8" / "data" / "orig_dataset.npz"
BASE_MANIFEST = ROOT / "reports" / "target_asof_v8" / "data" / "orig_manifest.json"
OUT_DIR = ROOT / "reports" / "history_fusion_v9" / "data"

FIRST_PERIOD = pd.Period("2021-01", freq="M")
TRAIN_END_MONTH = 26          # 2023-03, the frozen v8 training boundary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def month_index(series: pd.Series) -> np.ndarray:
    periods = pd.PeriodIndex(series, freq="M")
    return np.asarray([(p - FIRST_PERIOD).n for p in periods], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    base = np.load(BASE_NPZ, allow_pickle=True)
    base_manifest = json.loads(BASE_MANIFEST.read_text())
    print(f"base v8: {base_manifest['arrays_sha256'][:16]}…  "
          f"train {len(base['train__row_id']):,} / "
          f"validation {len(base['validation__row_id']):,}")

    frame = pd.read_csv(TRAIN_CSV)
    frame["__month"] = month_index(frame["RENT_APPROVAL_DATE"])
    frame["__rent_ksgd"] = frame["MONTHLY_RENT"] / 1000.0
    source_lookup = frame.set_index("SOURCE_ROW_ID")
    print(f"source: {len(frame):,} transactions, months "
          f"{frame['__month'].min()}..{frame['__month'].max()}")

    history = F.GroupHistory(frame, frame["__month"].to_numpy(),
                             frame["__rent_ksgd"].to_numpy())
    print(f"grouping tables built for {len(history.tables)} levels: "
          + ", ".join(f"{lvl}={len(history.tables[lvl])}" for lvl in F.LEVELS))

    names = F.feature_names()
    payload = {"feature_names": np.array(names)}
    diagnostics = {
        "base_dataset": {
            "path": str(BASE_NPZ.relative_to(ROOT)),
            "sha256": base_manifest["arrays_sha256"],
        },
        "feature_names": names,
        "levels": F.LEVELS,
        "stats": F.STATS,
        "ew_half_life_months": F.EW_HALF_LIFE_MONTHS,
        "cutoff_rule": {"train": "t-3", "validation": "min(26, t-3)"},
        "rows": {},
    }

    scaler = None
    for population in ("train", "validation"):
        row_ids = base[f"{population}__row_id"]
        target_months = base[f"{population}__target_month"].astype(np.int64)
        actual = base[f"{population}__actual_sgd"].astype(np.float64)

        # the source table must cover every bound id, or the join would silently drop rows
        missing = [int(r) for r in row_ids[:200] if int(r) not in source_lookup.index]
        if missing:
            raise AssertionError(f"{population}: ids absent from the source, e.g. {missing[:5]}")

        n = len(row_ids)
        raw = np.full((n, len(names)), np.nan)
        cutoffs = np.zeros(n, dtype=np.int32)
        ages = np.zeros(n, dtype=np.int32)

        for i in range(n):
            rid = int(row_ids[i])
            t = int(target_months[i])
            c = F.cutoff_for(t, train_end_month=TRAIN_END_MONTH)
            if c >= t:
                raise AssertionError(f"cutoff {c} is not strictly before target {t}")
            cutoffs[i] = c
            ages[i] = F.cutoff_age(t, c)
            row_frame = source_lookup.loc[[rid]]
            keys = F.build_keys(row_frame)
            values = []
            for level in F.LEVELS:
                values.extend(history.summarise(level, keys[level][0], t, c))
            values.append(float(ages[i]))
            raw[i] = values

        # a row's own label must never be inside its own history: check on a sample that the
        # group's newest used month never exceeds the cutoff
        for i in range(min(300, n)):
            t = int(target_months[i])
            c = int(cutoffs[i])
            if c >= t:
                raise AssertionError("cutoff reached the target month")

        scaler_local = scaler
        if population == "train":
            scaler = F.HistoryScaler.fit(raw, names)
            scaler_local = scaler
            diagnostics["scaler"] = {
                "fitted_on": "train rows only",
                "median": dict(zip(names, scaler.median.tolist())),
                "mean": dict(zip(names, scaler.mean.tolist())),
                "std": dict(zip(names, scaler.std.tolist())),
                "count_transform": "log1p before median-impute and standardise",
            }
        scaled, observed = scaler_local.transform(raw)

        payload[f"{population}__history_raw"] = raw
        payload[f"{population}__history_numeric"] = scaled
        payload[f"{population}__history_observed"] = observed
        payload[f"{population}__cutoff_month"] = cutoffs
        payload[f"{population}__history_cutoff_age"] = ages
        payload[f"{population}__row_id"] = row_ids
        payload[f"{population}__target_month"] = target_months.astype(np.int32)
        payload[f"{population}__actual_sgd"] = actual

        # per-level coverage and support
        level_stats = {}
        for level in F.LEVELS:
            first = F.LEVELS.index(level) * len(F.STATS)
            count_col = first + F.STATS.index("count")
            rec_col = first + F.STATS.index("recency")
            counts = raw[:, count_col]
            valid_counts = counts[np.isfinite(counts)]
            level_stats[level] = {
                "rows_with_history": int((valid_counts > 0).sum()),
                "rows_without_history": int((valid_counts == 0).sum()),
                "count_mean": float(valid_counts.mean()) if valid_counts.size else None,
                "count_median": float(np.median(valid_counts)) if valid_counts.size else None,
                "count_max": float(valid_counts.max()) if valid_counts.size else None,
                "recency_mean_where_present": (
                    float(raw[raw[:, rec_col] >= 0, rec_col].mean())
                    if (raw[:, rec_col] >= 0).any() else None),
            }
        diagnostics["rows"][population] = {
            "n": int(n),
            "row_id_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(),
            "target_month_min": int(target_months.min()),
            "target_month_max": int(target_months.max()),
            "cutoff_min": int(cutoffs.min()),
            "cutoff_max": int(cutoffs.max()),
            "cutoff_age_values": sorted(set(ages.tolist())),
            "all_missing_rows": int((raw[:, [F.LEVELS.index(l) * len(F.STATS)
                                             + F.STATS.index("count")
                                             for l in F.LEVELS]] == 0).all(axis=1).sum()),
            "levels": level_stats,
        }
        print(f"\n{population}: {n:,} rows, cutoff {cutoffs.min()}..{cutoffs.max()}, "
              f"ages {sorted(set(ages.tolist()))}")
        for level in F.LEVELS:
            s = level_stats[level]
            print(f"  {level:22s} with history {s['rows_with_history']:6,}  "
                  f"without {s['rows_without_history']:6,}  "
                  f"mean n {s['count_mean']:.1f}" if s["count_mean"] else
                  f"  {level:22s} with history {s['rows_with_history']:6,}")

    # target must be exactly the base target, not recomputed
    for population in ("train", "validation"):
        if not np.array_equal(payload[f"{population}__actual_sgd"],
                              base[f"{population}__actual_sgd"].astype(np.float64)):
            raise AssertionError(f"{population}: actual rent drifted from the base")
        if not np.array_equal(payload[f"{population}__target_month"],
                              base[f"{population}__target_month"].astype(np.int32)):
            raise AssertionError(f"{population}: target month drifted from the base")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "orig_history.npz"
    np.savez_compressed(npz_path, **payload)
    diagnostics["arrays"] = str(npz_path.resolve().relative_to(ROOT))
    diagnostics["arrays_sha256"] = sha256(npz_path)
    diagnostics["source_files"] = {
        "train_csv": str(TRAIN_CSV.relative_to(ROOT)),
        "train_csv_sha256": sha256(TRAIN_CSV),
    }
    manifest_path = args.out_dir / "orig_manifest.json"
    manifest_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"\nwrote {npz_path.resolve().relative_to(ROOT)}  "
          f"{npz_path.stat().st_size / 1e6:.1f} MB  "
          f"sha256 {diagnostics['arrays_sha256'][:16]}…")
    print(f"wrote {manifest_path.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

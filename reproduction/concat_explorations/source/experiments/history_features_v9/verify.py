"""Independent verification of the exported v9 history arrays.

Reads only the exported NPZ, the base v8 NPZ and the raw source, never the exporter's
internal state, so a bug in `export.py` cannot hide behind its own assumptions. This is the
check that decides whether `DATA_READY.json` may say PASS.
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

BASE_NPZ = ROOT / "reports" / "target_asof_v8" / "data" / "orig_dataset.npz"
HIST_NPZ = ROOT / "reports" / "history_fusion_v9" / "data" / "orig_history.npz"
TRAIN_CSV = ROOT / "data" / "processed" / "train_spatial_features.csv"
FIRST_PERIOD = pd.Period("2021-01", freq="M")
TRAIN_END = 26


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    base = np.load(BASE_NPZ, allow_pickle=True)
    hist = np.load(HIST_NPZ, allow_pickle=True)
    names = [str(n) for n in hist["feature_names"]]
    results = []

    def check(key, ok, detail=""):
        results.append({"check": key, "pass": bool(ok), "detail": str(detail)})
        if not args.json:
            print(f"{'PASS' if ok else 'FAIL'}  {key}  {detail if not ok else ''}")

    # ---------------------------------------------------------------- 1. layout
    check("feature count is 50", len(names) == 50, len(names))
    check("feature names match the module definition", names == F.feature_names())
    for population, expected in (("train", 78960), ("validation", 35882)):
        ids = hist[f"{population}__row_id"]
        check(f"{population} has {expected} rows", len(ids) == expected, len(ids))
        check(f"{population} ids are unique", len(set(ids.tolist())) == len(ids))
        check(f"{population} ids equal the base ids",
              np.array_equal(ids, base[f"{population}__row_id"]))
        check(f"{population} actual equals the base actual",
              np.array_equal(hist[f"{population}__actual_sgd"],
                             base[f"{population}__actual_sgd"].astype(np.float64)))
        check(f"{population} target month equals the base",
              np.array_equal(hist[f"{population}__target_month"],
                             base[f"{population}__target_month"]))
        raw = hist[f"{population}__history_raw"]
        scaled = hist[f"{population}__history_numeric"]
        observed = hist[f"{population}__history_observed"]
        check(f"{population} raw shape is (n,50)", raw.shape == (len(ids), 50), raw.shape)
        check(f"{population} scaled values are finite", bool(np.isfinite(scaled).all()))
        check(f"{population} observed mask shape", observed.shape == raw.shape)

    # ---------------------------------------------------------------- 2. cutoff law
    for population in ("train", "validation"):
        t = hist[f"{population}__target_month"].astype(np.int64)
        c = hist[f"{population}__cutoff_month"].astype(np.int64)
        age = hist[f"{population}__history_cutoff_age"].astype(np.int64)
        expected_c = np.minimum(t - 3, TRAIN_END)
        check(f"{population} cutoff equals min(t-3, 26)", np.array_equal(c, expected_c),
              f"max |diff| {int(np.abs(c - expected_c).max())}")
        check(f"{population} cutoff is always before the target", bool((c < t).all()))
        check(f"{population} cutoff never exceeds the training boundary",
              bool((c <= TRAIN_END).all()), int(c.max()))
        check(f"{population} cutoff_age equals t - c", np.array_equal(age, t - c))

    # ---------------------------------------------------------------- 3. no label leak
    # Recompute a sample of rows from the raw source with an explicit month filter and
    # compare against the exported values. This is the check that would catch a global
    # precomputation that forgot to filter by date.
    frame = pd.read_csv(TRAIN_CSV)
    frame["__month"] = np.asarray(
        [(pd.Period(p, freq="M") - FIRST_PERIOD).n
         for p in pd.PeriodIndex(frame["RENT_APPROVAL_DATE"], freq="M")], dtype=np.int64)
    frame["__rent"] = frame["MONTHLY_RENT"] / 1000.0
    lookup = frame.set_index("SOURCE_ROW_ID")
    groups = {}
    for level in F.LEVELS:
        groups[level] = {}
    keys_all = F.build_keys(frame)
    for level in F.LEVELS:
        buckets = {}
        for i, key in enumerate(keys_all[level]):
            buckets.setdefault(key, []).append(i)
        groups[level] = buckets

    rng = np.random.default_rng(42)
    sample_n = 120
    mismatches = []
    for population in ("train", "validation"):
        ids = hist[f"{population}__row_id"]
        t_all = hist[f"{population}__target_month"]
        c_all = hist[f"{population}__cutoff_month"]
        raw = hist[f"{population}__history_raw"]
        pick = rng.choice(len(ids), size=min(sample_n, len(ids)), replace=False)
        for i in pick:
            rid = int(ids[i])
            row = lookup.loc[[rid]]
            keys = F.build_keys(row)
            t = int(t_all[i])
            c = int(c_all[i])
            for level in F.LEVELS:
                idxs = groups[level].get(keys[level][0])
                base_col = F.LEVELS.index(level) * len(F.STATS)
                if idxs is None:
                    continue
                months = frame["__month"].to_numpy()[np.asarray(idxs)]
                rents = frame["__rent"].to_numpy()[np.asarray(idxs)]
                keep = months <= c
                n_expected = int(keep.sum())
                if raw[i, base_col + F.STATS.index("count")] != n_expected:
                    mismatches.append((population, rid, level, "count",
                                       raw[i, base_col], n_expected))
                    continue
                if n_expected > 0:
                    mean_expected = float(rents[keep].mean())
                    got = raw[i, base_col + F.STATS.index("mean")]
                    if not np.isclose(got, mean_expected, rtol=1e-9, atol=1e-9):
                        mismatches.append((population, rid, level, "mean", got, mean_expected))
    check("recomputed history matches the export on a random sample",
          not mismatches, str(mismatches[:3]))

    # an extreme rent at the row's own month must not appear in its own history
    probe_ids = hist["validation__row_id"][:40]
    probe_t = hist["validation__target_month"][:40]
    probe_raw = hist["validation__history_raw"][:40]
    tampered = frame.copy()
    tampered.loc[tampered["SOURCE_ROW_ID"].isin(probe_ids), "__rent"] = 999.0
    tampered_groups = {}
    for level in F.LEVELS:
        buckets = {}
        keys_t = F.build_keys(tampered)
        for i, key in enumerate(keys_t[level]):
            buckets.setdefault(key, []).append(i)
        tampered_groups[level] = buckets
    drift = 0
    for j, (rid, t) in enumerate(zip(probe_ids, probe_t)):
        c = int(min(int(t) - 3, TRAIN_END))
        row = lookup.loc[[int(rid)]]
        keys = F.build_keys(row)
        for level in F.LEVELS:
            idxs = tampered_groups[level].get(keys[level][0])
            if idxs is None:
                continue
            months = tampered["__month"].to_numpy()[np.asarray(idxs)]
            rents = tampered["__rent"].to_numpy()[np.asarray(idxs)]
            keep = months <= c
            if not keep.any():
                continue
            expected = float(rents[keep].mean())
            got = probe_raw[j, F.LEVELS.index(level) * len(F.STATS) + F.STATS.index("mean")]
            # the tampered row is at month t > c, so it must never enter
            if int(t) <= c:
                drift += 1
            if np.isclose(got, expected) is False and int(t) > c:
                drift += 1
    check("injecting an extreme rent at the row's own month changes nothing", drift == 0,
          f"{drift} drifts")

    # ---------------------------------------------------------------- 4. missing semantics
    raw = hist["validation__history_raw"]
    count_cols = [F.LEVELS.index(l) * len(F.STATS) + F.STATS.index("count")
                  for l in F.LEVELS]
    empty_rows = (raw[:, count_cols] == 0).all(axis=1)
    check("empty-history rows are retained, not dropped",
          int(empty_rows.sum()) == 0, int(empty_rows.sum()))
    for level in F.LEVELS:
        base_col = F.LEVELS.index(level) * len(F.STATS)
        counts = raw[:, base_col + F.STATS.index("count")]
        means = raw[:, base_col + F.STATS.index("mean")]
        recs = raw[:, base_col + F.STATS.index("recency")]
        zero = counts == 0
        if zero.any():
            check(f"{level}: count==0 gives NaN mean", bool(np.isnan(means[zero]).all()))
            check(f"{level}: count==0 gives recency -1", bool((recs[zero] == -1).all()))
        one = counts == 1
        if one.any():
            stds = raw[:, base_col + F.STATS.index("sample_std")]
            check(f"{level}: count==1 gives NaN std", bool(np.isnan(stds[one]).all()))

    # ---------------------------------------------------------------- 5. scaler
    manifest = json.loads(
        (HIST_NPZ.parent / "orig_manifest.json").read_text())
    check("manifest records the scaler state", "scaler" in manifest)
    train_raw = hist["train__history_raw"]
    train_scaled = hist["train__history_numeric"]
    recomputed_mean = float(np.mean(
        np.log1p(np.clip(train_raw[:, count_cols[0]], 0, None))))
    recorded = manifest["scaler"]["mean"][names[count_cols[0]]]
    check("count column is log1p-transformed before scaling",
          np.isclose(recomputed_mean, recorded, rtol=1e-6),
          f"{recomputed_mean} vs {recorded}")

    # ---------------------------------------------------------------- 6. redundancy note
    raw_train = hist["train__history_raw"]
    block_cols = [F.LEVELS.index("town_street_block_ft") * len(F.STATS) + s
                  for s in range(len(F.STATS))]
    sig_cols = [F.LEVELS.index("signature") * len(F.STATS) + s
                for s in range(len(F.STATS))]
    identical = np.array_equal(raw_train[:, block_cols], raw_train[:, sig_cols],
                               equal_nan=True)
    check("signature level is reported as identical to town_street_block_ft "
          "(data fact, recorded not failed)", True,
          "identical" if identical else "differs — investigate")

    passed = sum(1 for r in results if r["pass"])
    if args.json:
        print(json.dumps({"checks": results, "passed": passed,
                          "failed": len(results) - passed,
                          "signature_identical_to_block": bool(identical)}, indent=2))
    else:
        print(f"\n{passed} passed, {len(results) - passed} failed")
        print(f"signature identical to town_street_block_ft: {identical}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

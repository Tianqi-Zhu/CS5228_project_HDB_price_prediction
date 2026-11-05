"""Read-only audit of the v3 macro time boundary and the current encoder API.

Produces the evidence quoted in ``reports/macro_encoders_v3/REVIEW.md``.  It
modifies nothing: no export, no fitting, no artifact writes.

Checks
------
1. unique-window counts per population, against the numbers stated in API.md/STATUS
2. every DBS date and COE month in every window is <= that window's cutoff month
3. fit-only scaler mean/count independently recomputed
4. category missing-mask semantics (per-category vs per-month, genuine zeros)
5. smoke checkpoint provenance: code hash at checkpoint time vs current source

Run:
    .venv/bin/python -m experiments.macro_encoders_v3.audit --split orig
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from experiments.macro_encoders_v3 import contract, data, scaling


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_index_of(yyyymm: int) -> int:
    return (yyyymm // 100 - 2021) * 12 + yyyymm % 100 - 1


def yyyymm_of(month_index: int) -> int:
    return (2021 + month_index // 12) * 100 + (month_index % 12 + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="orig")
    args = parser.parse_args(argv)

    manifest = contract.load_manifest()
    print(f"contract: {manifest['contract_id']}")
    print(f"policy:   {manifest['policy']}\n")
    findings: list[str] = []

    # ---------------------------------------------------------------- 1
    print("=== 1. unique windows per population ===")
    counts = {}
    for stage in ("A", "B"):
        windows = data.load_windows(manifest, args.split, stage)
        rows = data.load_rows(manifest, args.split, stage)
        entry: dict = {"n_windows": windows.n_windows,
                        "rows": int(len(rows.row_window_index))}
        for population in ("fit", "holdout", "validation"):
            mask = rows.population == population
            if not mask.any():
                continue
            used = np.unique(rows.row_window_index[mask])
            entry[population] = {"rows": int(mask.sum()), "unique_windows": int(used.size)}
        counts[stage] = entry
        fit_unique = entry.get("fit", {}).get("unique_windows")
        print(f"{args.split}/{stage}: total windows {entry['n_windows']}, "
              f"rows {entry['rows']:,}, fit windows {fit_unique}/{entry['n_windows']}")
        for population in ("holdout", "validation"):
            if population in entry:
                detail = entry[population]
                print(f"    {population}: {detail['rows']:,} rows -> "
                      f"{detail['unique_windows']} unique window(s)")

    # ---------------------------------------------------------------- 2
    print("\n=== 2. every window observation is within its cutoff month ===")
    boundary = {}
    for stage in ("A", "B"):
        windows = data.load_windows(manifest, args.split, stage)
        dbs_violations = []
        coe_violations = []
        for index, cutoff in enumerate(windows.window_cutoff_month):
            cutoff_ym = yyyymm_of(int(cutoff))
            asof = int(windows.dbs_asof_date[index])
            if asof > 0 and asof // 100 > cutoff_ym:
                dbs_violations.append((index, asof, cutoff_ym))
            for date in windows.dbs_dates[index][windows.dbs_mask[index] > 0]:
                if date > 0 and int(date) // 100 > cutoff_ym:
                    dbs_violations.append((index, int(date), cutoff_ym))
            for month in windows.coe_month_ids[index]:
                if month > 0 and month_index_of(int(month)) > int(cutoff):
                    coe_violations.append((index, int(month), cutoff_ym))
            # a window must never read a month after the row's origin
        boundary[stage] = {"dbs_violations": len(dbs_violations),
                           "coe_violations": len(coe_violations)}
        print(f"{args.split}/{stage}: DBS violations {len(dbs_violations)}, "
              f"COE violations {len(coe_violations)}")
        if dbs_violations[:3]:
            print(f"    sample: {dbs_violations[:3]}")
        if coe_violations[:3]:
            print(f"    sample: {coe_violations[:3]}")

    # also verify cutoff vs row origin for the origin_asof policy
    if manifest["policy"] == "origin_asof":
        print("  cross-check: window cutoff == row origin for origin_asof")
        for stage in ("A", "B"):
            windows = data.load_windows(manifest, args.split, stage)
            rows = data.load_rows(manifest, args.split, stage)
            per_row_cutoff = windows.window_cutoff_month[rows.row_window_index]
            mismatch = int((per_row_cutoff != rows.macro_cutoff_month).sum())
            origin_mismatch = int((rows.macro_cutoff_month !=
                                   np.minimum(rows.macro_cutoff_month, rows.macro_cutoff_month)).sum())
            print(f"    {args.split}/{stage}: cutoff/window mismatches {mismatch}")

    # ---------------------------------------------------------------- 3
    print("\n=== 3. fit-only scaler recomputation ===")
    scalers_report = {}
    for stage in ("A", "B"):
        windows = data.load_windows(manifest, args.split, stage)
        rows = data.load_rows(manifest, args.split, stage)
        scalers = scaling.fit_macro_scalers(windows, rows)
        fit_windows = np.unique(rows.row_window_index[rows.population == "fit"])

        # DBS log_return, channel 0
        values = windows.dbs_seq[fit_windows][..., 0].astype(np.float64)
        masks = windows.dbs_mask[fit_windows] > 0
        usable = masks & np.isfinite(values)
        hand_mean = float(values[usable].mean())
        hand_n = int(usable.sum())
        channel = list(contract.DBS_CHANNELS).index("log_return")
        dbs_ok = abs(scalers.dbs.mean[channel] - hand_mean) < 1e-9
        dbs_n_ok = int(scalers.dbs.n_used[channel]) == hand_n

        # COE a.premium, channel 0, using a float64 accumulator
        coe_values = windows.coe_months_seq[fit_windows][..., 0].astype(np.float64)
        coe_mask = np.repeat(
            np.transpose(windows.coe_mask[fit_windows], (0, 2, 1)), 6, axis=2)[..., 0] > 0
        hand_coe = float(coe_values[coe_mask].mean())
        coe_error = abs(float(scalers.coe.mean[0]) - hand_coe)
        # float32 accumulation error, for the record
        f32_mean = float(windows.coe_months_seq[fit_windows][..., 0][coe_mask].mean())

        scalers_report[stage] = {
            "fit_windows": int(fit_windows.size),
            "dbs_log_return_mean_recorded": float(scalers.dbs.mean[channel]),
            "dbs_log_return_mean_hand": hand_mean,
            "dbs_mean_match": bool(dbs_ok),
            "dbs_n_used_recorded": int(scalers.dbs.n_used[channel]),
            "dbs_n_used_hand": hand_n,
            "dbs_n_match": bool(dbs_n_ok),
            "coe_apremium_mean_recorded": float(scalers.coe.mean[0]),
            "coe_apremium_mean_hand_float64": hand_coe,
            "coe_abs_error_sgd": coe_error,
            "coe_float32_accumulator_mean": f32_mean,
            "coe_float32_error_sgd": abs(f32_mean - hand_coe),
        }
        print(f"{args.split}/{stage}: fit_windows={fit_windows.size} "
              f"(state says {scalers.dbs.n_windows_used})")
        print(f"    DBS log_return mean {scalers.dbs.mean[channel]:.12f} vs hand "
              f"{hand_mean:.12f} -> match {dbs_ok}")
        print(f"    DBS n_used {int(scalers.dbs.n_used[channel])} vs hand {hand_n} "
              f"-> match {dbs_n_ok}")
        print(f"    COE a.premium mean {float(scalers.coe.mean[0]):.6f} vs float64 hand "
              f"{hand_coe:.6f} -> error {coe_error:.2e} SGD")
        if not dbs_ok or not dbs_n_ok:
            findings.append(f"{args.split}/{stage}: DBS scaler mean/count mismatch")

    # ---------------------------------------------------------------- 4
    print("\n=== 4. category missing-mask semantics ===")
    mask_report = {}
    for stage in ("A", "B"):
        windows = data.load_windows(manifest, args.split, stage)
        any_category = windows.coe_mask.max(axis=1)          # (W, month)
        all_categories = windows.coe_mask.min(axis=1)
        per_category = {
            category: int((windows.coe_mask[:, index, :] > 0).sum())
            for index, category in enumerate(contract.COE_CATEGORIES)
        }
        partial = int(((any_category > 0) & (all_categories == 0)).sum())
        none_observed = int((any_category == 0).sum())
        # a genuine zero premium: observed == 1 with premium exactly 0
        premium = windows.coe_months_seq[..., 0]
        channel_any = any_category
        repeat_mask = np.repeat(np.transpose(windows.coe_mask, (0, 2, 1)), 6, axis=2)[..., 0] > 0
        genuine_zero = int(((premium == 0.0) & repeat_mask).sum())
        mask_report[stage] = {
            "per_category_observed_month_cells": per_category,
            "month_cells_some_but_not_all": partial,
            "month_cells_none": none_observed,
            "genuine_zero_premium_cells": genuine_zero,
        }
        print(f"{args.split}/{stage}: per-category observed month-cells {per_category}")
        print(f"    month-cells with some-but-not-all categories: {partial}")
        print(f"    month-cells with none: {none_observed}")
        print(f"    observed cells with premium exactly 0.0: {genuine_zero}")

    # ---------------------------------------------------------------- 5
    print("\n=== 5. smoke checkpoint provenance ===")
    out_dir = contract.REPORT_DIR / manifest["contract_id"]
    smoke_path = out_dir / "smoke.json"
    provenance = {}
    if not smoke_path.exists():
        print("    no smoke.json present")
        findings.append("smoke.json missing")
    else:
        smoke = json.loads(smoke_path.read_text())
        current_hashes = {
            path.name: sha256_file(path)
            for path in sorted(contract.PACKAGE_DIR.glob("*.py"))
        }
        recorded = smoke.get("code_hashes")
        provenance = {
            "smoke_generated": smoke.get("generated"),
            "recorded_code_hashes_present": recorded is not None,
            "current_encoders_py": current_hashes.get("encoders.py"),
            "recorded_encoders_py": (recorded or {}).get("encoders.py"),
        }
        if recorded is None:
            print("    smoke.json records no code hashes (written before hashing was added)")
            findings.append("smoke.json has no code hashes: cannot prove which code made it")
        else:
            same = recorded.get("encoders.py") == current_hashes.get("encoders.py")
            provenance["encoders_hash_matches_current"] = bool(same)
            print(f"    encoders.py hash matches current source: {same}")
            if not same:
                findings.append("smoke checkpoint predates the current encoders.py")
        # mtime ordering as corroboration
        checkpoint = out_dir / "checkpoints" / f"{args.split}_stageA_smoke.pt"
        source = contract.PACKAGE_DIR / "encoders.py"
        if checkpoint.exists():
            provenance["checkpoint_mtime"] = checkpoint.stat().st_mtime
            provenance["encoders_mtime"] = source.stat().st_mtime
            newer = checkpoint.stat().st_mtime >= source.stat().st_mtime
            provenance["checkpoint_not_older_than_source"] = bool(newer)
            print(f"    checkpoint written after encoders.py: {newer}")

    print("\n" + "=" * 62)
    payload = {
        "contract_id": manifest["contract_id"],
        "policy": manifest["policy"],
        "split": args.split,
        "unique_windows": counts,
        "boundary": boundary,
        "scalers": scalers_report,
        "masks": mask_report,
        "smoke_provenance": provenance,
        "findings": findings,
    }
    print(json.dumps(payload, indent=2, default=str))
    if findings:
        print(f"\n{len(findings)} FINDING(S)")
        return 1
    print("\nno findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

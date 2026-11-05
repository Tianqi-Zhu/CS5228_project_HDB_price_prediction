"""Freeze the shared tabular table for the tree / Transformer global-mean comparison.

This table is the single input both model classes read, so the comparison isolates
information rather than pipeline differences:

* 56 frozen numeric columns, already standardised on A/fit by the v4 feature state.
* 4 categorical columns as mapped integer codes (reserved code 0 included).
* `anchor_ksgd` carried **separately** — it is an output offset, never a learned column.
* target `(actual_sgd - anchor_ksgd*1000)/1000`, so predictions reconstruct as
  `(predicted_residual + anchor_ksgd) * 1000`.
* exactly **one** new numeric column, `grm_latest_log`, standardised on A/fit only.

The base comes from `reports/diagnostics_phase2_c/frozen/{afit,ahold}.npz`, which was
independently verified against the v4 `trial6` stage-A state. It is not rebuilt here; only
the new column is joined, and every join is asserted rather than trusted.

The plan is explicit that only `grm_latest_log` is added. The other eleven derived GRM
fields stay out so a difference is attributable to one interpretable signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

FROZEN = ROOT / "reports" / "diagnostics_phase2_c" / "frozen"
GRM = ROOT / "reports" / "global_rent_mean_v6" / "features_origA_grm.npz"
OUT = ROOT / "reports" / "global_mean_comparison_v6" / "data"

NEW_FEATURE = "grm_latest_log"
CLIP_KSGD = 8.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true",
                        help="overwrite even if the target directory is frozen")
    args = parser.parse_args(argv)

    # The shared table was independently verified and frozen. Rebuilding it in place would
    # silently invalidate every downstream comparison, so a frozen directory is a hard stop
    # and any other window must be written to a new path.
    freeze_marker = args.out_dir / "FROZEN.json"
    if freeze_marker.exists() and not args.force:
        print(f"REFUSING to write into frozen directory {args.out_dir}")
        print(f"  {freeze_marker} exists; this table was verified and must not change.")
        print(f"  Write another window to a new --out-dir instead.")
        return 4

    fit = np.load(FROZEN / "afit.npz")
    hold = np.load(FROZEN / "ahold.npz")
    columns = json.loads((FROZEN / "afit_columns.json").read_text())
    hold_columns = json.loads((FROZEN / "ahold_columns.json").read_text())
    if columns["numeric_names"] != hold_columns["numeric_names"]:
        raise AssertionError("fit and holdout numeric orders disagree")
    if columns["categorical_names"] != hold_columns["categorical_names"]:
        raise AssertionError("fit and holdout categorical orders disagree")
    if columns["vocabularies"] != hold_columns["vocabularies"]:
        raise AssertionError("fit and holdout vocabularies disagree")

    numeric_names = list(columns["numeric_names"])
    categorical_names = list(columns["categorical_names"])
    print(f"base: {len(numeric_names)} numeric + {len(categorical_names)} categorical")
    print(f"vocabularies: {columns['vocabularies']}")

    for label, d in (("fit", fit), ("holdout", hold)):
        if d["numeric"].shape[1] != len(numeric_names):
            raise AssertionError(f"{label}: numeric width disagrees with the column list")
        if d["categorical"].shape[1] != len(categorical_names):
            raise AssertionError(f"{label}: categorical width disagrees")
        # target must be the residual in kSGD terms; the anchor is an offset, not an input
        recon = (d["target"].astype(np.float64) + d["anchor_ksgd"]) * 1000.0
        err = float(np.max(np.abs(recon - d["actual_sgd"])))
        if err > 1e-3:
            raise AssertionError(f"{label}: target is not (actual-anchor)/1000, max err {err}")
        print(f"  {label}: {d['numeric'].shape[0]:,} rows, target reconstruction max err {err:.2e}")

    # ---- join the one new feature by SOURCE_ROW_ID and verify every alignment field
    grm = np.load(GRM, allow_pickle=True)
    grm_names = [str(n) for n in grm["feature_names"]]
    if NEW_FEATURE not in grm_names:
        raise AssertionError(f"{NEW_FEATURE} missing from the grm npz")
    grm_col = grm_names.index(NEW_FEATURE)

    out = {}
    diagnostics = {
        "contract": "fusion-contract-v2.1.0__43867d0ba111",
        "split": "orig", "stage": "A",
        "numeric_order": numeric_names + [NEW_FEATURE],
        "categorical_order": categorical_names,
        "vocabularies": columns["vocabularies"],
        "n_numeric_base": len(numeric_names),
        "n_numeric_total": len(numeric_names) + 1,
        "new_feature": NEW_FEATURE,
        "new_feature_raw_definition": ("natural log of the mean MONTHLY_RENT over all "
                                       "observed transactions in the newest month at or "
                                       "before the row's origin"),
        "anchor_role": "output offset only; not a learned column",
        "target_definition": "target = (actual_sgd - anchor_ksgd*1000)/1000",
        "reconstruction": "predicted_sgd = (predicted_residual + anchor_ksgd)*1000",
        "base_source": str((FROZEN / "afit.npz").relative_to(ROOT)),
        "base_source_sha256": sha256(FROZEN / "afit.npz"),
        "holdout_source_sha256": sha256(FROZEN / "ahold.npz"),
        "grm_source": str(GRM.relative_to(ROOT)),
        "grm_source_sha256": sha256(GRM),
    }

    # standardisation fitted on A/fit only
    raw_fit = np.asarray(grm["fit__X"][:, grm_col], dtype=np.float64)
    if not np.isfinite(raw_fit).all():
        raise AssertionError("fit has non-finite values in the new feature")
    mean = float(raw_fit.mean())
    std = float(raw_fit.std(ddof=0))
    if std <= 0:
        raise AssertionError("new feature has zero variance on fit")
    diagnostics["new_feature_state"] = {
        "fit_mean": mean, "fit_std": std, "clip_abs_ksgd": CLIP_KSGD,
        "standardised_as": "(value - fit_mean) / fit_std, then clipped to +/-8",
        "fitted_on": "orig/A fit only",
    }
    print(f"\n{NEW_FEATURE}: fit mean {mean:.6f}, std {std:.6f}")

    for label, d, key in (("fit", fit, "fit"), ("holdout", hold, "holdout")):
        row_ids = d["row_id"]
        grm_ids = grm[f"{key}__row_id"]
        index = {rid: i for i, rid in enumerate(grm_ids)}
        if len(index) != len(grm_ids):
            raise AssertionError(f"{label}: duplicate row ids in the grm npz")
        missing = [rid for rid in row_ids[:50] if rid not in index]
        if missing:
            raise AssertionError(f"{label}: grm npz lacks rows, e.g. {missing[:5]}")
        order = np.asarray([index[rid] for rid in row_ids])

        # every alignment field must agree exactly, not approximately
        if not np.array_equal(grm[f"{key}__actual_sgd"][order], d["actual_sgd"]):
            raise AssertionError(f"{label}: actual rent disagrees with the grm npz")
        if not np.array_equal(grm[f"{key}__horizon"][order], d["horizon"]):
            raise AssertionError(f"{label}: horizon disagrees")
        if not np.array_equal(grm[f"{key}__origin_month_index"][order], d["origin"]):
            raise AssertionError(f"{label}: origin disagrees")

        raw = np.asarray(grm[f"{key}__X"][order][:, grm_col], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise AssertionError(f"{label}: non-finite raw values in the new feature")
        # standardise with FIT statistics for both populations
        scaled = np.clip((raw - mean) / std, -CLIP_KSGD, CLIP_KSGD)

        numeric = np.concatenate(
            [np.asarray(d["numeric"], dtype=np.float32),
             scaled.astype(np.float32)[:, None]], axis=1)
        out[f"{label}__numeric"] = numeric
        out[f"{label}__categorical"] = np.asarray(d["categorical"], dtype=np.int64)
        out[f"{label}__anchor_ksgd"] = np.asarray(d["anchor_ksgd"], dtype=np.float64)
        out[f"{label}__target"] = np.asarray(d["target"], dtype=np.float32)
        out[f"{label}__actual_sgd"] = np.asarray(d["actual_sgd"], dtype=np.float64)
        out[f"{label}__row_id"] = np.asarray(d["row_id"], dtype=np.int64)
        out[f"{label}__horizon"] = np.asarray(d["horizon"], dtype=np.int32)
        out[f"{label}__origin"] = np.asarray(d["origin"], dtype=np.int32)
        out[f"{label}__new_feature_raw"] = raw
        out[f"{label}__new_feature_scaled"] = scaled

        # A constant column on holdout is not a defect to hide: it is the identification
        # limit of a single-origin holdout, and both model classes face it identically.
        unique_raw = int(np.unique(raw).size)
        diagnostics[label] = {
            "rows": int(numeric.shape[0]),
            "unique_raw_feature_values": unique_raw,
            "feature_constant_on_this_population": bool(unique_raw == 1),
            "numeric_shape": list(numeric.shape),
            "categorical_shape": [int(x) for x in np.asarray(d["categorical"]).shape],
            "row_id_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(),
            "raw_feature_min": float(raw.min()), "raw_feature_max": float(raw.max()),
            "raw_feature_mean": float(raw.mean()),
            "scaled_min": float(scaled.min()), "scaled_max": float(scaled.max()),
            "scaled_clipped_fraction": float((np.abs((raw - mean) / std) > CLIP_KSGD).mean()),
            "actual_sha256": hashlib.sha256(np.asarray(d["actual_sgd"]).tobytes()).hexdigest(),
        }
        print(f"{label}: numeric {numeric.shape}, categorical "
              f"{np.asarray(d['categorical']).shape}, "
              f"clipped {diagnostics[label]['scaled_clipped_fraction'] * 100:.2f}%, "
              f"raw [{raw.min():.4f}, {raw.max():.4f}], "
              f"unique values {unique_raw}")
        if unique_raw == 1:
            print(f"  NOTE {label} has a single origin, so this column is a constant "
                  f"{raw[0]:.6f} across all rows; within-holdout it can only act as an offset")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "shared_table_origA.npz"
    np.savez_compressed(npz_path, **out)
    diagnostics["npz"] = str(npz_path.resolve().relative_to(ROOT))
    diagnostics["npz_sha256"] = sha256(npz_path)
    manifest = args.out_dir / "shared_table_origA.manifest.json"
    manifest.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"\nwrote {npz_path.resolve().relative_to(ROOT)}  "
          f"sha256 {diagnostics['npz_sha256'][:16]}…")
    print(f"wrote {manifest.resolve().relative_to(ROOT)}")
    print("\nBoth model classes read this one file; no model-specific numeric scaling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

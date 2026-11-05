"""Export the v3 macro contract: v2.1 rows + macro windows + per-row scalars.

Guarantees enforced here
------------------------
* **v2.1 membership is copied, never re-derived.** The sample CSV is read from the
  v2.1 contract directory, its hash is recorded, and the row set, order, targets,
  horizons and origins are asserted to match before and after adding the two macro
  columns.
* **Windows are deduplicated.** Each distinct ``(policy, cutoff month)`` is stored
  once; rows carry ``macro_window_index``.
* **Raw fixed units.** No scaling is applied. Per-scalar fit-population medians and
  standard deviations are written separately so a consumer can fit its own scaler
  without touching holdout or validation rows.
* **Never overwrite.** The output directory is named by ``contract_id`` which
  includes the source hashes; an existing directory is a hard error.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import coe, contract, dbs, windows

COLUMNS_ADDED = ("macro_cutoff_month", "macro_window_index")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash() -> dict:
    out = {}
    for path in sorted(contract.PACKAGE_DIR.glob("*.py")):
        out[path.name] = sha256_file(path)
    return out


def load_v2_manifest() -> dict:
    if not contract.V2_MANIFEST.exists():
        raise FileNotFoundError(
            f"v2.1 contract manifest not found at {contract.V2_MANIFEST}; "
            "v3 references it by path and does not rebuild it")
    return json.loads(contract.V2_MANIFEST.read_text())


def v2_sample_path(manifest: dict, split_id: str, stage: str) -> Path:
    entry = manifest["splits"][split_id]["stages"][stage]
    return contract.ROOT / entry["sample_csv"]


def load_v2_samples(manifest: dict, split_id: str, stage: str) -> pd.DataFrame:
    path = v2_sample_path(manifest, split_id, stage)
    frame = pd.read_csv(path)
    required = {"SOURCE_ROW_ID", "split_id", "stage", "population", "target_month_index",
                "origin_month_index", "horizon_months", "actual_rent_sgd"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"v2.1 sample table missing columns: {sorted(missing)}")
    return frame


def verify_membership(frame: pd.DataFrame, split_id: str, stage: str) -> None:
    """Assert the v2.1 invariants still hold; nothing is silently repaired."""
    if set(frame["split_id"].unique()) != {split_id}:
        raise AssertionError(f"{split_id}/{stage}: mixed split ids")
    if set(frame["stage"].unique()) != {stage}:
        raise AssertionError(f"{split_id}/{stage}: mixed stages")
    if (frame["origin_month_index"]
            != frame["target_month_index"] - frame["horizon_months"]).any():
        raise AssertionError(f"{split_id}/{stage}: origin != target - horizon")
    if not frame["SOURCE_ROW_ID"].is_unique:
        raise AssertionError(f"{split_id}/{stage}: duplicate SOURCE_ROW_ID")
    spec = contract.split_spec(split_id)
    valid = frame[frame["population"] == "validation"]
    if len(valid):
        if (valid["origin_month_index"] != spec["cutoff_index"]).any():
            raise AssertionError(f"{split_id}/{stage}: validation origin != cutoff")
        months = sorted(valid["target_month_index"].unique().tolist())
        expected = list(range(spec["val_start_index"], spec["val_end_index"] + 1))
        if months != expected:
            raise AssertionError(f"{split_id}/{stage}: validation months {months} != {expected}")


def attach_macro_columns(frame: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict]:
    """Add ``macro_cutoff_month`` and ``macro_window_index`` to a v2.1 table."""
    cutoffs = windows.macro_cutoff_months(frame, policy)
    row_window, unique, counts = windows.window_index(cutoffs)
    out = frame.copy()
    out["macro_cutoff_month"] = cutoffs.astype(np.int64)
    out["macro_window_index"] = row_window
    info = {
        "n_windows": int(len(unique)),
        "window_count_sum": int(counts.sum()),
        "n_rows": int(len(out)),
        "unique_cutoffs": [int(value) for value in unique],
        "unique_cutoffs_months": [contract.month_label(int(value)) for value in unique],
    }
    # Every row maps to exactly one window and the mapping covers all rows.
    if int(counts.sum()) != len(out):
        raise AssertionError("macro window counts do not cover every row")
    if int(row_window.max()) >= len(unique) if len(unique) else False:
        raise AssertionError("row_window_index out of range")
    return out, info


def scalar_encoding(scalars: pd.DataFrame, masks: pd.DataFrame, fit_mask: np.ndarray) -> dict:
    """Fit-population median/SD per scalar, ignoring masked entries."""
    out = {}
    for column in scalars.columns:
        values = scalars[column].to_numpy(dtype=float)
        usable = masks[column].to_numpy(dtype=float) > 0
        usable &= fit_mask
        finite = values[usable & np.isfinite(values)]
        if finite.size == 0:
            out[column] = {"median": None, "std": None, "n_used": 0}
            continue
        out[column] = {
            "median": float(np.median(finite)),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
            "n_used": int(finite.size),
        }
    return out


def export_split(
    split_id: str,
    stage: str,
    policy: str,
    v2_manifest: dict,
    dbs_frame,
    coe_monthly,
    output: Path,
    source_hashes: dict,
    hashes: dict,
) -> dict:
    samples = load_v2_samples(v2_manifest, split_id, stage)
    verify_membership(samples, split_id, stage)
    before_ids = samples["SOURCE_ROW_ID"].to_numpy()
    before_target = samples["actual_rent_sgd"].to_numpy(dtype=float)

    table, mapping_info = attach_macro_columns(samples, policy)
    arrays, window_summary = windows.build_macro_windows(
        table, policy, dbs_frame=dbs_frame, coe_monthly=coe_monthly)

    # --- gather window scalars back to rows ------------------------------
    row_window = arrays["row_window_index"]
    per_row: dict[str, np.ndarray] = {
        "row_window_index": row_window,
        "row_id": table["SOURCE_ROW_ID"].to_numpy(dtype=np.int64),
        "macro_cutoff_month": table["macro_cutoff_month"].to_numpy(dtype=np.int32),
    }
    if dbs_frame is not None:
        per_row["dbs_scalars"] = arrays["dbs_scalars"][row_window]
        per_row["dbs_scalar_mask"] = arrays["dbs_scalar_mask"][row_window]
    if coe_monthly is not None:
        per_row["coe_scalars"] = arrays["coe_scalars"][row_window]
        per_row["coe_scalar_mask"] = arrays["coe_scalar_mask"][row_window]

    # --- write ------------------------------------------------------------
    sample_path = output / "samples" / split_id / f"stage{stage}.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(sample_path, index=False)

    macro_path = output / "macro" / split_id / f"stage{stage}.npz"
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(macro_path, **arrays)

    features_path = output / "features" / split_id / f"stage{stage}.npz"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, **per_row)

    # --- encoding from the fit population only ---------------------------
    fit_mask = (table["population"] == "fit").to_numpy()
    encoding: dict = {
        "contract_version": contract.CONTRACT_VERSION,
        "policy": policy,
        "split_id": split_id,
        "stage": stage,
        "units": {
            "dbs_ret_*": "cumulative log return over N valid trading observations",
            "dbs_vol_*": "standard deviation of daily log returns, ddof=1",
            "dbs_drawdown_126": "ratio to trailing 126-observation peak, <= 0",
            "dbs_volume_rel_21": "difference of mean log-volume, recent 21 vs prior 21",
            "dbs_gap_max_days_63": "calendar days, largest gap inside the window",
            "dbs_asof_age_days": "days from as-of date to the cutoff month end",
            "coe_premium_*": "SGD per certificate of entitlement",
            "coe_quota": "vehicles offered",
            "coe_ratio": "bids / quota, summed over the month's auctions",
        },
        "fitted_on": "population == 'fit' of this stage only",
        "n_fit_rows": int(fit_mask.sum()),
    }
    if dbs_frame is not None:
        scalars = pd.DataFrame(per_row["dbs_scalars"], columns=contract.DBS_SCALARS)
        masks = pd.DataFrame(per_row["dbs_scalar_mask"], columns=contract.DBS_SCALARS)
        encoding["dbs_scalars"] = scalar_encoding(scalars, masks, fit_mask)
    if coe_monthly is not None:
        flat_names = contract.coe_scalar_names()
        coe_values = per_row["coe_scalars"].reshape(len(table), -1)
        coe_masks = per_row["coe_scalar_mask"].reshape(len(table), -1)
        scalars = pd.DataFrame(coe_values, columns=flat_names)
        masks = pd.DataFrame(coe_masks, columns=flat_names)
        encoding["coe_scalars"] = scalar_encoding(scalars, masks, fit_mask)

    encoding_path = output / "encoding" / split_id / f"stage{stage}.json"
    encoding_path.parent.mkdir(parents=True, exist_ok=True)
    encoding_path.write_text(json.dumps(encoding, indent=2) + "\n")

    # --- membership assertion after all transformations ------------------
    after = pd.read_csv(sample_path)
    if not np.array_equal(after["SOURCE_ROW_ID"].to_numpy(), before_ids):
        raise AssertionError(f"{split_id}/{stage}: row order changed during export")
    if not np.allclose(after["actual_rent_sgd"].to_numpy(dtype=float), before_target):
        raise AssertionError(f"{split_id}/{stage}: targets changed during export")

    v2_entry = v2_manifest["splits"][split_id]["stages"][stage]
    return {
        "policy": policy,
        "v2_sample_csv": str(v2_sample_path(v2_manifest, split_id, stage).relative_to(contract.ROOT)),
        "v2_sample_csv_sha256": v2_entry["sample_csv_sha256"],
        "n_rows": int(len(table)),
        "populations": table.groupby("population").size().to_dict(),
        "horizon_range": [int(table["horizon_months"].min()), int(table["horizon_months"].max())],
        "row_id_sha256": sha256_text(",".join(map(str, table["SOURCE_ROW_ID"].tolist()))),
        "window_mapping": mapping_info,
        "windows": window_summary,
        "artifacts": {
            "samples": {"path": str(sample_path.relative_to(contract.ROOT)),
                        "sha256": sha256_file(sample_path)},
            "macro": {"path": str(macro_path.relative_to(contract.ROOT)),
                      "sha256": sha256_file(macro_path)},
            "features": {"path": str(features_path.relative_to(contract.ROOT)),
                         "sha256": sha256_file(features_path)},
            "encoding": {"path": str(encoding_path.relative_to(contract.ROOT)),
                         "sha256": sha256_file(encoding_path)},
        },
    }


def main(policy: str = contract.DEFAULT_POLICY,
         split_ids=("orig", "early", "long"),
         stages=contract.STAGES) -> Path:
    if policy not in contract.POLICIES:
        raise ValueError(f"unknown policy {policy!r}")

    v2_manifest = load_v2_manifest()
    source_hashes = {
        "rent": sha256_file(contract.RENT_SOURCE),
        "stock": sha256_file(contract.STOCK_SOURCE),
        "coe": sha256_file(contract.COE_SOURCE),
    }
    # The v3 split definitions must agree with the v2.1 contract we inherit from.
    for split_id, spec in contract.SPLITS.items():
        v2_split = v2_manifest["splits"][split_id]["split"]
        for key in ("cutoff", "val_start", "val_end"):
            if spec[key] != v2_split[key]:
                raise AssertionError(
                    f"{split_id}: v3 {key}={spec[key]!r} disagrees with v2.1 {v2_split[key]!r}")

    if source_hashes["rent"] != v2_manifest["source_sha256"]:
        raise AssertionError("rent source hash differs from the v2.1 contract's")

    hashes = code_hash()
    contract_id = (f"{contract.CONTRACT_VERSION}__{source_hashes['rent'][:12]}"
                   f"__{source_hashes['stock'][:12]}__{source_hashes['coe'][:12]}")
    output = contract.REPORT_ROOT / contract_id
    if output.exists():
        raise FileExistsError(
            f"{output.relative_to(contract.ROOT)} already exists; refusing to overwrite. "
            "Change the inputs (which changes contract_id) or remove it deliberately.")
    output.mkdir(parents=True)

    print(f"policy={policy} contract_id={contract_id}", flush=True)
    dbs_frame = dbs.build_dbs_frame(dbs.parse_stock_frame())
    coe_monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    print(f"DBS rows={len(dbs_frame)}  COE months={len(coe_monthly.months)}", flush=True)

    manifest = {
        "contract_version": contract.CONTRACT_VERSION,
        "contract_id": contract_id,
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "policy": policy,
        "policy_description": contract.POLICY_DESCRIPTION[policy],
        "policies_implemented": list(contract.POLICIES),
        "source_hashes": source_hashes,
        "code_hashes": hashes,
        "v2_contract": {
            "path": str(contract.V2_CONTRACT_DIR.relative_to(contract.ROOT)),
            "contract_id": v2_manifest["contract_id"],
            "manifest_sha256": sha256_file(contract.V2_MANIFEST),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "contract_identity": contract.contract_identity(),
        "source_counts": {
            "dbs_rows": int(len(dbs_frame)),
            "dbs_observed": int(dbs_frame.observed.sum()),
            "dbs_asof_first": str(dbs_frame.dates.min().date()),
            "dbs_asof_last": str(dbs_frame.dates.max().date()),
            "coe_rows": int(len(coe.parse_coe_frame())),
            "coe_months": int(len(coe_monthly.months)),
            "coe_first_month": contract.month_label(int(coe_monthly.months.min())),
            "coe_last_month": contract.month_label(int(coe_monthly.months.max())),
        },
        "splits": {},
    }
    for split_id in split_ids:
        manifest["splits"][split_id] = {"stages": {}}
        manifest["splits"][split_id]["split"] = contract.split_spec(split_id)
        for stage in stages:
            print(f"exporting {split_id}/{stage} ...", flush=True)
            manifest["splits"][split_id]["stages"][stage] = export_split(
                split_id, stage, policy, v2_manifest, dbs_frame, coe_monthly,
                output, source_hashes, hashes)

    for name, path in (("rent", contract.RENT_SOURCE), ("stock", contract.STOCK_SOURCE),
                       ("coe", contract.COE_SOURCE)):
        if sha256_file(path) != source_hashes[name]:
            raise RuntimeError(f"{name} source changed during export")

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output.relative_to(contract.ROOT)}", flush=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=contract.DEFAULT_POLICY,
                        choices=list(contract.POLICIES))
    parser.add_argument("--split", action="append", choices=list(contract.SPLITS))
    parser.add_argument("--stage", action="append", choices=list(contract.STAGES))
    args = parser.parse_args()
    main(policy=args.policy,
         split_ids=tuple(args.split) if args.split else ("orig", "early", "long"),
         stages=tuple(args.stage) if args.stage else contract.STAGES)

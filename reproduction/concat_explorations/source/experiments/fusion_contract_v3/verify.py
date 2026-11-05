"""Independent verification of the exported v3 macro artifacts.

Reloads everything from disk and checks the plan's invariants without trusting
exporter state.  Run after any export; a reviewer can run it too.

Checks
------
1. Source, v2.1 manifest and code hashes match the manifest.
2. Row membership and order equal the v2.1 sample CSV the export claims to inherit.
3. Row IDs, targets, horizons and origins agree with the v2.1 arrays.
4. ``window_count`` sums to the row count and every ``macro_window_index`` is in range.
5. Per-row gathered scalars equal the window scalars they index.
6. The declared cutoff rule holds per population (``origin_asof`` only).
7. Padding is zero and masks are consistent (DBS, COE).
8. COE final month is flagged incomplete and category E's absent month is unobserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import contract, coe, dbs

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_contract_dir() -> Path:
    candidates = sorted(contract.REPORT_ROOT.glob("fusion-contract-v3*/manifest.json"))
    if not candidates:
        raise SystemExit("no v3 contract exported; run the exporter first")
    return candidates[-1].parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.contract_dir or latest_contract_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    print(f"contract_id: {manifest['contract_id']}")
    print(f"policy:      {manifest['policy']}\n")

    def resolve(relative: str) -> Path:
        return contract.ROOT / relative

    # ---- provenance ------------------------------------------------
    check("rent source hash matches", sha256_file(contract.RENT_SOURCE)
          == manifest["source_hashes"]["rent"])
    check("stock source hash matches", sha256_file(contract.STOCK_SOURCE)
          == manifest["source_hashes"]["stock"])
    check("coe source hash matches", sha256_file(contract.COE_SOURCE)
          == manifest["source_hashes"]["coe"])
    check("v2.1 manifest hash matches",
          sha256_file(contract.V2_MANIFEST) == manifest["v2_contract"]["manifest_sha256"])
    for name, recorded in manifest["code_hashes"].items():
        actual = sha256_file(contract.PACKAGE_DIR / name)
        check(f"code hash {name}", actual == recorded, f"{actual} != {recorded}")

    v2_manifest = json.loads(contract.V2_MANIFEST.read_text())
    v2_order = v2_manifest["splits"]
    dbs_frame = dbs.build_dbs_frame(dbs.parse_stock_frame())
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())

    for split_id, split_block in manifest["splits"].items():
        spec = contract.split_spec(split_id)
        print(f"\n--- split {split_id} ({spec['cutoff']} -> {spec['val_end']})")
        for stage, entry in split_block["stages"].items():
            samples = pd.read_csv(resolve(entry["artifacts"]["samples"]["path"]))
            macro = np.load(resolve(entry["artifacts"]["macro"]["path"]))
            features = np.load(resolve(entry["artifacts"]["features"]["path"]))

            # ---- membership inherited from v2.1 ----------------------
            v2_entry = v2_manifest["splits"][split_id]["stages"][stage]
            v2_samples = pd.read_csv(contract.ROOT / v2_entry["sample_csv"])
            check(f"{split_id}/{stage}: row ids equal the v2.1 table",
                  np.array_equal(samples["SOURCE_ROW_ID"].to_numpy(),
                                 v2_samples["SOURCE_ROW_ID"].to_numpy()))
            check(f"{split_id}/{stage}: populations equal the v2.1 table",
                  samples["population"].tolist() == v2_samples["population"].tolist())
            check(f"{split_id}/{stage}: targets equal the v2.1 table",
                  np.allclose(samples["actual_rent_sgd"].to_numpy(),
                              v2_samples["actual_rent_sgd"].to_numpy(), rtol=0, atol=0))
            check(f"{split_id}/{stage}: horizons equal the v2.1 table",
                  np.array_equal(samples["horizon_months"].to_numpy(),
                                 v2_samples["horizon_months"].to_numpy()))
            check(f"{split_id}/{stage}: origins equal the v2.1 table",
                  np.array_equal(samples["origin_month_index"].to_numpy(),
                                 v2_samples["origin_month_index"].to_numpy()))
            check(f"{split_id}/{stage}: recorded v2.1 sample hash matches",
                  sha256_file(contract.ROOT / v2_entry["sample_csv"])
                  == entry["v2_sample_csv_sha256"])

            # ---- window mapping -------------------------------------
            counts = macro["window_count"]
            row_window = macro["row_window_index"]
            check(f"{split_id}/{stage}: window counts sum to the row count",
                  int(counts.sum()) == len(samples), f"{int(counts.sum())} vs {len(samples)}")
            check(f"{split_id}/{stage}: row_window_index in range",
                  int(row_window.max()) < len(counts))
            check(f"{split_id}/{stage}: features row_window_index matches the macro block",
                  np.array_equal(features["row_window_index"], row_window))
            check(f"{split_id}/{stage}: row_id in the feature block matches the table",
                  np.array_equal(features["row_id"], samples["SOURCE_ROW_ID"].to_numpy()))

            # ---- per-row gathered scalars equal their window --------
            gathered_dbs = features["dbs_scalars"]
            window_dbs = macro["dbs_scalars"]
            check(f"{split_id}/{stage}: DBS scalars are a gather of the window block",
                  np.allclose(gathered_dbs, window_dbs[row_window],
                              rtol=0, atol=0, equal_nan=True))
            gathered_coe = features["coe_scalars"]
            window_coe = macro["coe_scalars"]
            check(f"{split_id}/{stage}: COE scalars are a gather of the window block",
                  np.allclose(gathered_coe, window_coe[row_window],
                              rtol=0, atol=0, equal_nan=True))

            # ---- cutoff rule ----------------------------------------
            if manifest["policy"] == contract.POLICY_ORIGIN_ASOF:
                check(f"{split_id}/{stage}: macro cutoff equals the row origin",
                      np.array_equal(samples["macro_cutoff_month"].to_numpy(),
                                     samples["origin_month_index"].to_numpy()))
                check(f"{split_id}/{stage}: no cutoff exceeds the split cutoff",
                      int(samples["macro_cutoff_month"].max()) <= spec["cutoff_index"])
            elif manifest["policy"] == contract.POLICY_TARGET_PREVIOUS_MONTH:
                check(f"{split_id}/{stage}: macro cutoff is the month before the target",
                      np.array_equal(samples["macro_cutoff_month"].to_numpy(),
                                     samples["target_month_index"].to_numpy() - 1))

            # ---- DBS padding and as-of ------------------------------
            dbs_mask = macro["dbs_mask"]
            dbs_seq = macro["dbs_seq"]
            dbs_dates = macro["dbs_dates"]
            check(f"{split_id}/{stage}: DBS padded positions are zero",
                  bool(np.all(dbs_seq[dbs_mask == 0] == 0)))
            check(f"{split_id}/{stage}: DBS padded dates are zero",
                  bool(np.all(dbs_dates[dbs_mask == 0] == 0)))
            check(f"{split_id}/{stage}: DBS as-of date is the last masked-in date",
                  bool(np.all(dbs_dates[mask_last(dbs_mask)] == macro["dbs_asof_date"])))
            check(f"{split_id}/{stage}: DBS as-of dates never exceed the cutoff month",
                  all(int(str(value)[:6]) <= _yyyymm(spec_cutoff)
                      for value, spec_cutoff in zip(macro["dbs_asof_date"],
                                                    macro["window_cutoff_month"])))

            # ---- COE padding, incomplete, category E ----------------
            coe_mask = macro["coe_mask"]
            coe_seq = macro["coe_seq"]
            check(f"{split_id}/{stage}: COE padded positions are zero",
                  bool(np.all(coe_seq[coe_mask == 0] == 0)))
            check(f"{split_id}/{stage}: COE month ids are YYYYMM or zero",
                  bool(np.all((macro["coe_months"] == 0)
                              | ((macro["coe_months"] >= 202101)
                                 & (macro["coe_months"] <= 203012)))))

    # ---- COE source-level facts ------------------------------------
    print("\n--- COE source facts")
    last = monthly.month_position(monthly.last_source_month)
    check("final source month is 2026-08",
          contract.month_label(monthly.last_source_month) == "2026-08")
    check("final month is flagged incomplete for every category",
          monthly.incomplete[:, last].tolist() == [1.0, 1.0, 1.0, 1.0])
    e_index = list(monthly.categories).index("e")
    check("category E has no 2026-08 row (absent, not zero)",
          monthly.observed[e_index, last] == 0.0)
    check("category E premiums are all finite and positive where observed",
          bool(np.all(monthly.premium[e_index][monthly.observed[e_index] > 0] > 0)))

    # ---- DBS source-level facts ------------------------------------
    print("\n--- DBS source facts")
    check("DBS has exactly 1403 rows", len(dbs_frame) == 1403)
    check("DBS has exactly 1401 valid closes", int(dbs_frame.observed.sum()) == 1401)
    check("DBS gap days never exceed the masking threshold on valid one-step returns",
          bool(np.all(dbs_frame.gap_days[np.isfinite(dbs_frame.log_return)]
                      <= dbs.MAX_CALENDAR_GAP_DAYS)))

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all v3 contract verification checks passed")
    return 0


def mask_last(mask: np.ndarray) -> np.ndarray:
    """Boolean mask selecting, per row, the last position with mask == 1."""
    out = np.zeros(mask.shape, dtype=bool)
    for index in range(mask.shape[0]):
        hits = np.flatnonzero(mask[index] > 0)
        if hits.size:
            out[index, hits[-1]] = True
    return out


def _yyyymm(month_index: int) -> int:
    return (2021 + month_index // 12) * 100 + (month_index % 12 + 1)


if __name__ == "__main__":
    raise SystemExit(main())

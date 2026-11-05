"""Verify the exported contract artifacts against the source, independently.

This is the check a reviewer (or the neural pane) should be able to run without
trusting the exporter: it re-reads the CSVs and ``.npz`` files from disk, rebuilds
nothing from the exporter's in-memory state, and asserts the plan's Gate-1
invariants directly from the artifacts.

Run:
    .venv/bin/python -m experiments.fusion_contract_v2.verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v2 import contract

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def latest_contract_dir() -> Path:
    candidates = sorted(contract.REPORT_ROOT.glob("fusion-contract-v*/manifest.json"))
    if not candidates:
        raise SystemExit("no exported contract found; run the exporter first")
    return candidates[-1].parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.contract_dir or latest_contract_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    print(f"contract_id: {manifest['contract_id']}\n")

    # ---- source integrity -------------------------------------------
    from experiments.fusion_contract_v2 import export as exporter
    check("source hash matches the manifest",
          exporter.sha256_file(contract.SOURCE) == manifest["source_sha256"],
          manifest["source_sha256"])
    for name, recorded in manifest["code_hashes"].items():
        actual = exporter.sha256_file(contract.PACKAGE_DIR / name)
        check(f"code hash {name}", actual == recorded, f"{actual} != {recorded}")

    def resolve(relative: str) -> Path:
        """Manifest paths are repo-relative; resolve them from the project root."""
        return contract.ROOT / relative

    source = pd.read_csv(contract.SOURCE, usecols=[
        contract.ID_COL, contract.TARGET, contract.DATE_COL, contract.MONTH_COL,
        "TOWN_CLEAN", "FLAT_TYPE_CLEAN"])
    source_full = pd.read_csv(contract.SOURCE, usecols=[
        contract.ID_COL, "TOWN_CLEAN", "STREET_CLEAN", "BLOCK_CLEAN", "FLAT_TYPE_CLEAN"])

    for split_id, split_manifest in manifest["splits"].items():
        spec = contract.split_spec(split_id)
        print(f"\n--- split {split_id} ({spec['cutoff']} -> {spec['val_end']})")
        check(f"{split_id}: declared validation months",
              spec["n_validation_months"] == split_manifest["split"]["n_validation_months"])

        for stage, stage_manifest in split_manifest["stages"].items():
            samples = pd.read_csv(resolve(stage_manifest["sample_csv"]))
            arrays = np.load(resolve(stage_manifest["arrays"]["path"]))

            # -- row/ID/target/date alignment -------------------------
            check(f"{split_id}/{stage}: array rows match the sample table",
                  len(arrays["row_id"]) == len(samples),
                  f"{len(arrays['row_id'])} vs {len(samples)}")
            check(f"{split_id}/{stage}: row ids align in order",
                  np.array_equal(arrays["row_id"], samples[contract.ID_COL].to_numpy()))
            check(f"{split_id}/{stage}: targets align",
                  np.allclose(arrays["target_sgd"], samples["actual_rent_sgd"], rtol=0, atol=1e-3))
            check(f"{split_id}/{stage}: horizons align",
                  np.array_equal(arrays["horizon"], samples["horizon_months"].to_numpy()))
            check(f"{split_id}/{stage}: origins align",
                  np.array_equal(arrays["origin"], samples["origin_month_index"].to_numpy()))
            check(f"{split_id}/{stage}: recorded row-id hash reproduces",
                  exporter.sha256_text(",".join(map(str, samples[contract.ID_COL].tolist())))
                  == stage_manifest["row_id_sha256"])

            # -- information boundaries ------------------------------
            check(f"{split_id}/{stage}: origin == target - horizon for every row",
                  bool((samples["origin_month_index"]
                        == samples["target_month_index"] - samples["horizon_months"]).all()))
            check(f"{split_id}/{stage}: horizons are 1..MAX",
                  bool(samples["horizon_months"].between(1, contract.MAX_HORIZON).all()))
            check(f"{split_id}/{stage}: no origin exceeds the cutoff",
                  int(samples["origin_month_index"].max()) <= spec["cutoff_index"],
                  str(int(samples["origin_month_index"].max())))

            valid = samples[samples["population"] == "validation"]
            if len(valid):
                check(f"{split_id}/{stage}: validation origin is the cutoff",
                      bool((valid["origin_month_index"] == spec["cutoff_index"]).all()))
                check(f"{split_id}/{stage}: validation horizon == target - cutoff",
                      bool((valid["horizon_months"]
                            == valid["target_month_index"] - spec["cutoff_index"]).all()))
                check(f"{split_id}/{stage}: validation months are contiguous",
                      sorted(valid["target_month_index"].unique().tolist())
                      == list(range(spec["val_start_index"], spec["val_end_index"] + 1)),
                      str(sorted(valid["target_month_index"].unique().tolist())))
                expected_rows = int(
                    ((source[contract.MONTH_COL] >= spec["val_start_index"])
                     & (source[contract.MONTH_COL] <= spec["val_end_index"])).sum())
                check(f"{split_id}/{stage}: validation rows equal the source row count in the window",
                      len(valid) == expected_rows and len(valid) > 0,
                      f"{len(valid)} vs {expected_rows}")

            # -- stage A stops before the outer fitting months --------
            if stage == "A":
                inner = spec["inner_cutoff_index"]
                fit = samples[samples["population"] == "fit"]
                holdout = samples[samples["population"] == "holdout"]
                check(f"{split_id}/A: fit population uses only months <= inner cutoff",
                      int(fit["target_month_index"].max()) <= inner,
                      str(int(fit["target_month_index"].max())))
                check(f"{split_id}/A: holdout population lies strictly after the inner cutoff",
                      int(holdout["target_month_index"].min()) > inner)
                check(f"{split_id}/A: holdout stops at the cutoff",
                      int(holdout["target_month_index"].max()) <= spec["cutoff_index"])
                check(f"{split_id}/A: panel fit_end equals the inner cutoff",
                      stage_manifest["fit_end_index"] == inner)
            else:
                fit = samples[samples["population"] == "fit"]
                check(f"{split_id}/B: fit population covers through the cutoff",
                      int(fit["target_month_index"].max()) <= spec["cutoff_index"]
                      and int(fit["target_month_index"].max()) > spec["inner_cutoff_index"])
                check(f"{split_id}/B: panel fit_end equals the cutoff",
                      stage_manifest["fit_end_index"] == spec["cutoff_index"])

            # -- array shapes and finiteness -------------------------
            n = len(samples)
            check(f"{split_id}/{stage}: static shape",
                  arrays["static"].shape == (n, len(contract.static_columns())),
                  str(arrays["static"].shape))
            check(f"{split_id}/{stage}: context shape",
                  arrays["context"].shape == (n, len(contract.context_columns())),
                  str(arrays["context"].shape))
            check(f"{split_id}/{stage}: history shape",
                  arrays["history"].shape == (n, len(contract.history_columns())),
                  str(arrays["history"].shape))
            expected_sequence = (n, len(contract.HISTORY_LEVELS),
                                 contract.HISTORY_LEN, len(contract.PANEL_CHANNELS))
            check(f"{split_id}/{stage}: sequence shape",
                  arrays["sequence"].shape == expected_sequence, str(arrays["sequence"].shape))
            check(f"{split_id}/{stage}: mask shape",
                  arrays["sequence_mask"].shape == expected_sequence[:3])
            check(f"{split_id}/{stage}: padding positions are zero",
                  bool(np.all(arrays["sequence"][arrays["sequence_mask"] == 0] == 0)))

        # -- validation features frozen across horizons --------------
        stage_b = pd.read_csv(resolve(split_manifest["stages"]["B"]["sample_csv"]))
        arrays_b = np.load(resolve(split_manifest["stages"]["B"]["arrays"]["path"]))
        valid_mask = (stage_b["population"] == "validation").to_numpy()
        if valid_mask.any():
            details = stage_b.loc[valid_mask, [contract.ID_COL]].merge(
                source_full[[contract.ID_COL, "TOWN_CLEAN", "FLAT_TYPE_CLEAN",
                             "STREET_CLEAN", "BLOCK_CLEAN"]],
                on=contract.ID_COL, how="left")
            history = pd.DataFrame(arrays_b["history"][valid_mask],
                                  columns=contract.history_columns())

            # A frozen snapshot means: within one *level key*, every horizon sees
            # the identical value.  Grouping matters -- aggregating two blocks
            # together would show legitimate block-level differences.
            level_group = {
                "global": pd.Series(["__all__"] * len(details)),
                "flat_type": details["FLAT_TYPE_CLEAN"].astype(str),
                "town_ft": (details["TOWN_CLEAN"].astype(str) + "|"
                            + details["FLAT_TYPE_CLEAN"].astype(str)),
                "block_ft": (details["TOWN_CLEAN"].astype(str) + "|"
                             + details["STREET_CLEAN"].astype(str) + "|"
                             + details["BLOCK_CLEAN"].astype(str) + "|"
                             + details["FLAT_TYPE_CLEAN"].astype(str)),
            }
            varying = []
            for level in contract.HISTORY_LEVELS:
                keys = level_group[level].to_numpy()
                for stat in contract.HISTORY_STATS:
                    column = contract.history_column(level, stat)
                    spread = history[column].groupby(keys).nunique(dropna=False)
                    if (spread > 1).any():
                        varying.append(column)
            check(f"{split_id}: validation history is frozen across horizons",
                  varying == [], str(varying[:5]))

            # sequence: the final slot of a 12-month window is the boundary month,
            # so it must be constant per level key across every horizon too.
            seq = arrays_b["sequence"][valid_mask]
            last_slot = seq[:, :, contract.HISTORY_LEN - 1, 2]  # log1p(count)
            varying_seq = []
            for index, level in enumerate(contract.HISTORY_LEVELS):
                keys = level_group[level].to_numpy()
                spread = pd.Series(last_slot[:, index]).groupby(keys).nunique(dropna=False)
                if (spread > 1).any():
                    varying_seq.append(level)
            check(f"{split_id}: validation sequence tail is frozen across horizons",
                  varying_seq == [], str(varying_seq))

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all contract verification checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

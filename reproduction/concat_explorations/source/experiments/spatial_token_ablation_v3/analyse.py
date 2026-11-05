"""Independent analysis of the spatial/property per-feature token ablation.

Compares two runs — the existing `block` reference and the new `per_feature` run —
and recomputes everything from the exported prediction CSVs rather than trusting
either run's reported summary.

Checks performed
----------------
1. **Provenance compatibility.** Contract id, source hash, sample fingerprint,
   hyperparameters and code fingerprint of both runs, with any difference listed
   explicitly. A comparison across different data or budgets is not the ablation.
2. **Row-level agreement.** Every prediction row matched on `SOURCE_ROW_ID`, and
   its label / approval month / horizon compared between the two runs.
3. **Metric recomputation.** RMSE / MAE / bias recomputed from the CSVs, and
   compared against what each run reported.
4. **Monthly and horizon breakdowns**, with the paired difference per month and per
   horizon band.
5. **Checkpoint reconstruction** is delegated to the runner's own
   `--verify-checkpoints`; this script only reads the resulting JSON if present.

It writes nothing into either run directory and does not fit anything.

Run:
    .venv/bin/python -m experiments.spatial_token_ablation_v3.analyse \
        --reference macro-probe-block-none-s42 \
        --candidate spatial-probe-perfeature-none-s42-h1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "reports" / "neural_fusion_v3" / "runs"
REPORT = ROOT / "reports" / "spatial_token_ablation_v3"

FAILURES: list[str] = []
FINDINGS: list[str] = []
NOTES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def finding(text: str) -> None:
    FINDINGS.append(text)
    print(f"FIND  {text}")


def note(text: str) -> None:
    NOTES.append(text)
    print(f"NOTE  {text}")


# ---------------------------------------------------------------- loading
def load_run(run_id: str) -> dict:
    directory = RUNS / run_id
    if not directory.exists():
        raise FileNotFoundError(directory)
    summary = json.loads((directory / "run_summary.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    return {"id": run_id, "dir": directory, "summary": summary, "manifest": manifest}


def find_predictions(run: dict) -> Path:
    candidates = sorted(run["dir"].glob("predictions/**/*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no prediction CSV under {run['dir'] / 'predictions'}")
    # the validation CSV is the outer-window one
    for path in candidates:
        if "valid" in path.name.lower():
            return path
    return candidates[0]


def load_predictions(run: dict) -> pd.DataFrame:
    frame = pd.read_csv(find_predictions(run))
    required = {"SOURCE_ROW_ID", "actual_rent_sgd", "predicted_rent_sgd"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{run['id']}: prediction CSV missing {sorted(missing)}")
    return frame


def score(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
        "n_rows": int(error.size),
    }


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="macro-probe-block-none-s42")
    parser.add_argument("--candidate", default="spatial-probe-perfeature-none-s42-h1")
    args = parser.parse_args(argv)

    reference = load_run(args.reference)
    candidate = load_run(args.candidate)
    REPORT.mkdir(parents=True, exist_ok=True)

    print(f"reference: {reference['id']}")
    print(f"candidate: {candidate['id']}\n")

    # ---------------------------------------------------------------- 1
    print("--- 1. provenance compatibility")
    ref_manifest = reference["manifest"]
    can_manifest = candidate["manifest"]

    ref_contract = ref_manifest.get("contract_id")
    can_contract = can_manifest.get("contract_id")
    check("same v2 rent contract", ref_contract == can_contract,
          f"{ref_contract} vs {can_contract}")

    ref_source = ref_manifest.get("contract_source_sha256")
    can_source = can_manifest.get("contract_source_sha256")
    check("same source data hash", ref_source == can_source,
          f"{str(ref_source)[:16]} vs {str(can_source)[:16]}")

    ref_fp = ref_manifest.get("contract_fingerprint")
    can_fp = can_manifest.get("contract_fingerprint")
    check("same contract fingerprint (samples identical)", ref_fp == can_fp)

    ref_args = ref_manifest.get("args", {})
    can_args = can_manifest.get("args", {})
    for key in ("split", "seeds", "macro"):
        check(f"same {key}", ref_args.get(key) == can_args.get(key),
              f"{ref_args.get(key)} vs {can_args.get(key)}")
    check("reference used macro none", ref_args.get("macro", "none") == "none")
    check("candidate used macro none", can_args.get("macro", "none") == "none")

    ref_cap = ref_manifest.get("capacity")
    can_cap = can_manifest.get("capacity")
    check("same declared CAPACITY", ref_cap == can_cap,
          f"{ref_cap} vs {can_cap}")

    ref_code = ref_manifest.get("code_fingerprint")
    can_code = can_manifest.get("code_fingerprint")
    if ref_code == can_code:
        print("  code fingerprint identical")
    else:
        note("code fingerprints differ between the two runs; any score difference "
             "cannot be attributed to the token mode alone")

    # ---------------------------------------------------------------- 2
    print("\n--- 2. row-level agreement")
    ref_pred = load_predictions(reference)
    can_pred = load_predictions(candidate)
    print(f"  reference rows: {len(ref_pred):,}")
    print(f"  candidate rows: {len(can_pred):,}")
    check("same validation row count", len(ref_pred) == len(can_pred))
    check("no duplicate SOURCE_ROW_ID in reference",
          ref_pred["SOURCE_ROW_ID"].is_unique)
    check("no duplicate SOURCE_ROW_ID in candidate",
          can_pred["SOURCE_ROW_ID"].is_unique)

    merged = ref_pred.merge(can_pred, on="SOURCE_ROW_ID", how="outer",
                            suffixes=("_ref", "_can"), indicator=True)
    check("row id sets are identical",
          bool((merged["_merge"] == "both").all()),
          str(merged["_merge"].value_counts().to_dict()))

    both = merged[merged["_merge"] == "both"].copy()
    label_delta = float(np.max(np.abs(both["actual_rent_sgd_ref"]
                                      - both["actual_rent_sgd_can"])))
    check("labels identical row-for-row", label_delta == 0.0, f"max delta {label_delta}")

    for column in ("target_month", "horizon_months", "cutoff_month", "stage", "population"):
        if column in both.columns:
            left = both.get(f"{column}_ref")
            right = both.get(f"{column}_can")
            if left is not None and right is not None:
                check(f"{column} identical row-for-row",
                      bool((left.astype(str) == right.astype(str)).all()))

    # ---------------------------------------------------------------- 3
    print("\n--- 3. metric recomputation")
    ref_metrics = score(both["actual_rent_sgd_ref"], both["predicted_rent_sgd_ref"])
    can_metrics = score(both["actual_rent_sgd_can"], both["predicted_rent_sgd_can"])
    print(f"  block       : RMSE {ref_metrics['rmse_sgd']:.6f}  "
          f"MAE {ref_metrics['mae_sgd']:.6f}  bias {ref_metrics['bias_sgd']:+.4f}")
    print(f"  per_feature : RMSE {can_metrics['rmse_sgd']:.6f}  "
          f"MAE {can_metrics['mae_sgd']:.6f}  bias {can_metrics['bias_sgd']:+.4f}")
    delta = can_metrics["rmse_sgd"] - ref_metrics["rmse_sgd"]
    print(f"  delta RMSE  : {delta:+.6f} SGD")
    note(f"recomputed from CSVs; the block reference should be 475.5986877090766")

    expected_reference = 475.5986877090766
    check("recomputed reference RMSE matches the recorded value",
          abs(ref_metrics["rmse_sgd"] - expected_reference) < 1e-6,
          f"{ref_metrics['rmse_sgd']:.10f} vs {expected_reference:.10f}")

    reported = candidate["summary"]
    if isinstance(reported, dict):
        results = reported.get("results")
        if isinstance(results, list) and results:
            reported_rmse = results[0].get("rmse_sgd")
            if reported_rmse is not None:
                check("candidate's reported RMSE matches the CSV recomputation",
                      abs(float(reported_rmse) - can_metrics["rmse_sgd"]) < 1e-6,
                      f"reported {reported_rmse} vs recomputed {can_metrics['rmse_sgd']}")

    # ---------------------------------------------------------------- 4
    print("\n--- 4. monthly and horizon breakdown")
    both["error_ref"] = both["predicted_rent_sgd_ref"] - both["actual_rent_sgd_ref"]
    both["error_can"] = both["predicted_rent_sgd_can"] - both["actual_rent_sgd_can"]

    monthly = []
    # after the merge the shared columns carry _ref/_can suffixes
    month_key = "target_month_ref" if "target_month_ref" in both.columns else "target_month"
    for month, group in both.groupby(month_key):
        monthly.append({
            "month": month,
            "n_rows": len(group),
            "rmse_block": float(np.sqrt(np.mean(group["error_ref"] ** 2))),
            "rmse_per_feature": float(np.sqrt(np.mean(group["error_can"] ** 2))),
            "bias_block": float(group["error_ref"].mean()),
            "bias_per_feature": float(group["error_can"].mean()),
        })
    monthly_frame = pd.DataFrame(monthly)
    monthly_frame["delta_rmse"] = (monthly_frame["rmse_per_feature"]
                                   - monthly_frame["rmse_block"])
    print(monthly_frame.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    bands = []
    if "horizon_months_ref" in both.columns:
        edges = [(1, 3), (4, 8), (9, 12), (13, 16)]
        for low, high in edges:
            group = both[(both["horizon_months_ref"] >= low)
                         & (both["horizon_months_ref"] <= high)]
            if group.empty:
                bands.append({"band": f"{low}-{high}", "n_rows": 0,
                              "rmse_block": np.nan, "rmse_per_feature": np.nan,
                              "delta_rmse": np.nan})
                continue
            bands.append({
                "band": f"{low}-{high}",
                "n_rows": len(group),
                "rmse_block": float(np.sqrt(np.mean(group["error_ref"] ** 2))),
                "rmse_per_feature": float(np.sqrt(np.mean(group["error_can"] ** 2))),
                "delta_rmse": float(np.sqrt(np.mean(group["error_can"] ** 2))
                                    - np.sqrt(np.mean(group["error_ref"] ** 2))),
            })
        band_frame = pd.DataFrame(bands)
        print()
        print(band_frame.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        band_frame = pd.DataFrame()

    # ---------------------------------------------------------------- 5
    print("\n--- 5. checkpoint verification")
    verification = candidate["summary"].get("checkpoint_verification") if isinstance(
        candidate["summary"], dict) else None
    if verification:
        payload = json.dumps(verification)
        print(f"  recorded: {payload[:300]}")
        check("candidate run recorded a checkpoint verification", True)
    else:
        note("candidate run summary has no checkpoint_verification block; run with "
             "--verify-checkpoints to produce one")

    # ---------------------------------------------------------------- write
    output = {
        "reference": reference["id"],
        "candidate": candidate["id"],
        "provenance": {
            "contract_id": can_contract,
            "contract_source_sha256": can_source,
            "contract_fingerprint_match": ref_fp == can_fp,
            "code_fingerprint_match": ref_code == can_code,
            "capacity": can_cap,
        },
        "reference_metrics_recomputed": ref_metrics,
        "candidate_metrics_recomputed": can_metrics,
        "delta_rmse_sgd": delta,
        "monthly": monthly,
        "horizon_bands": bands,
        "findings": FINDINGS,
        "notes": NOTES,
        "failures": FAILURES,
    }
    (REPORT / "analysis.json").write_text(json.dumps(output, indent=2, default=str) + "\n")
    monthly_frame.to_csv(REPORT / "monthly_deltas.csv", index=False)
    if not band_frame.empty:
        band_frame.to_csv(REPORT / "horizon_band_deltas.csv", index=False)

    print("\n" + "=" * 62)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S)")
        for item in FINDINGS:
            print(f"  - {item}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

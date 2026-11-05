"""Phase-2 diagnostics E and F: within-group dispersion and inner-residual structure.

Two separate questions, kept apart from any training.

**E — how much variation remains inside coarse groups?**

Groups use the *cleaned* block key. Plain `BLOCK` is not a building identifier in this
data: the same block number recurs across streets and towns, so grouping on `BLOCK` alone
merges unrelated buildings. Keys are built from the processed spatial table:
`BLOCK_CLEAN + STREET_CLEAN` and `BLOCK_CLEAN + STREET_CLEAN + TOWN_CLEAN`.

For each grouping it reports how many groups reach sizes >=2, >=3 and >=5 **together with
the share of rows those groups cover**, because a grouping that isolates a handful of
large repeats says little about the population. The dispersion statistic is computed on the
rows inside the qualifying groups only, and both a plain within-group MAE and a
DOF-corrected variance are reported. The MAE is deliberately **not** divided by `n - g`:
that ratio has no standard interpretation. Only the variance has a DOF-corrected form
(`SSE / (n - g)`), and it is labelled as such.

This is residual variation **conditional on a coarse grouping of features we already
have**. It is NOT a proven irreducible-noise floor: unobserved unit-level differences
(renovation state, exact floor, view, negotiation) remain inside those groups. It must
never be subtracted from an RMSE to claim headroom.

**F — what shape do the inner residuals have?**

Join `inner_predictions.npz` (trial6's A/holdout predictions) to the contract by
SOURCE_ROW_ID with `validate="one_to_one"`, and **assert** that ids, labels and horizons
agree exactly. Then describe error by town, region, flat type, area bin, horizon, and by
*actual* versus *predicted* price band.

Band-by-actual is post-hoc and label-conditioned: it can look like structure only because
the band is defined on the target. Band-by-predicted is the honest heteroscedasticity view.

Inner-only. No B/validation is loaded. No parameter is selected.

Run:
    OMP_NUM_THREADS=1 .venv/bin/python -m experiments.diagnostics_phase2_a.dispersion
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
SEARCH = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-mps-s42"
OUT = ROOT / "reports" / "diagnostics_phase2_a"
PROCESSED = ROOT / "data" / "processed" / "train_spatial_features.csv"

INNER_CUTOFF_MONTH = 26          # target month index; 26 == 2023-03
A_HOLDOUT_ROWS = 35882


def within_group_stats(frame: pd.DataFrame, keys: list[str], label: str,
                       min_size: int) -> dict:
    """Dispersion among rows in groups of at least `min_size`.

    MAE is a plain within-group MAE (no DOF division, which would have no meaning for an
    absolute-error statistic). The variance is reported both raw and DOF-corrected as
    `SSE / (n - g)`.
    """
    grouped = frame.groupby(keys, sort=False, observed=True)["actual_rent_sgd"]
    sizes = grouped.size()
    qualifying = sizes[sizes >= min_size].index

    if len(qualifying) == 0:
        return {"grouping": label, "keys": keys, "min_group_size": min_size,
                "n_groups_total": int(len(sizes)), "n_groups_qualifying": 0,
                "rows_qualifying": 0, "rows_qualifying_fraction": 0.0,
                "note": "no group reached the threshold"}

    mask = frame.set_index(keys).index.isin(qualifying)
    subset = frame[mask]
    sub_grouped = subset.groupby(keys, sort=False, observed=True)["actual_rent_sgd"]
    means = sub_grouped.transform("mean")

    n = int(len(subset))
    g = int(len(qualifying))
    sse = float(((subset["actual_rent_sgd"] - means) ** 2).sum())
    sae = float((subset["actual_rent_sgd"] - means).abs().sum())
    dof = n - g

    return {
        "grouping": label, "keys": keys, "min_group_size": min_size,
        "n_groups_total": int(len(sizes)),
        "n_groups_qualifying": g,
        "groups_ge2": int((sizes >= 2).sum()),
        "groups_ge3": int((sizes >= 3).sum()),
        "groups_ge5": int((sizes >= 5).sum()),
        "rows_qualifying": n,
        "rows_qualifying_fraction": round(n / len(frame), 6),
        "median_group_size": float(sizes.median()),
        "max_group_size": int(sizes.max()),
        "within_mae_plain": sae / n,
        "within_mse_raw": sse / n,
        "within_mse_dof_corrected": (sse / dof) if dof > 0 else None,
        "within_rmse_dof_corrected": (float(np.sqrt(sse / dof)) if dof > 0 else None),
        "mae_units_note": ("within_mae_plain is SAE / n inside the qualifying rows; an MAE "
                           "has no DOF-corrected form and is not divided by n - g"),
    }


def bin_by(series: pd.Series, bins: int) -> pd.Series:
    try:
        return pd.qcut(series, q=bins, duplicates="drop", labels=False)
    except ValueError:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)


def describe_residuals(frame: pd.DataFrame, group: str, label: str,
                       conditioning: str) -> list[dict]:
    rows = []
    for value, block in frame.groupby(group, sort=True, observed=True):
        if len(block) < 30:
            continue
        error = block["predicted_rent_sgd"].to_numpy() - block["actual_rent_sgd"].to_numpy()
        rows.append({
            "grouping": label, "conditioning": conditioning,
            "bucket": value if isinstance(value, str) else float(value),
            "n": int(len(block)),
            "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
            "mae_sgd": float(np.mean(np.abs(error))),
            "bias_sgd": float(np.mean(error)),
            "error_var_sgd2": float(np.var(error)),
        })
    return rows


def main() -> int:
    from neural_fusion_v2 import contract as C

    OUT.mkdir(parents=True, exist_ok=True)
    print("--- loading processed source table and contract populations (inner only)")

    processed = pd.read_csv(PROCESSED, low_memory=False)
    print(f"      {PROCESSED.name}: {len(processed):,} rows, {len(processed.columns)} cols")
    required = {"SOURCE_ROW_ID", "BLOCK_CLEAN", "STREET_CLEAN", "TOWN_CLEAN",
                "FLAT_TYPE", "FURNISHED", "REGION", "FLOOR_AREA_SQM"}
    missing = required - set(processed.columns)
    if missing:
        raise SystemExit(f"processed table is missing {sorted(missing)}")
    if not processed["SOURCE_ROW_ID"].is_unique:
        raise SystemExit("processed SOURCE_ROW_ID is not unique")

    manifest = C.load_manifest(V2_DIR)
    a_fit = C.load_population(V2_DIR, manifest, "orig", "A", "fit")
    a_hold = C.load_population(V2_DIR, manifest, "orig", "A", "holdout")
    fit_rows = a_fit.samples.copy()
    hold_rows = a_hold.samples.copy()
    print(f"      A/fit {len(fit_rows):,}  A/holdout {len(hold_rows):,}")

    # ---------------------------------------------------------------- E
    print("\n--- E. within-group dispersion (training side, <= 2023-03)")
    side = fit_rows[["SOURCE_ROW_ID", "target_month_index"]].merge(
        processed, on="SOURCE_ROW_ID", how="left", validate="one_to_one")
    if side["BLOCK_CLEAN"].isna().any():
        raise SystemExit("A/fit rows failed to join the processed table")
    side = side[side["target_month_index"] <= INNER_CUTOFF_MONTH].copy()
    side["blockkey"] = (side["TOWN_CLEAN"].astype(str) + "|" + side["STREET_CLEAN"].astype(str)
                        + "|" + side["BLOCK_CLEAN"].astype(str))
    side["month_key"] = side["target_month_index"]
    side["area_bin"] = bin_by(side["FLOOR_AREA_SQM"], 5)
    side = side.rename(columns={"actual_rent_sgd": "actual_rent_sgd"})
    # the label comes from the contract, not the source table
    side = side.merge(fit_rows[["SOURCE_ROW_ID", "actual_rent_sgd"]],
                      on="SOURCE_ROW_ID", how="left", validate="one_to_one")
    if side["actual_rent_sgd"].isna().any():
        raise SystemExit("label merge lost rows")
    print(f"      E population {len(side):,} rows (A/fit, target <= month "
          f"{INNER_CUTOFF_MONTH})")
    print(f"      blockkey distinct: {side['blockkey'].nunique():,} "
          f"(vs BLOCK alone {side['BLOCK'].nunique() if 'BLOCK' in side else 'n/a'})")

    groupings = [
        ["blockkey", "month_key"],
        ["blockkey", "month_key", "FLAT_TYPE"],
        ["blockkey", "month_key", "FLAT_TYPE", "area_bin"],
        ["blockkey", "month_key", "FLAT_TYPE", "area_bin", "FURNISHED"],
    ]
    e_results = []
    for keys in groupings:
        for min_size in (2, 3, 5):
            stats = within_group_stats(side, keys, " + ".join(keys), min_size)
            e_results.append(stats)
            if stats["n_groups_qualifying"]:
                print(f"      {' + '.join(keys):46s} >={min_size}  "
                      f"groups {stats['n_groups_qualifying']:6d}  "
                      f"rows {stats['rows_qualifying']:6d} "
                      f"({stats['rows_qualifying_fraction']:.1%})  "
                      f"MAE {stats['within_mae_plain']:7.2f}  "
                      f"RMSE(dof) {stats['within_rmse_dof_corrected']:7.2f}")
            else:
                print(f"      {' + '.join(keys):46s} >={min_size}  no qualifying groups")

    (OUT / "dispersion_e.json").write_text(json.dumps({
        "population": f"A/fit, target_month_index <= {INNER_CUTOFF_MONTH}",
        "n_rows": int(len(side)),
        "block_key": "TOWN_CLEAN|STREET_CLEAN|BLOCK_CLEAN",
        "block_key_rationale": ("plain BLOCK is not unique per building: the same block "
                               "number recurs across streets and towns"),
        "groupings": e_results,
        "interpretation": {
            "what_it_is": "residual variation conditional on these coarse features",
            "what_it_is_not": ("NOT a proven irreducible noise floor; unobserved unit "
                               "differences remain inside groups"),
            "headroom_warning": ("never subtract this from an RMSE to claim precise "
                                 "available headroom"),
        },
    }, indent=2, default=str) + "\n")

    # ---------------------------------------------------------------- F
    print("\n--- F. inner residual structure (trial6 A/holdout)")
    saved = np.load(SEARCH / "trial6" / "inner_predictions.npz")
    predictions = pd.DataFrame({
        "SOURCE_ROW_ID": saved["row_id"].astype(np.int64),
        "predicted_rent_sgd": saved["predicted_sgd"].astype(np.float64),
        "npz_actual_sgd": saved["actual_sgd"].astype(np.float64),
        "npz_horizon": saved["horizon"].astype(np.int32),
    })
    print(f"      npz rows {len(predictions):,}")

    # explicit assertions instead of a printed delta
    if len(predictions) != A_HOLDOUT_ROWS:
        raise SystemExit(f"npz has {len(predictions)} rows, expected {A_HOLDOUT_ROWS}")
    if not predictions["SOURCE_ROW_ID"].is_unique:
        raise SystemExit("npz SOURCE_ROW_ID is not unique")

    contract_side = hold_rows[["SOURCE_ROW_ID", "actual_rent_sgd",
                               "horizon_months"]].copy()
    merged = predictions.merge(contract_side, on="SOURCE_ROW_ID", how="outer",
                               validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise SystemExit(f"join is not complete: {merged['_merge'].value_counts().to_dict()}")
    if not np.array_equal(merged["SOURCE_ROW_ID"].to_numpy(),
                          contract_side["SOURCE_ROW_ID"].to_numpy()):
        raise SystemExit("id order differs from the contract population")

    label_delta = float(np.max(np.abs(merged["npz_actual_sgd"]
                                      - merged["actual_rent_sgd"])))
    horizon_mismatch = int((merged["npz_horizon"]
                            != merged["horizon_months"]).sum())
    print(f"      joined {len(merged):,} rows one-to-one with the contract")
    print(f"      assert label max|delta| = {label_delta:.6e} (must be 0)")
    print(f"      assert horizon mismatches = {horizon_mismatch} (must be 0)")
    if label_delta != 0.0:
        raise SystemExit(f"npz labels disagree with the contract by {label_delta}")
    if horizon_mismatch != 0:
        raise SystemExit(f"{horizon_mismatch} horizon values disagree with the contract")

    merged = merged.merge(processed, on="SOURCE_ROW_ID", how="left",
                          validate="one_to_one")
    if merged["REGION"].isna().any():
        raise SystemExit("processed table failed to supply REGION for some holdout rows")
    merged["error"] = merged["predicted_rent_sgd"] - merged["actual_rent_sgd"]
    merged["area_bin"] = bin_by(merged["FLOOR_AREA_SQM"], 5)
    merged["actual_band"] = bin_by(merged["actual_rent_sgd"], 5)
    merged["predicted_band"] = bin_by(merged["predicted_rent_sgd"], 5)

    error = merged["error"].to_numpy()
    overall = {"rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
               "mae_sgd": float(np.mean(np.abs(error))),
               "bias_sgd": float(np.mean(error)), "n": int(len(error))}
    print(f"      overall RMSE {overall['rmse_sgd']:.4f}  MAE {overall['mae_sgd']:.4f}  "
          f"bias {overall['bias_sgd']:+.4f}")

    f_rows = []
    for column, label, conditioning in (
            ("TOWN", "town", "feature"),
            ("REGION", "region", "feature"),
            ("FLAT_TYPE", "flat_type", "feature"),
            ("horizon_months", "horizon", "feature"),
            ("area_bin", "area_bin", "feature"),
            ("actual_band", "price_band", "actual (post-hoc, label-conditioned)"),
            ("predicted_band", "price_band", "predicted (honest heteroscedasticity)")):
        f_rows.extend(describe_residuals(merged, column, label, conditioning))

    for row in f_rows:
        print(f"      {row['grouping']:12s} {str(row['bucket'])[:20]:20s} "
              f"n {row['n']:6d}  RMSE {row['rmse_sgd']:8.2f}  "
              f"MAE {row['mae_sgd']:8.2f}  bias {row['bias_sgd']:+8.2f}")

    # rank correlation with proper tie handling
    abs_error = pd.Series(np.abs(error))
    predicted = merged["predicted_rent_sgd"].reset_index(drop=True)
    spearman = float(abs_error.rank(method="average").corr(
        predicted.rank(method="average")))
    hetero = {
        "corr_abs_error_vs_predicted_pearson":
            float(np.corrcoef(np.abs(error), predicted.to_numpy())[0, 1]),
        "corr_abs_error_vs_predicted_spearman": spearman,
        "spearman_ties": "average ranks via pandas.rank(method='average'), not argsort",
        "predicted_band_rmse_spread": (
            max(r["rmse_sgd"] for r in f_rows
                if r["grouping"] == "price_band"
                and r["conditioning"].startswith("predicted"))
            - min(r["rmse_sgd"] for r in f_rows
                  if r["grouping"] == "price_band"
                  and r["conditioning"].startswith("predicted"))),
    }
    print(f"\n      Pearson(|error|, predicted)  = "
          f"{hetero['corr_abs_error_vs_predicted_pearson']:+.4f}")
    print(f"      Spearman(|error|, predicted) = {spearman:+.4f}")
    print(f"      predicted-band RMSE spread   = "
          f"{hetero['predicted_band_rmse_spread']:.2f} SGD")

    (OUT / "residual_f.json").write_text(json.dumps({
        "source": str((SEARCH / 'trial6' / 'inner_predictions.npz').relative_to(ROOT)),
        "n_rows": int(len(merged)),
        "join_validation": {"one_to_one": True, "rows": int(len(merged)),
                            "label_max_abs_delta": label_delta,
                            "horizon_mismatches": horizon_mismatch},
        "overall": overall,
        "groups": f_rows,
        "heteroscedasticity": hetero,
        "interpretation": {
            "actual_band": ("post-hoc and label-conditioned; structure here can be an "
                            "artifact of defining bands on the target"),
            "predicted_band": "the honest view: bands do not depend on the label",
            "no_selection": "nothing here was used to choose a model or hyperparameter",
        },
    }, indent=2, default=str) + "\n")

    print(f"\nwrote {OUT / 'dispersion_e.json'} and {OUT / 'residual_f.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cheap LightGBM scalar controls on the v3 macro arms.

Arms (identical rows, targets, horizons and v2.1 scalars; only macro blocks change):

* ``macro_none``  - v2.1 aligned scalars only
* ``macro_dbs``   - + 12 DBS scalars
* ``macro_coe``   - + 36 COE scalars (4 categories x 9)
* ``macro_both``  - + both

This uses the *scalar* aggregates, not the sequences, so it answers "is there any
signal in the macro block at all" cheaply before any recurrent encoder is built. A
null result here does not prove a sequence would fail, and a positive result does not
prove a sequence is needed — both are stated wherever these numbers are reported.

The v2.1 scalar blocks are read from the v2.1 feature arrays so the macro arms differ
from a v2.1 control only by the added columns.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import contract

try:
    import lightgbm
    from lightgbm import LGBMRegressor
    LIGHTGBM_ERROR = None
except Exception as error:  # pragma: no cover
    lightgbm = None
    LGBMRegressor = None
    LIGHTGBM_ERROR = repr(error)

LIGHTGBM_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 30, "reg_lambda": 3.0,
    "n_jobs": 2, "verbosity": -1,
}
SEED = 42


def latest_contract_dir() -> Path:
    candidates = sorted(contract.REPORT_ROOT.glob("fusion-contract-v3*/manifest.json"))
    if not candidates:
        raise SystemExit("no v3 contract exported; run the exporter first")
    return candidates[-1].parent


def load_v21_scalars(split_id: str, stage: str, row_ids: np.ndarray) -> pd.DataFrame:
    """v2.1 static+context+history scalars, aligned to ``row_ids``."""
    manifest = json.loads(contract.V2_MANIFEST.read_text())
    entry = manifest["splits"][split_id]["stages"][stage]
    arrays_path = contract.ROOT / entry["arrays"]["path"]
    arrays = np.load(arrays_path)

    v2_order = json.loads((contract.V2_CONTRACT_DIR / "manifest.json").read_text())
    feature_order = v2_order["splits"][split_id]["feature_order"]
    static_names = list(feature_order["static"])
    context_names = list(feature_order["context"])
    history_names = list(feature_order["history"])

    blocks = {
        "static": pd.DataFrame(arrays["static"], columns=static_names),
        "context": pd.DataFrame(arrays["context"], columns=context_names),
        "history": pd.DataFrame(arrays["history"], columns=history_names),
    }
    frame = pd.concat([blocks["static"], blocks["context"], blocks["history"]], axis=1)
    frame.insert(0, "SOURCE_ROW_ID", arrays["row_id"])
    if not np.array_equal(arrays["row_id"], row_ids):
        # The v3 sample table is the v2.1 table plus two columns, so order must match.
        raise AssertionError("v2.1 array row ids do not align with the v3 sample table")
    return frame.drop(columns=["SOURCE_ROW_ID"])


def load_macro_scalars(split_id: str, stage: str, macro_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = macro_dir / "features" / split_id / f"stage{stage}.npz"
    arrays = np.load(path)
    dbs_values = pd.DataFrame(arrays["dbs_scalars"], columns=contract.DBS_SCALARS)
    dbs_masks = pd.DataFrame(arrays["dbs_scalar_mask"], columns=contract.DBS_SCALARS)
    # A masked scalar is absent, not zero.
    dbs_values = dbs_values.mask(dbs_masks < 0.5)

    n_cat = len(contract.COE_CATEGORIES)
    n_scalar = len(contract.COE_SCALARS)
    flat_names = contract.coe_scalar_names()
    coe_values = pd.DataFrame(
        arrays["coe_scalars"].reshape(len(dbs_values), n_cat * n_scalar), columns=flat_names)
    coe_masks = pd.DataFrame(
        arrays["coe_scalar_mask"].reshape(len(dbs_values), n_cat * n_scalar), columns=flat_names)
    coe_values = coe_values.mask(coe_masks < 0.5)
    return dbs_values, coe_values


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "rmse_sgd": float(np.sqrt(np.mean(np.square(error)))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
        "n_rows": int(len(error)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="orig", choices=list(contract.SPLITS))
    parser.add_argument("--stage", default="B", choices=list(contract.STAGES))
    parser.add_argument("--contract-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if LGBMRegressor is None:
        print(f"BLOCKED: LightGBM unavailable: {LIGHTGBM_ERROR}")
        return 3

    root = args.contract_dir or latest_contract_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["splits"][args.split]["stages"][args.stage]
    samples = pd.read_csv(contract.ROOT / entry["artifacts"]["samples"]["path"])

    row_ids = samples["SOURCE_ROW_ID"].to_numpy()
    v21 = load_v21_scalars(args.split, args.stage, row_ids)
    dbs_values, coe_values = load_macro_scalars(args.split, args.stage, root)

    fit_mask = (samples["population"] == "fit").to_numpy()
    valid_mask = (samples["population"] == "validation").to_numpy()
    if not fit_mask.any() or not valid_mask.any():
        raise ValueError(f"{args.split}/{args.stage}: needs fit and validation rows")

    y_fit = samples.loc[fit_mask, "actual_rent_sgd"].to_numpy(dtype=float)
    y_valid = samples.loc[valid_mask, "actual_rent_sgd"].to_numpy(dtype=float)
    print(f"[{args.split}/{args.stage}] fit={fit_mask.sum():,} valid={valid_mask.sum():,} "
          f"policy={manifest['policy']}", flush=True)

    arms = {
        "macro_none": [],
        "macro_dbs": [dbs_values],
        "macro_coe": [coe_values],
        "macro_both": [dbs_values, coe_values],
    }

    out_dir = root / "tree_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, prediction_frames = [], []
    for arm, extras in arms.items():
        frame = pd.concat([v21] + extras, axis=1)
        categorical = [column for column in contract_v21_categoricals()
                       if column in frame.columns]
        for column in categorical:
            frame[column] = frame[column].astype("int64").astype("category")
        train_x = frame.loc[fit_mask].reset_index(drop=True)
        valid_x = frame.loc[valid_mask].reset_index(drop=True)

        params = dict(LIGHTGBM_PARAMS, random_state=SEED)
        model = LGBMRegressor(**params)
        started = time.perf_counter()
        model.fit(train_x, y_fit, categorical_feature=categorical)
        seconds = time.perf_counter() - started
        predicted = np.asarray(model.predict(valid_x), dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{arm}: non-finite predictions")

        metrics = {
            "arm": arm, "split_id": args.split, "stage": args.stage, "seed": SEED,
            "policy": manifest["policy"], **score(y_valid, predicted),
            "n_train": int(fit_mask.sum()), "n_validation": int(valid_mask.sum()),
            "n_features": int(frame.shape[1]),
            "n_macro_features": int(sum(extra.shape[1] for extra in extras)),
            "seconds": round(seconds, 2),
            "parameters": json.dumps(params, sort_keys=True),
            "contract_id": manifest["contract_id"],
            "source_hashes": manifest["source_hashes"],
        }
        rows.append(metrics)
        print(f"  {arm:12s} RMSE {metrics['rmse_sgd']:8.3f}  MAE {metrics['mae_sgd']:8.3f} "
              f"bias {metrics['bias_sgd']:+8.2f}  feats {metrics['n_features']}  "
              f"({seconds:.1f}s)", flush=True)

        subset = samples.loc[valid_mask].reset_index(drop=True)
        prediction_frames.append(pd.DataFrame({
            "SOURCE_ROW_ID": subset["SOURCE_ROW_ID"].to_numpy(),
            "split_id": args.split, "stage": args.stage, "policy": manifest["policy"],
            "cutoff_month": contract.split_spec(args.split)["cutoff"],
            "horizon_months": subset["horizon_months"].to_numpy(),
            "RENT_APPROVAL_DATE": subset["approval_month"].to_numpy(),
            "actual_rent_sgd": subset["actual_rent_sgd"].to_numpy(),
            "predicted_rent_sgd": predicted,
            "model": f"lgb_{arm}", "seed": SEED,
        }))

    metrics_path = out_dir / f"tree_control_metrics_stage{args.stage}.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    predictions_path = out_dir / f"tree_control_predictions_stage{args.stage}.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)

    run_manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "split": args.split, "stage": args.stage, "seed": SEED,
        "threads": LIGHTGBM_PARAMS["n_jobs"],
        "lightgbm": getattr(lightgbm, "__version__", None),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
        "contract_id": manifest["contract_id"],
        "policy": manifest["policy"],
        "source_hashes": manifest["source_hashes"],
        "metrics": str(metrics_path.relative_to(contract.ROOT)),
        "predictions": str(predictions_path.relative_to(contract.ROOT)),
        "note": ("Scalar macro aggregates only; sequences are not used here. "
                 "A null result does not prove a sequence encoder would fail."),
    }
    (out_dir / f"run_manifest_stage{args.stage}.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir.relative_to(contract.ROOT)}", flush=True)
    return 0


def contract_v21_categoricals() -> tuple[str, ...]:
    """Categorical columns of the v2.1 static block, in v2.1 order."""
    return ("TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN")


if __name__ == "__main__":
    raise SystemExit(main())

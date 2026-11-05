"""Matched LightGBM controls on the frozen Gate-1 sample table.

Two arms, identical rows, identical parameters, identical seeds:

* ``aligned_static``  - static block + context block (no historical rent)
* ``aligned_history`` - the same two blocks plus the four-level smoothed history

The point is not a new best score.  It is a comparison in which the tree and the
neural controls read *the same rows*, *the same origins* and *the same history
boundary*.  The project's original 469.21 reference used ``t-1`` history at the
target row's own date; this table uses ``origin = t - h``.  They are different
information boundaries and the old score is therefore **not** a fair comparison
to any network trained on this table.  It is preserved separately, labelled.

Run:
    .venv/bin/python -m experiments.fusion_contract_v2.tree_control --split orig
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v2 import contract, export

try:
    import lightgbm
    from lightgbm import LGBMRegressor
    LIGHTGBM_ERROR = None
except Exception as error:  # pragma: no cover - environment dependent
    lightgbm = None
    LGBMRegressor = None
    LIGHTGBM_ERROR = repr(error)

# Fixed, identical to the project's existing benchmark so capacity cannot explain
# a delta between the two arms.
LIGHTGBM_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 30, "reg_lambda": 3.0,
    "n_jobs": 2, "verbosity": -1,
}
SEED = 42
CATEGORICAL_ROLES = set(contract.CATEGORICAL_COLUMNS)


def latest_contract_dir() -> Path:
    candidates = sorted(contract.REPORT_ROOT.glob("fusion-contract-v*/manifest.json"))
    if not candidates:
        raise SystemExit("no exported contract found; run the exporter first")
    return candidates[-1].parent


def load_stage(root: Path, split_id: str, stage: str):
    manifest = json.loads((root / "manifest.json").read_text())
    stage_manifest = manifest["splits"][split_id]["stages"][stage]
    samples = pd.read_csv(contract.ROOT / stage_manifest["sample_csv"])
    arrays = np.load(contract.ROOT / stage_manifest["arrays"]["path"])
    return manifest, stage_manifest, samples, arrays


def scalar_matrix(arm: str, arrays, samples) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the scalar matrix for one arm from the exported arrays only."""
    columns = contract.arm_scalar_columns(arm)
    static_names = list(contract.static_columns())
    context_names = list(contract.context_columns())
    history_names = list(contract.history_columns())

    blocks = [pd.DataFrame(arrays["static"], columns=static_names),
              pd.DataFrame(arrays["context"], columns=context_names)]
    if contract.ROLE_HISTORY in contract.ARM_INPUTS[arm]:
        blocks.append(pd.DataFrame(arrays["history"], columns=history_names))
    frame = pd.concat(blocks, axis=1)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"arm {arm} wants columns the export does not carry: {missing}")
    frame = frame[columns]
    if len(frame) != len(samples):
        raise ValueError(f"arm {arm}: {len(frame)} feature rows vs {len(samples)} samples")

    categorical = [column for column in columns if column in CATEGORICAL_ROLES]
    for column in categorical:
        # Exported category codes are already integers; declare them as a pandas
        # categorical so LightGBM uses its native categorical handling.
        frame[column] = frame[column].astype("int64").astype("category")
    return frame, categorical


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "rmse_sgd": float(np.sqrt(np.mean(np.square(error)))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
        "n_rows": int(len(error)),
    }


def band_of(horizon: int) -> str:
    for low, high in ((1, 3), (4, 8), (9, 12), (13, 16)):
        if low <= horizon <= high:
            return f"{low}-{high}"
    return "unassigned"


def run_arm(arm: str, train_x: pd.DataFrame, train_y: np.ndarray,
            valid_x: pd.DataFrame, valid_y: np.ndarray,
            categorical: list[str]) -> tuple[np.ndarray, float, dict]:
    params = dict(LIGHTGBM_PARAMS, random_state=SEED)
    model = LGBMRegressor(**params)
    started = time.perf_counter()
    model.fit(train_x, train_y, categorical_feature=categorical)
    seconds = time.perf_counter() - started
    predicted = np.asarray(model.predict(valid_x), dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError(f"{arm} produced non-finite predictions")
    return predicted, seconds, params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=list(contract.SPLITS))
    parser.add_argument("--contract-dir", type=Path, default=None)
    parser.add_argument("--stage", default="B", choices=list(contract.STAGES))
    args = parser.parse_args(argv)

    if LGBMRegressor is None:
        print(f"BLOCKED: LightGBM unavailable: {LIGHTGBM_ERROR}")
        return 3

    root = args.contract_dir or latest_contract_dir()
    split_ids = tuple(args.split) if args.split else ("orig",)
    out_dir = root / "tree_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, predictions_all = [], []
    for split_id in split_ids:
        manifest, stage_manifest, samples, arrays = load_stage(root, split_id, args.stage)
        fit_mask = (samples["population"] == "fit").to_numpy()
        valid_mask = (samples["population"] == "validation").to_numpy()
        if not fit_mask.any() or not valid_mask.any():
            raise ValueError(f"{split_id}/{args.stage}: needs fit and validation rows")

        train_y = samples.loc[fit_mask, "actual_rent_sgd"].to_numpy(dtype=float)
        valid_y = samples.loc[valid_mask, "actual_rent_sgd"].to_numpy(dtype=float)
        print(f"\n[{split_id}/{args.stage}] fit={fit_mask.sum():,} valid={valid_mask.sum():,}",
              flush=True)

        for arm in ("aligned_static", "aligned_history"):
            matrix, categorical = scalar_matrix(arm, arrays, samples)
            train_x = matrix.loc[fit_mask].reset_index(drop=True)
            valid_x = matrix.loc[valid_mask].reset_index(drop=True)
            predicted, seconds, params = run_arm(arm, train_x, train_y, valid_x, valid_y, categorical)

            metrics = {
                "arm": arm, "split_id": split_id, "stage": args.stage, "seed": SEED,
                **score(valid_y, predicted),
                "n_train": int(fit_mask.sum()),
                "n_validation": int(valid_mask.sum()),
                "n_features": int(matrix.shape[1]),
                "feature_columns": list(matrix.columns),
                "categorical_columns": categorical,
                "seconds": round(seconds, 2),
                "parameters": json.dumps(params, sort_keys=True),
                "history_levels": list(contract.HISTORY_LEVELS)
                if contract.ROLE_HISTORY in contract.ARM_INPUTS[arm] else [],
                "contract_id": manifest["contract_id"],
                "source_sha256": manifest["source_sha256"],
                "row_id_sha256": stage_manifest["row_id_sha256"],
            }
            rows.append(metrics)
            print(f"  {arm:18s} RMSE {metrics['rmse_sgd']:8.4f}  MAE {metrics['mae_sgd']:8.4f} "
                  f"bias {metrics['bias_sgd']:+8.2f}  ({seconds:.1f}s)", flush=True)

            valid_samples = samples.loc[valid_mask].reset_index(drop=True)
            predictions_all.append(pd.DataFrame({
                contract.ID_COL: valid_samples[contract.ID_COL].to_numpy(),
                "split_id": split_id,
                "stage": args.stage,
                "cutoff_month": contract.split_spec(split_id)["cutoff"],
                "horizon_months": valid_samples["horizon_months"].to_numpy(),
                "origin_month_index": valid_samples["origin_month_index"].to_numpy(),
                "RENT_APPROVAL_DATE": valid_samples["approval_month"].to_numpy(),
                "actual_rent_sgd": valid_samples["actual_rent_sgd"].to_numpy(),
                "predicted_rent_sgd": predicted,
                "model": f"lgb_{arm}",
                "seed": SEED,
            }))

    metrics_path = out_dir / "tree_control_metrics.csv"
    existing = pd.read_csv(metrics_path) if metrics_path.exists() else None
    fresh = pd.DataFrame(rows)
    if existing is not None:
        key = ["arm", "split_id", "stage", "seed"]
        merged = pd.concat([existing, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        merged.to_csv(metrics_path, index=False)
    else:
        fresh.to_csv(metrics_path, index=False)

    predictions = pd.concat(predictions_all, ignore_index=True)
    predictions_path = out_dir / f"tree_control_predictions_stage{args.stage}.csv"
    predictions.to_csv(predictions_path, index=False)

    run_manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "stage": args.stage,
        "splits": list(split_ids),
        "seed": SEED,
        "threads": LIGHTGBM_PARAMS["n_jobs"],
        "lightgbm": getattr(lightgbm, "__version__", None),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "contract_id": manifest["contract_id"],
        "source_sha256": manifest["source_sha256"],
        "code_hashes": export.code_hash(),
        "predictions": str(predictions_path.relative_to(contract.ROOT)),
        "metrics": str(metrics_path.relative_to(contract.ROOT)),
        "note": ("Matched arms read identical rows, origins and history boundary. "
                 "The historical-feature boundary is origin = t - h, so this is NOT "
                 "directly comparable to the project's earlier 469.21 t-1 reference."),
    }
    (out_dir / f"run_manifest_stage{args.stage}.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir.relative_to(contract.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ridge and gradient-boosting controls on the v4 input protocol.

Both use the exact table `protocol.load_tables` builds: 56 numeric columns from the fit-only
scaler, one-hot categoricals, and the anchor as a feature, predicting the same residual. Nothing
here touches the Transformer or its checkpoints.

Ridge is solved in closed form with a small predeclared alpha sweep chosen on A/holdout. That
sweep is the linear model's own inner selection, so it is reported, not hidden.

LightGBM is optional: the environment may not have it, in which case the run records the
absence instead of installing anything (the plan forbids changing dependencies mid-study).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import protocol as P

OUT = Path(__file__).resolve().parents[2] / "reports" / "diagnostics_phase2_c"
# 0.01 is excluded: with this design the sqrt of the smallest curvature already
# amplifies float error, and the sweep's differences there are below its own noise
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge with an unpenalised intercept, solved by SVD rather than normal equations.

    The centred Gram matrix of this design is numerically singular (the one-hot block has
    exactly collinear columns, `cond` reported as inf), so `solve(G + aI)` overflows at small
    alpha. Solving through the SVD of the centred design is the same estimator and stays
    finite: `coef = V diag(s/(s^2+a)) U^T y`.
    """
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    centred = X - x_mean
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    # Four columns are constant in fit (hist_recency__global, hist_exact__global,
    # anchor_missing, FURNISHED_CLEAN__1), which leaves exact null directions. Forming
    # s/(s^2+alpha) on a 1e-14 singular value is numerically meaningless even though the
    # product happens to stay finite, so null directions are cut explicitly instead.
    tol = max(centred.shape) * np.finfo(np.float64).eps * s.max()
    positive = s > tol
    shrink = np.zeros_like(s)
    shrink[positive] = s[positive] / (s[positive] ** 2 + alpha)
    coef = vt[positive].T @ (shrink[positive] * (u[:, positive].T @ (y - y_mean)))
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def ridge_predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return X @ coef + intercept


def run_ridge(train: P.Table, hold: P.Table, name: str = "ridge",
              with_anchor_feature: bool = False) -> dict:
    started = time.perf_counter()
    X_fit, names = train.design_matrix(one_hot=True,
                                       with_anchor_feature=with_anchor_feature)
    X_hold, _ = hold.design_matrix(one_hot=True,
                                   with_anchor_feature=with_anchor_feature)
    y_fit = train.target.astype(np.float64)

    sweep = []
    for alpha in ALPHAS:
        coef, intercept = ridge_fit(X_fit, y_fit, alpha)
        if not np.isfinite(coef).all():
            raise FloatingPointError(f"alpha {alpha}: non-finite ridge coefficients")
        hold_metrics = P.score(hold, ridge_predict(X_hold, coef, intercept))
        fit_metrics = P.score(train, ridge_predict(X_fit, coef, intercept))
        sweep.append({"alpha": alpha, "holdout": hold_metrics, "fit": fit_metrics})

    # select on holdout only; the fit score is recorded for the over/under-fit picture
    best = min(sweep, key=lambda row: row["holdout"]["rmse_sgd"])
    coef, intercept = ridge_fit(X_fit, y_fit, best["alpha"])
    hold_residual = ridge_predict(X_hold, coef, intercept)
    predicted = (hold_residual + hold.anchor_ksgd) * P.TARGET_SCALE

    out = OUT / "stageA"
    out.mkdir(parents=True, exist_ok=True)
    P.export_predictions(out / f"{name}_holdout.csv", hold, predicted, name)
    summary = {
        "method": name, "stage": "A", "n_fit": len(train), "n_holdout": len(hold),
        "anchor_as_feature": bool(with_anchor_feature),
        "protocol_label": ("augmentation arm (anchor level as a feature); NOT the main "
                           "comparison" if with_anchor_feature
                           else "main protocol: 56 numeric + 4 categorical, no anchor input"),
        "n_features": int(X_fit.shape[1]), "feature_names": names,
        "selected_alpha": best["alpha"], "alpha_sweep": sweep,
        "holdout": best["holdout"], "fit": best["fit"],
        "anchor_only_fit": P.anchor_only(train), "anchor_only_holdout": P.anchor_only(hold),
        "parameters": int(coef.size + 1),
        "seconds": round(time.perf_counter() - started, 2),
        "protocol": train.describe(),
    }
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_lightgbm(train: P.Table, hold: P.Table, name: str = "lightgbm") -> dict:
    """A modest, predeclared budget. Records absence rather than installing anything."""
    try:
        import lightgbm as lgb
    except Exception as error:                      # noqa: BLE001 - report, never install
        record = {"method": name, "available": False, "error": repr(error),
                  "note": "lightgbm is not installed in this environment; the study forbids "
                          "changing dependencies, so this control is recorded as missing"}
        out = OUT / "stageA"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}_summary.json").write_text(json.dumps(record, indent=2))
        return record

    started = time.perf_counter()
    X_fit, names = train.design_matrix(one_hot=False)   # LGB splits codes natively
    X_hold, _ = hold.design_matrix(one_hot=False)

    scores = []
    models = {}
    for rounds in (300, 600):
        model = lgb.LGBMRegressor(
            n_estimators=rounds, learning_rate=0.05, num_leaves=63, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=2, verbose=-1, random_state=42)
        model.fit(X_fit, train.target.astype(np.float64),
                  categorical_feature=[i for i, n in enumerate(names) if n.endswith("__code")])
        hold_metrics = P.score(hold, model.predict(X_hold))
        fit_metrics = P.score(train, model.predict(X_fit))
        scores.append({"n_estimators": rounds, "holdout": hold_metrics, "fit": fit_metrics})
        models[rounds] = model

    best = min(scores, key=lambda row: row["holdout"]["rmse_sgd"])
    model = models[best["n_estimators"]]
    predicted = (model.predict(X_hold) + hold.anchor_ksgd) * P.TARGET_SCALE
    gain = sorted(zip(names, model.booster_.feature_importance("gain")),
                  key=lambda pair: -pair[1])

    out = OUT / "stageA"
    out.mkdir(parents=True, exist_ok=True)
    P.export_predictions(out / f"{name}_holdout.csv", hold, predicted, name)
    summary = {
        "method": name, "available": True, "stage": "A",
        "n_fit": len(train), "n_holdout": len(hold),
        "selected_estimators": best["n_estimators"], "budget_sweep": scores,
        "holdout": best["holdout"], "fit": best["fit"],
        "anchor_only_fit": P.anchor_only(train), "anchor_only_holdout": P.anchor_only(hold),
        "top_gain_features": [{"feature": f, "gain": float(g)} for f, g in gain[:15]],
        "seconds": round(time.perf_counter() - started, 2),
        "categorical_treatment": "raw integer codes passed as categorical_feature",
        "protocol": train.describe(),
    }
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="ridge,lightgbm")
    args = parser.parse_args(argv)

    train, hold = P.load_tables()
    checks = P.parity(train, hold)
    print(f"protocol parity: {checks}", flush=True)
    if not checks["ok"]:
        raise SystemExit("protocol parity failed; not running controls")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol_parity.json").write_text(json.dumps(
        {**checks, "fit": train.describe(), "holdout": hold.describe()}, indent=2))

    results = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"[{method}] fitting on {len(train)} rows, {train.numeric.shape[1]} numeric ...",
              flush=True)
        if method == "ridge":
            summary = run_ridge(train, hold)
        elif method == "lightgbm":
            summary = run_lightgbm(train, hold)
        else:
            raise SystemExit(f"unknown method {method!r}")
        if summary.get("available") is False:
            print(f"    UNAVAILABLE: {summary['error']}", flush=True)
        else:
            print(f"    holdout RMSE {summary['holdout']['rmse_sgd']:.4f} | MAE "
                  f"{summary['holdout']['mae_sgd']:.4f} | bias "
                  f"{summary['holdout']['bias_sgd']:.4f} | {summary['seconds']}s", flush=True)
            print(f"    anchor-only holdout RMSE "
                  f"{summary['anchor_only_holdout']['rmse_sgd']:.4f}", flush=True)
        results.append(summary)

    (OUT / "stageA_control_summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

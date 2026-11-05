"""Save the fitted boosters for the two global-bank arms, without re-running the grid.

The contrast itself already ran; this only refits the same two configurations and writes the
model files the plan asks for, so a reviewer can reload and rescore instead of trusting the
CSV. Deterministic under a fixed seed and identical inputs, so the refit must reproduce the
recorded RMSE — asserted here rather than assumed.
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
import pandas as pd  # noqa: E402

from experiments.market_dynamics_v5 import tree_arms as T  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="orig")
    parser.add_argument("--stage", default="A")
    parser.add_argument("--features", type=Path,
                        default=T.OUT_DIR.parent / "global_rent_mean_v6"
                        / "features_origA_grm.npz")
    parser.add_argument("--recorded", type=Path,
                        default=T.OUT_DIR.parent / "global_rent_mean_v6"
                        / "tree_arms" / "tree_arms_metrics.csv")
    parser.add_argument("--out", type=Path,
                        default=T.OUT_DIR.parent / "global_rent_mean_v6" / "models")
    args = parser.parse_args(argv)

    from lightgbm import LGBMRegressor

    frozen = np.load(args.features, allow_pickle=True)
    names = [str(n) for n in frozen["feature_names"]]
    scalars = T.load_v21_scalars(args.split, args.stage)
    fit, hold = scalars["fit"], scalars["holdout"]

    for label, frame, key in (("fit", fit, "fit__row_id"),
                              ("holdout", hold, "holdout__row_id")):
        index = {rid: i for i, rid in enumerate(frozen[key])}
        order = np.asarray([index[rid] for rid in frame["SOURCE_ROW_ID"]])
        block = pd.DataFrame(frozen[f"{label}__X"][order], columns=names)
        for column in block.columns:
            frame[column] = block[column].to_numpy()
        frame["__actual"] = frozen[f"{label}__actual_sgd"][order]
        if not np.allclose(frame["__actual"].to_numpy(float),
                           frozen[f"{label}__actual_sgd"][order], rtol=0, atol=0):
            raise AssertionError(f"{label}: target drifted")

    y_fit = fit["__actual"].to_numpy(float)
    y_hold = hold["__actual"].to_numpy(float)
    recorded = pd.read_csv(args.recorded)
    args.out.mkdir(parents=True, exist_ok=True)

    drop = ["SOURCE_ROW_ID", "__actual"]
    for arm in ("base", "global"):
        columns = T.arm_columns(names, arm)
        X_fit = fit.drop(columns=drop + [c for c in names if c not in columns])
        X_hold = hold.drop(columns=drop + [c for c in names if c not in columns])
        cats = [c for c in T.CATEGORICALS if c in X_fit.columns]
        report = {}
        for column in cats:
            f = X_fit[column].astype("int64")
            h = X_hold[column].astype("int64")
            cats_list = sorted(set(f.unique()))
            unseen = sorted(set(h.unique()) - set(cats_list))
            if unseen:
                code = max(cats_list) + 1
                h = h.where(~h.isin(unseen), code)
                cats_list = cats_list + [code]
                report[column] = {"codes": [int(x) for x in unseen], "mapped_to": int(code)}
            X_fit[column] = pd.Categorical(f, categories=cats_list)
            X_hold[column] = pd.Categorical(h, categories=cats_list)

        model = LGBMRegressor(**dict(T.BASE_PARAMS, n_estimators=300, random_state=T.SEED))
        model.fit(X_fit, y_fit, categorical_feature=cats)
        pred = np.asarray(model.predict(X_hold), float)
        rmse = float(np.sqrt(np.mean((pred - y_hold) ** 2)))
        expected = float(recorded.loc[recorded.arm == arm, "rmse_sgd"].iloc[0])
        path = args.out / f"lgb_{arm}.txt"
        model.booster_.save_model(str(path))
        print(f"{arm:8s} refit RMSE {rmse:.6f}  recorded {expected:.6f}  "
              f"delta {rmse - expected:+.2e}  -> {path.name}")

    manifest = {"split": args.split, "stage": args.stage,
                "features_npz_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
                "note": "boosters refit with the same config solely to persist the models; "
                        "the reported contrast came from the original run"}
    (args.out / "models_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

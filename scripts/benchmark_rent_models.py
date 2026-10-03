"""Reproducible HDB monthly-rent benchmark on a chronological holdout.

Run from repository root with ``python scripts/benchmark_rent_models.py``.
The validation window matches the project's existing 2024-04 split.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/train_spatial_features.csv"
OUTPUT = ROOT / "reports/model_benchmark"
VALIDATION_START = "2024-04"
SEED = 42

CAT_FEATURES = ["TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN"]
BASE_NUMERIC = [
    "FLOOR_AREA_SQM", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN",
    "APPROVAL_MONTH_COS", "YEARS_SINCE_LEASE_COMMENCEMENT",
]
BASE_FEATURES = CAT_FEATURES + BASE_NUMERIC
SPATIAL_FEATURES = [
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "BUILDING_AGE_YEARS",
    "NEAREST_OPEN_MRT_KM", "OPEN_MRT_COUNT_500M", "OPEN_MRT_COUNT_1000M",
    "OPEN_MRT_COUNT_2000M", "NEAREST_MALL_KM", "MALL_COUNT_500M",
    "MALL_COUNT_1000M", "MALL_COUNT_2000M", "NEAREST_PRIMARY_SCHOOL_KM",
    "PRIMARY_SCHOOL_COUNT_500M", "PRIMARY_SCHOOL_COUNT_1000M",
    "PRIMARY_SCHOOL_COUNT_2000M", "DISTANCE_TO_CBD_KM",
]

CATBOOST_PARAMS = {
    "iterations": 400, "depth": 6, "learning_rate": 0.05,
    "l2_leaf_reg": 3.0, "loss_function": "RMSE", "random_seed": SEED,
    "thread_count": 4, "verbose": False, "allow_writing_files": False,
}
LIGHTGBM_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 30, "reg_lambda": 3.0, "random_state": SEED,
    "n_jobs": 4, "verbosity": -1,
}
KNN_TRAIN_CAP = 12_000


def split_by_month(frame: pd.DataFrame, start: str = VALIDATION_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    date = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    if date.isna().any():
        raise ValueError("Invalid RENT_APPROVAL_DATE values")
    fit = frame.loc[date < pd.Timestamp(start)].copy()
    valid = frame.loc[date >= pd.Timestamp(start)].copy()
    if fit.empty or valid.empty or fit["RENT_APPROVAL_DATE"].max() >= valid["RENT_APPROVAL_DATE"].min():
        raise ValueError("Empty or overlapping temporal split")
    return fit, valid


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "rmse_sgd": float(np.sqrt(np.mean(np.square(error)))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
    }


def prepared_features(fit: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All category and missing-value decisions come from fitting rows only."""
    train_x, valid_x = fit[features].copy(), valid[features].copy()
    for column in features:
        if column in CAT_FEATURES:
            train_x[column] = train_x[column].fillna("unknown").astype(str)
            valid_x[column] = valid_x[column].fillna("unknown").astype(str)
        else:
            train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
            valid_x[column] = pd.to_numeric(valid_x[column], errors="coerce")
            median = train_x[column].median()
            if not np.isfinite(median):
                raise ValueError(f"No finite training median for {column}")
            train_x[column] = train_x[column].fillna(median)
            valid_x[column] = valid_x[column].fillna(median)
    return train_x, valid_x


def lgb_categories(train_x: pd.DataFrame, valid_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit, valid = train_x.copy(), valid_x.copy()
    for column in CAT_FEATURES:
        dtype = pd.CategoricalDtype(categories=sorted(fit[column].unique()))
        fit[column] = fit[column].astype(dtype)
        valid[column] = pd.Categorical(valid[column].where(valid[column].isin(dtype.categories)), categories=dtype.categories)
    return fit, valid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_hash = sha256(SOURCE)
    frame = pd.read_csv(SOURCE)
    needed = set(BASE_FEATURES + SPATIAL_FEATURES + ["SOURCE_ROW_ID", "RENT_APPROVAL_DATE", "MONTHLY_RENT"])
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Source missing columns: {sorted(missing)}")
    fit, valid = split_by_month(frame)
    y_fit = pd.to_numeric(fit["MONTHLY_RENT"], errors="coerce")
    y_valid = pd.to_numeric(valid["MONTHLY_RENT"], errors="coerce")
    if y_fit.isna().any() or y_valid.isna().any():
        raise ValueError("Missing or invalid target")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "SOURCE_ROW_ID": valid["SOURCE_ROW_ID"].to_numpy(),
        "RENT_APPROVAL_DATE": valid["RENT_APPROVAL_DATE"].to_numpy(),
        "actual_rent_sgd": y_valid.to_numpy(),
    })
    rows = []

    def record(name: str, pred: np.ndarray, n_train: int, seconds: float, features: list[str], params: dict) -> None:
        pred = np.asarray(pred, dtype=float)
        if len(pred) != len(valid) or not np.isfinite(pred).all():
            raise ValueError(f"{name} produced invalid predictions")
        predictions[f"pred_{name}_sgd"] = pred
        result = {"model": name, **score(y_valid.to_numpy(), pred), "n_train": n_train,
                  "n_validation": len(valid), "seconds": round(seconds, 2),
                  "n_features": len(features), "feature_set": "spatial" if features == BASE_FEATURES + SPATIAL_FEATURES else "base",
                  "parameters": json.dumps(params, sort_keys=True)}
        rows.append(result)
        print(f"{name}: RMSE {result['rmse_sgd']:.2f} SGD; MAE {result['mae_sgd']:.2f} SGD ({seconds:.1f}s)", flush=True)

    started = time.perf_counter()
    dummy = DummyRegressor(strategy="mean").fit(fit[BASE_FEATURES], y_fit)
    record("train_mean", dummy.predict(valid[BASE_FEATURES]), len(fit), time.perf_counter() - started,
           BASE_FEATURES, {"strategy": "mean"})

    for feature_set, features in (("base", BASE_FEATURES), ("spatial", BASE_FEATURES + SPATIAL_FEATURES)):
        train_x, valid_x = prepared_features(fit, valid, features)
        started = time.perf_counter()
        cat = CatBoostRegressor(**CATBOOST_PARAMS)
        cat.fit(train_x, y_fit, cat_features=CAT_FEATURES)
        if feature_set == "spatial":
            cat.save_model(str(OUTPUT / "catboost_spatial.cbm"))
        record(f"catboost_{feature_set}", cat.predict(valid_x), len(fit), time.perf_counter() - started,
               features, CATBOOST_PARAMS)

        started = time.perf_counter()
        lgb_train, lgb_valid = lgb_categories(train_x, valid_x)
        lgb = LGBMRegressor(**LIGHTGBM_PARAMS)
        lgb.fit(lgb_train, y_fit, categorical_feature=CAT_FEATURES)
        if feature_set == "spatial":
            lgb.booster_.save_model(str(OUTPUT / "lightgbm_spatial.txt"))
            (OUTPUT / "spatial_preprocessing.json").write_text(json.dumps({
                "feature_order": features,
                "numeric_training_medians": {column: float(train_x[column].median()) for column in features if column not in CAT_FEATURES},
                "categorical_training_levels": {column: list(lgb_train[column].cat.categories) for column in CAT_FEATURES},
                "missing_category_fill": "unknown",
            }, indent=2, ensure_ascii=False) + "\n")
        record(f"lightgbm_{feature_set}", lgb.predict(lgb_valid), len(fit), time.perf_counter() - started,
               features, LIGHTGBM_PARAMS)

    # KNN is included as a non-tree reference. Bound its fitting set to keep the
    # complete validation evaluation tractable; this is not a full-data comparison.
    knn_fit = fit.sample(n=min(KNN_TRAIN_CAP, len(fit)), random_state=SEED)
    numeric = BASE_NUMERIC
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    knn = Pipeline([("preprocess", preprocessing),
                    ("model", KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=4))])
    started = time.perf_counter()
    knn.fit(knn_fit[BASE_FEATURES], knn_fit["MONTHLY_RENT"])
    record("knn_base_sample", knn.predict(valid[BASE_FEATURES]), len(knn_fit), time.perf_counter() - started,
           BASE_FEATURES, {"n_neighbors": 20, "weights": "distance", "random_seed": SEED,
                           "training_cap": KNN_TRAIN_CAP, "preprocessing": "train-only numeric median+standardize; categorical one-hot"})

    metrics = pd.DataFrame(rows).sort_values("rmse_sgd")
    if sha256(SOURCE) != source_hash:
        raise RuntimeError("Source data changed during benchmark")
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "validation_predictions.csv", index=False)
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": source_hash,
        "validation_start": VALIDATION_START, "training_end": fit["RENT_APPROVAL_DATE"].max(),
        "validation_end": valid["RENT_APPROVAL_DATE"].max(),
        "n_source": len(frame), "n_train": len(fit), "n_validation": len(valid),
        "target": "MONTHLY_RENT", "units": "SGD per month", "seed": SEED,
        "base_features": BASE_FEATURES, "spatial_and_building_features_added": SPATIAL_FEATURES,
        "excluded": ["FEE", "SOURCE_ROW_ID", "BLOCK", "STREET", "NEAREST_PLANNED_MRT_KM"],
        "knn_fit_rows": len(knn_fit), "knn_sampling": "uniform random training-only sample",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "catboost": catboost.__version__,
                     "lightgbm": lightgbm.__version__},
        "interpretation": "Exploratory single chronological holdout; previous project EDA examined all labeled rows.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    lookup = metrics.set_index("model")["rmse_sgd"]
    summary = f"""# 租金预测模型基准（时间留出验证）

用 `MONTHLY_RENT` 预测月租（新元），训练至 {manifest['training_end']}，验证期为 {VALIDATION_START} 至 {manifest['validation_end']}。训练 {len(fit):,} 行，验证 {len(valid):,} 行。RMSE 越低越好。

| 模型 | 验证 RMSE（SGD） | MAE（SGD） | 训练行数 |
|---|---:|---:|---:|
"""
    for row in metrics.itertuples():
        summary += f"| {row.model} | {row.rmse_sgd:.2f} | {row.mae_sgd:.2f} | {row.n_train:,} |\n"
    summary += f"""
LightGBM 加入空间及建筑特征后，RMSE 从 {lookup['lightgbm_base']:.2f} 降至 {lookup['lightgbm_spatial']:.2f}，下降 {lookup['lightgbm_base'] - lookup['lightgbm_spatial']:.2f} SGD。CatBoost 的同项下降为 {lookup['catboost_base'] - lookup['catboost_spatial']:.2f} SGD。这个对比固定了数据划分和各模型超参数；新增特征同时含位置、周边设施和建筑属性，不能单独归因于设施。

KNN 仅使用固定随机种子抽出的 {len(knn_fit):,} 条训练样本，作为轻量参考，不用于和全量训练的树模型做严格公平排名。所有模型在完整验证集评估。空间设施快照的历史可得性尚未逐期核实；这是单次探索性时间留出验证，且项目此前 EDA 已查看所有标签，不能当作完全未触碰的最终测试分数。

运行：`DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib .venv-model-benchmark/bin/python scripts/benchmark_rent_models.py`（本机 LightGBM 需 libomp；其他环境可直接用项目建模依赖运行）。模型文件与 `spatial_preprocessing.json` 位于本目录，另有 `metrics.csv`、`validation_predictions.csv`、`manifest.json`。
"""
    (OUTPUT / "README.md").write_text(summary)
    print(f"Saved {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

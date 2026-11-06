"""Model-specific preprocessing for the initial family comparison."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.pipeline.config import CATEGORICAL, CATBOOST_SETTINGS, FAMILY_SETTINGS, FEATURE_SETS


def build_transformer(numeric, categorical, *, scale_numeric=False):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
         ("numbers", Pipeline(steps), numeric)],
        sparse_threshold=0,
    )


def make_pipeline(family: str, features=None, settings=None) -> Pipeline:
    features = list(features or FEATURE_SETS["C_property_time_town"])
    if family == "ridge":
        estimator = Ridge(**{**FAMILY_SETTINGS["ridge"], **(settings or {})})
    elif family == "random_forest":
        estimator = RandomForestRegressor(**{**FAMILY_SETTINGS["random_forest"], **(settings or {})})
    else:
        raise ValueError(f"Unknown model family: {family}")
    preprocessing = build_transformer(
        [c for c in features if c not in CATEGORICAL],
        [c for c in features if c in CATEGORICAL],
        scale_numeric=family == "ridge",
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def lightgbm_inputs(x_fit: pd.DataFrame, x_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use categories learned on fitting rows; unseen levels become missing."""
    transformer = CategoryFrame([c for c in CATEGORICAL if c in x_fit])
    transformer.fit(x_fit)
    return transformer.transform(x_fit), transformer.transform(x_val)


class CategoryFrame(BaseEstimator, TransformerMixin):
    """Persist training-only category levels alongside a native tree model."""

    def __init__(self, columns, mode="lightgbm"):
        self.columns = columns
        self.mode = mode

    def fit(self, X, y=None):
        self.categories_ = {c: pd.Index(X[c].dropna().unique()) for c in self.columns}
        return self

    def transform(self, X):
        result = X.copy()
        for column in self.columns:
            if self.mode == "lightgbm":
                levels = self.categories_[column]
                known = result[column].where(result[column].isin(levels))
                result[column] = known.astype(pd.CategoricalDtype(categories=levels))
            else:
                result[column] = result[column].fillna("unknown").astype(str)
        return result


def build_model(family: str, features=None, settings=None) -> Pipeline:
    """One factory for historical experiments, validation and full refit."""
    features = list(features or FEATURE_SETS["C_property_time_town"])
    categories = [c for c in features if c in CATEGORICAL]
    if family in ("ridge", "random_forest"):
        return make_pipeline(family, features, settings)
    if family == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(**{**FAMILY_SETTINGS[family], **(settings or {})})
    elif family == "catboost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(
            **{**CATBOOST_SETTINGS, **(settings or {})}, cat_features=categories,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("preprocess", CategoryFrame(categories, family)), ("model", estimator)])

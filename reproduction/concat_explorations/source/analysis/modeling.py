"""Model-specific preprocessing for the initial family comparison."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.config import CATEGORICAL, FAMILY_SETTINGS, NUMERIC


def make_pipeline(family: str) -> Pipeline:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if family == "ridge":
        numeric_steps.append(("scale", StandardScaler()))
        estimator = Ridge(**FAMILY_SETTINGS["ridge"])
    elif family == "random_forest":
        estimator = RandomForestRegressor(**FAMILY_SETTINGS["random_forest"])
    else:
        raise ValueError(f"Unknown model family: {family}")
    preprocessing = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("numbers", Pipeline(numeric_steps), NUMERIC),
        ],
        sparse_threshold=0,
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def lightgbm_inputs(x_fit: pd.DataFrame, x_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use categories learned on fitting rows; unseen levels become missing."""
    fitting = x_fit.copy()
    validation = x_val.copy()
    for column in CATEGORICAL:
        dtype = pd.CategoricalDtype(categories=pd.Index(fitting[column].dropna().unique()))
        fitting[column] = fitting[column].astype(dtype)
        known_values = validation[column].where(validation[column].isin(dtype.categories))
        validation[column] = known_values.astype(dtype)
    return fitting, validation

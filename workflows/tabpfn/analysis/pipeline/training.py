"""Shared training, time validation and persistence for four ML families."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analysis.pipeline.config import DATE, FEATURE_SETS, MODELS, TARGET, VALIDATION_START
from analysis.pipeline.data import check_output_path, temporal_split
from analysis.pipeline.models import build_model


@dataclass
class TrainerConfig:
    model_family: str = "lightgbm"
    feature_set: str = "C_property_time_town"
    validation_start: str = VALIDATION_START
    fit_window_months: int | None = None
    split_strategy: str = "time"
    test_size: float = 0.2
    random_state: int = 42
    model_settings: dict = field(default_factory=dict)


class HDBTrainer:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()
        self.model = None

    @property
    def features(self):
        return list(FEATURE_SETS[self.config.feature_set])

    def split_frame(self, frame: pd.DataFrame):
        return temporal_split(frame, self.config.validation_start, self.config.fit_window_months)

    def split_data(self, X: pd.DataFrame, y: pd.Series, dates=None):
        """Time split by default; random split is an explicit research option."""
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must align")
        if self.config.split_strategy == "random":
            return train_test_split(X, y, test_size=self.config.test_size,
                                    random_state=self.config.random_state)
        if self.config.split_strategy != "time":
            raise ValueError("split_strategy must be 'time' or 'random'")
        if dates is None:
            if DATE not in X:
                raise ValueError("Time split requires dates or RENT_APPROVAL_DATE")
            dates = X[DATE]
        if len(dates) != len(X):
            raise ValueError("Date and feature row counts differ")
        if isinstance(dates, pd.Series) and not dates.index.equals(X.index):
            raise ValueError("Dates and features must have matching indices")
        positions = pd.DataFrame({DATE: list(dates), "_position": range(len(X))})
        fitting, validation = self.split_frame(positions)
        fi, vi = fitting["_position"].to_numpy(), validation["_position"].to_numpy()
        return X.iloc[fi].copy(), X.iloc[vi].copy(), y.iloc[fi].copy(), y.iloc[vi].copy()

    def default_model(self):
        return build_model(self.config.model_family, self.features,
                           self.config.model_settings).named_steps["model"]

    def build_pipeline(self, transformer=None, model=None) -> Pipeline:
        if transformer is not None:
            estimator = model if model is not None else self.default_model()
            return Pipeline([("preprocessor", transformer), ("model", estimator)])
        if model is not None:
            raise ValueError("A custom model requires a custom transformer")
        return build_model(self.config.model_family, self.features, self.config.model_settings)

    def train(self, pipeline, X_train, y_train):
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        return pipeline

    def fit_frame(self, frame: pd.DataFrame):
        return self.train(self.build_pipeline(), frame[self.features], frame[TARGET])

    def save_model(self, model, path: str | Path = MODELS / "baseline_submissions" / "hdb_rent_model.joblib"):
        path = check_output_path(Path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    @staticmethod
    def load_model(path: str | Path):
        return joblib.load(path)

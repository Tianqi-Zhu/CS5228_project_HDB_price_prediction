from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TrainerConfig:
    test_size: float = 0.2
    random_state: int = 42


class HDBTrainer:
    """Train regression models for HDB rent prediction."""

    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()
        self.model = None

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """Create train/test split."""
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

    def default_model(self) -> RegressorMixin:
        """Return a reasonable baseline model."""
        return RandomForestRegressor(
            n_estimators=300,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def build_pipeline(self, transformer, model=None) -> Pipeline:
        """Combine preprocessing and model into one sklearn pipeline."""
        model = model or self.default_model()

        return Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("model", model),
            ]
        )

    def train(self, pipeline: Pipeline, X_train, y_train) -> Pipeline:
        """Fit model pipeline."""
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        return pipeline

    def save_model(
        self,
        model: Pipeline,
        path: str | Path = "models/hdb_rent_model.joblib",
    ):
        """Persist trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    @staticmethod
    def load_model(path: str | Path):
        """Load persisted model."""
        return joblib.load(path)

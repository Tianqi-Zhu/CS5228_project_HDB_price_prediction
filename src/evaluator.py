from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class HDBEvaluator:
    """Evaluate HDB rental regression models."""

    def evaluate(self, model, X_test, y_test) -> dict:
        """Compute common regression metrics."""
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        return {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": rmse,
            "r2": r2_score(y_test, predictions),
        }

    def prediction_frame(self, model, X_test, y_test) -> pd.DataFrame:
        """Return actual and predicted rents for inspection."""
        predictions = model.predict(X_test)

        return pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "predicted": predictions,
                "residual": y_test.to_numpy() - predictions,
            },
            index=y_test.index,
        )

    def print_report(self, metrics: dict):
        """Print a concise evaluation report."""
        print("Evaluation Metrics")
        print("-" * 30)

        for name, value in metrics.items():
            print(f"{name.upper():<8}: {value:,.4f}")

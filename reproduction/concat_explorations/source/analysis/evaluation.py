"""Common rental prediction metrics and group-level error summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
    }


def grouped_errors(frame: pd.DataFrame, by: str, model: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(by, sort=True):
        row = {"model": model, "group": str(group), "n": len(part)}
        row.update(metrics(part["actual"].to_numpy(), part["predicted"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)

"""Train the selected historical CatBoost model at a strict February 2025 origin."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import DATA
from analysis.data import sha256
from analysis.experiments.dynamic_market_index import FEATURES, OUT, fit_tree
from analysis.experiments.spatiotemporal_hybrid import (
    MONTH, historical_features, model_inputs, spatial_frame,
)


CUTOFF = pd.Period("2025-03", freq="M")


def main() -> None:
    selected = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    if selected["selected_row_model"] != "catboost":
        raise ValueError("The selected row model changed; review before making predictions")
    train_path, test_path = DATA / "train.csv", DATA / "test.csv"
    train = spatial_frame(pd.read_csv(train_path))
    test = spatial_frame(pd.read_csv(test_path))
    fitting = train[train[MONTH] < CUTOFF].copy()
    if fitting.empty or not (test[MONTH] >= CUTOFF).all():
        raise AssertionError("Unexpected fitting or test months")
    if fitting[MONTH].max() >= test[MONTH].min():
        raise AssertionError("Fitting labels overlap the first test month")
    fit_history, test_history = historical_features(fitting, test)
    x_fit, x_test = model_inputs(fitting.join(fit_history), test.join(test_history), FEATURES)
    predicted, seconds = fit_tree("catboost", fitting, test, x_fit, x_test)
    if len(predicted) != len(test) or not np.isfinite(predicted).all():
        raise AssertionError("Incomplete or nonfinite test predictions")
    submission = pd.DataFrame({"Id": np.arange(len(test)), "Predicted": predicted})
    submission.to_csv(OUT / "submission_strict_feb2025.csv", index=False, float_format="%.4f")
    details = {
        "model": "full-history CatBoost without index correction or recent-window ablation",
        "cutoff_exclusive": str(CUTOFF), "fitting_rows": len(fitting),
        "test_rows": len(test), "fit_predict_seconds": seconds,
        "source_sha256": {"train.csv": sha256(train_path), "test.csv": sha256(test_path)},
        "prediction_range_sgd": [float(predicted.min()), float(predicted.max())],
        "note": "All test months use one fixed origin. March 2025 labeled train rows are excluded.",
    }
    (OUT / "submission_strict_feb2025.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Saved {len(test)} predictions from {len(fitting)} fitting rows in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()

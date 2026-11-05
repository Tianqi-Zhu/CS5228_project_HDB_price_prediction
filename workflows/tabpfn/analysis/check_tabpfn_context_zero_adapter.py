"""Verify that zero DiagScale updates reproduce frozen TabPFN predictions.

Uses a small, identical saved-feature context and a fixed one-view seed so the
check is inexpensive and independent of the larger experiment schedule.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
from analysis.compare_tabpfn_diagscale_80_20 import BASE
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.tune_tabpfn_diagscale_80_20 import SavedDiagScaleRegressor


def main() -> None:
    bundle = joblib.load(BASE)
    context_x = bundle["X_train"].iloc[:2048]
    context_y = (np.asarray(bundle["y_train"], float)
                 - np.asarray(bundle["train_baseline"], float))[:2048]
    query_x = bundle["X_query"].iloc[:48]
    kwargs = dict(model_path=str(CHECKPOINT), device="cuda", n_estimators=1,
                  categorical_features_indices=category_positions(context_x),
                  auto_scale_n_estimators=False, random_state=42,
                  fit_mode="fit_with_cache", keep_cache_on_device=True,
                  inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1)
    frozen = TabPFNRegressor.create_default_for_version(version=ModelVersion.V3_5_FAST,
                                                        **kwargs)
    frozen.fit(context_x, context_y)
    reference = frozen.predict(query_x)
    del frozen
    gc.collect()
    torch.cuda.empty_cache()

    adapter = torch.load(ROOT / ".cache/tabpfn_diagscale_categorical_alignment_80_20/adapter.pt",
                         map_location="cpu", weights_only=True)
    zero = {key: torch.zeros_like(value) for key, value in adapter.items()}
    adapter_path = ROOT / ".cache/tabpfn_context_sampling/zero_adapter.pt"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(zero, adapter_path)
    unadapted = SavedDiagScaleRegressor.create_default_for_version(
        version=ModelVersion.V3_5_FAST, adapter_path=str(adapter_path), **kwargs)
    unadapted.fit(context_x, context_y)
    prediction = unadapted.predict(query_x)
    difference = np.asarray(prediction, float) - np.asarray(reference, float)
    result = {"context_rows": len(context_x), "query_rows": len(query_x),
              "max_absolute_residual_difference": float(np.abs(difference).max()),
              "rmse_residual_difference": float(np.sqrt(np.mean(difference**2))),
              "seed": 42, "inference_views": 1}
    output = REPORTS / "tabpfn_context_sampling/zero_adapter_identity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    if result["max_absolute_residual_difference"] > 1e-3:
        raise AssertionError(f"Zero adapter changed frozen predictions: {result}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Save ready-to-fit 80-column TabPFN bundles with documented preprocessing.

The existing fraction features are normalized by valid mapped buffer area.
No extra external scaler is appropriate for the frozen TabPFN pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from safetensors import safe_open

from analysis.config import ROOT
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable, write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.compare_worldcover_tree_cover import prepare, parent_paths, VARIANTS
from analysis.prepare_worldcover_tree_cover import locations
from analysis.experiments.worldcover_training_data import append_tree_features, TREE_COLUMNS, TREE_FRACTIONS, PREPROCESSING


def materialize():
    prepare()
    cfg, out, cache = locations()
    _, parent_path = parent_paths(cfg)
    source = cache / "features.joblib"
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
    signatures = {"parent_sha256": sha256(parent_path), "features_sha256": sha256(source),
                  "checkpoint_sha256": sha256(CHECKPOINT),
                  "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "analysis/experiments/worldcover_training_data.py")},
                  "preprocessing": PREPROCESSING,
                  "checkpoint_inference_config": json.loads(metadata["inference_config"])}
    immutable(out / "training_dataset_settings.json", signatures)
    manifest_path = out / "training_dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings_sha256"] != sha256(out / "training_dataset_settings.json"):
            raise ValueError("Training dataset settings changed")
        for entry in manifest["bundles"].values():
            if sha256(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("Materialized training data changed")
        print("Reused verified 80-feature training bundles", flush=True)
        return manifest
    parent, additions = joblib.load(parent_path), joblib.load(source)
    outputs, summaries = {}, []
    for stage, entry in parent.items():
        original = entry["bundle"]
        bundle = dict(original)
        for split in ("train", "query"):
            key = f"X_{split}"
            bundle[key] = append_tree_features(original[key], additions[stage][split])
            # Exactly the matrices the existing frozen runner passes to fit/predict.
            runner_matrix = original[key].join(additions[stage][split][VARIANTS["tree_100m_300m"]])
            pd.testing.assert_frame_equal(bundle[key], runner_matrix)
        path = cache / f"training_{stage}_80.joblib"
        joblib.dump(bundle, path, compress=3)
        restored = joblib.load(path)
        for key, value in bundle.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(restored[key], value)
            elif isinstance(value, pd.Series):
                pd.testing.assert_series_equal(restored[key], value)
            else:
                np.testing.assert_array_equal(restored[key], value)
        train = bundle["X_train"]
        for name in TREE_COLUMNS:
            series = train[name]
            summaries.append({"stage": stage, "feature": name, "dtype": str(series.dtype),
                              "training_rows": len(series), "observed_rows": int(series.notna().sum()),
                              "missing_rows": int(series.isna().sum()), "minimum": float(series.min()), "maximum": float(series.max()),
                              "training_mean": float(series.mean()), "training_std_ddof0": float(series.std(ddof=0)),
                              "external_transform": "identity"})
        outputs[stage] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                          "train_shape": list(train.shape), "query_shape": list(bundle["X_query"].shape),
                          "original_columns_preserved": True, "runner_matrices_identical": True,
                          "baseline_categorical_columns": list(train.iloc[:, :77].select_dtypes("category").columns),
                          "target_and_temporal_metadata_preserved": True}
        print(f"Saved {stage}: train {train.shape}; query {bundle['X_query'].shape}", flush=True)
    pd.DataFrame(summaries).to_csv(out / "training_feature_statistics.csv", index=False)
    manifest = {"settings_sha256": sha256(out / "training_dataset_settings.json"), "bundles": outputs,
                "added_columns": TREE_COLUMNS, "continuous_columns": TREE_FRACTIONS,
                "external_preprocessing_fitted": False,
                "target_policy": "Unchanged SGD rent labels; frozen runner fits y_train - train_baseline and restores query_baseline",
                "test_predictions_generated": False}
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    materialize()

"""Regression checks for data preservation, time boundaries and saved models."""

import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.pipeline.config import DATA, DATE, FEATURE_SETS, TARGET
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.data import HDBPreprocessor, check_output_path, protect_sources, temporal_split
from analysis.pipeline.models import build_model
from analysis.pipeline.spatial import nearest_distances, primary_schools
from analysis.pipeline.training import HDBTrainer
from analysis.pipeline.workflow import submission_frame


def sample():
    return pd.DataFrame({
        DATE: ["2023-01", "2023-12", "2024-03", "2024-04", "2025-03"],
        "TOWN": ["Bishan"] * 4 + ["unseen town"],
        "BLOCK": ["300"] * 5, "STREET": [" Example Street "] * 5,
        "FLAT_TYPE": ["4 room", "4-room", "3 room", "4-room", "4 room"],
        "FLAT_MODEL": ["model a"] * 4 + ["unseen model"],
        "FLOOR_AREA_SQM": [90., 100., 70., 100., 90.],
        "FURNISHED": ["yes"] * 5, "LEASE_COMMENCE_DATE": [1984] * 5,
        "FEE": [0] * 5, TARGET: [2000., 2400., 2200., 2800., 3000.],
    }, index=[8, 3, 7, 2, 1])


class PipelineChecks(unittest.TestCase):
    def test_duplicate_rows_and_index_preserved(self):
        raw = pd.concat([sample(), sample().iloc[[0]]])
        original = raw.copy(deep=True)
        prepared = HDBPreprocessor().transform(raw)
        self.assertEqual(len(raw), len(prepared))
        self.assertTrue(raw.index.equals(prepared.index))
        pd.testing.assert_frame_equal(original, raw)
        self.assertEqual(prepared["FLAT_TYPE_CLEAN"].iloc[0], prepared["FLAT_TYPE_CLEAN"].iloc[1])

    def test_time_boundary_and_recent_window(self):
        prepared = HDBPreprocessor().transform(sample())
        fitting, validation = temporal_split(prepared, "2024-04")
        self.assertEqual(fitting[DATE].max(), "2024-03")
        self.assertEqual(validation[DATE].min(), "2024-04")
        recent, recent_val = temporal_split(prepared, "2024-04", 4)
        self.assertEqual(recent[DATE].tolist(), ["2023-12", "2024-03"])
        pd.testing.assert_frame_equal(validation, recent_val)
        with self.assertRaises(ValueError):
            temporal_split(prepared, "2024-04", 0)
        with self.assertRaises(ValueError):
            temporal_split(prepared, "2026-01")

    def test_negative_lease_age_is_missing(self):
        raw = sample()
        raw.loc[8, "LEASE_COMMENCE_DATE"] = 2024
        result = HDBPreprocessor().transform(raw)
        self.assertTrue(pd.isna(result.loc[8, "FLAT_AGE_YEARS"]))
        self.assertEqual(raw.loc[8, "LEASE_COMMENCE_DATE"], 2024)

    def test_relu_preserves_source_branch_lease_missing_feature(self):
        from analysis.experiments.relu_mlp import feature_frame

        raw = sample()
        raw.loc[8, "LEASE_COMMENCE_DATE"] = 2024
        result = feature_frame(raw)
        self.assertTrue(pd.isna(result.loc[8, "LEASE_COMMENCE_DATE"]))
        self.assertTrue(pd.isna(result.loc[8, "FLAT_AGE_YEARS"]))
        self.assertEqual(result.loc[8, "LEASE_MISSING"], 1.0)

    def test_models_handle_unseen_categories_and_roundtrip(self):
        prepared = HDBPreprocessor().transform(sample())
        fitting, validation = temporal_split(prepared)
        columns = FEATURE_SETS["C_property_time_town"]
        overrides = {
            "ridge": {}, "random_forest": {"n_estimators": 5, "min_samples_leaf": 1},
            "catboost": {"iterations": 5, "depth": 2},
            "lightgbm": {"n_estimators": 5, "min_child_samples": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            for family, settings in overrides.items():
                with self.subTest(model=family):
                    model = build_model(family, columns, settings)
                    model.fit(fitting[columns], fitting[TARGET])
                    predictions = model.predict(validation[columns])
                    self.assertTrue(np.isfinite(predictions).all())
                    self.assertEqual(len(predictions), len(validation))
                    if family == "lightgbm":
                        levels = model.named_steps["preprocess"].categories_["TOWN_CLEAN"]
                        self.assertNotIn("unseen town", levels)
                        transformed = model.named_steps["preprocess"].transform(validation[columns])
                        self.assertTrue(pd.isna(transformed["TOWN_CLEAN"].iloc[-1]))
                    artifact = Path(directory) / f"{family}.joblib"
                    HDBTrainer().save_model(model, artifact)
                    reloaded = HDBTrainer.load_model(artifact)
                    np.testing.assert_allclose(predictions, reloaded.predict(validation[columns]))

    def test_submission_uses_template_order_and_rejects_invalid_predictions(self):
        template = pd.DataFrame({"Id": [42, 9, 17], "Predicted": [0., 0., 0.]})
        result = submission_frame(template, [1000., 2000., 3000.], 3)
        self.assertEqual(result.Id.tolist(), [42, 9, 17])
        self.assertEqual(result.Predicted.tolist(), [1000., 2000., 3000.])
        self.assertEqual(template.Predicted.tolist(), [0., 0., 0.])
        with self.assertRaises(ValueError):
            submission_frame(template, [1., 2.], 3)
        with self.assertRaises(ValueError):
            submission_frame(template, [1., np.nan, 3.], 3)

    def test_source_protection_and_output_guard(self):
        for path in [DATA / "train.csv", DATA / "new-output", Path("analysis/main.py"), Path(".git/test")]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_output_path(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(AssertionError):
                with protect_sources([path]):
                    path.write_text("changed", encoding="utf-8")

    def test_spatial_missing_coordinates_and_explicit_primary_population(self):
        points = pd.DataFrame({"LATITUDE": [1.3, np.nan], "LONGITUDE": [103.8, 103.8]})
        sites = points.iloc[[0]]
        values = nearest_distances(points, sites)
        self.assertAlmostEqual(values[0], 0.)
        self.assertTrue(np.isnan(values[1]))
        schools = pd.DataFrame({"MAINLEVEL_CODE": ["primary", "p1_s6", "secondary"]})
        self.assertEqual(len(primary_schools(schools, include_mixed=False)), 1)
        self.assertEqual(len(primary_schools(schools)), 2)

    def test_auxiliary_features_preserve_rows_and_target_encodings_do_not_use_validation_target(self):
        from analysis.experiments.deep_learning import add_target_encodings
        from analysis.experiments.historical_priors import hierarchical_prediction

        raw = sample()
        prepared = build_auxiliary_features(raw)
        self.assertEqual(len(prepared), len(raw))
        self.assertEqual(prepared[DATE].tolist(), raw[DATE].tolist())
        fitting, validation = temporal_split(prepared)
        _, encoded, columns = add_target_encodings(fitting, validation, window_months=18)
        changed = validation.copy()
        changed[TARGET] += 10000
        _, encoded_changed, changed_columns = add_target_encodings(fitting, changed, window_months=18)
        self.assertEqual(columns, changed_columns)
        pd.testing.assert_frame_equal(encoded[columns], encoded_changed[columns])
        self.assertTrue(np.isfinite(encoded[columns].to_numpy(float)).all())
        prior, _ = hierarchical_prediction(
            fitting, validation, base_method="constant", base_months=1,
            premium_window=24, alpha_scale=1.0,
        )
        changed_prior, _ = hierarchical_prediction(
            fitting, changed, base_method="constant", base_months=1,
            premium_window=24, alpha_scale=1.0,
        )
        np.testing.assert_allclose(prior, changed_prior)

    def test_public_modules_import(self):
        modules = [
            "analysis.main",
            "analysis.eda.run",
            "analysis.eda.overview",
            "analysis.eda.pdf_report",
            "analysis.eda.audits.main_data",
            "analysis.eda.audits.auxiliary",
            "analysis.experiments.baselines.baseline",
            "analysis.experiments.compare_models",
            "analysis.experiments.deep_learning",
            "analysis.experiments.auxiliary_boosting",
            "analysis.experiments.auxiliary_catboost",
            "analysis.experiments.historical_priors",
            "analysis.experiments.summarize_deep_learning",
            "analysis.experiments.baselines.auxiliary_ablation",
            "analysis.experiments.baselines.economic_feature_ablation",
            "analysis.experiments.baselines.spatiotemporal_hybrid",
            "analysis.experiments.neural_tabm",
            "analysis.experiments.relu_mlp",
            "analysis.verification",
            "analysis.studies.economic_signals",
            "analysis.studies.group_temporal_panel",
            "analysis.studies.temporal_correlations",
            "analysis.studies.spatial.run",
            "analysis.studies.spatial.overview",
            "analysis.studies.spatial.mrt_fixed_effects",
            "analysis.studies.spatial.mrt_matching",
            "analysis.studies.spatial.amenities_fixed_effects",
        ]
        for module in modules:
            with self.subTest(module=module):
                self.assertTrue(callable(importlib.import_module(module).main))

    def test_saved_result_consistency(self):
        from analysis.pipeline.config import EXPERIMENT_REPORTS
        if not EXPERIMENT_REPORTS.exists():
            self.skipTest('Local saved-result integration check: prepare experiment outputs first; see preflight --profile reports.')
        from analysis.verification import audit_results
        result = audit_results()
        self.assertTrue(result["ok"], result)

    def test_single_analysis_package(self):
        self.assertFalse(Path("src").exists())
        self.assertFalse(Path("scripts").exists())


if __name__ == "__main__":
    unittest.main()

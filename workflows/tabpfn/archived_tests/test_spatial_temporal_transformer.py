"""CPU checks for information cutoffs, ablations, replay and GPU queueing."""
import io
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import (
    EXPERIMENTS, MACRO_CHANNELS, RENT_CHANNELS, SEQUENCE_CHANNELS,
    FeaturePreprocessor, RentalTransformer, SequenceBank, SequencePreprocessor,
    episode_origins, forecast_context, macro_panel, rebuild_histories,
)
from analysis.run_spatial_temporal_transformer import select_epochs


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        months = pd.period_range("2021-01", "2024-12", freq="M")
        self.frame = pd.DataFrame({"_MONTH": months.repeat(2),
                                   "FLAT_TYPE_CLEAN": ["3-room", "4-room"]*len(months),
                                   "PLANNING_AREA": ["a"]*2*len(months)})
        self.adjusted = pd.Series(2000+np.arange(len(self.frame)), index=self.frame.index)
        self.macro = pd.DataFrame({c: np.arange(len(months), dtype=float)+i
                                   for i, c in enumerate(MACRO_CHANNELS)}, index=months)
        self.property = pd.DataFrame({"a": [1., 3., np.nan, 5.],
                                      "LATITUDE": [1.3, 1.31, 1.32, 1.33],
                                      "LONGITUDE": [103.8, 103.81, 103.82, 103.83],
                                      "cat": pd.Categorical(["x", "y", "x", "y"])})

    def test_episode_origins_are_earlier_and_cover_sixteen_horizons(self):
        months = pd.PeriodIndex(["2024-05"]*1024, freq="M")
        origins = episode_origins(months, np.arange(1024))
        self.assertEqual(set(months.asi8-origins.asi8), set(range(1, 17)))
        self.assertTrue((origins < months).all())
        self.assertTrue(episode_origins(["2021-01"], [0]).isna().all())

    def test_future_rent_and_macro_changes_cannot_change_sequences(self):
        origins = pd.PeriodIndex(["2023-03", "2022-06"], freq="M")
        query = self.frame.iloc[:2]
        bank = SequenceBank(self.frame, self.adjusted, self.macro)
        expected = bank.sequences(query, origins)
        rents = self.adjusted.copy()
        rents[self.frame._MONTH > origins.max()] *= 100
        macro = self.macro.copy()
        macro.loc[macro.index >= origins.max()] *= 100
        changed = SequenceBank(self.frame, rents, macro).sequences(query, origins)
        np.testing.assert_array_equal(expected, changed)

    def test_macro_publication_buffer_is_calendar_based(self):
        query = self.frame.iloc[:2]
        origin = pd.PeriodIndex(["2023-03"]*2, freq="M")
        seq = SequenceBank(self.frame, self.adjusted, self.macro).sequences(query, origin)
        np.testing.assert_array_equal(seq[0, -1, len(RENT_CHANNELS):], self.macro.loc["2023-02"].to_numpy())
        earlier = SequenceBank(self.frame, self.adjusted, self.macro).sequences(query, origin-1)
        np.testing.assert_array_equal(earlier[0, -1, len(RENT_CHANNELS):], self.macro.loc["2023-01"].to_numpy())

    def test_reference_warmup_and_cold_history_are_masked(self):
        origins = pd.PeriodIndex(["2021-11", pd.NaT], freq="M")
        seq = SequenceBank(self.frame, self.adjusted, self.macro).sequences(self.frame.iloc[:2], origins)
        self.assertTrue(np.isnan(seq[:, :, :len(RENT_CHANNELS)]).all())
        self.assertTrue(np.isnan(seq[1]).all())

    def test_property_scaler_unknown_categories_and_coordinate_shape(self):
        pre = FeaturePreprocessor().fit(self.property)
        mean = pre.mean.copy()
        query = self.property.copy()
        query["a"] = 1000.
        query["cat"] = pd.Categorical(["new"]*4)
        numeric, missing, category, spatial = pre.transform(query)
        np.testing.assert_array_equal(pre.mean, mean)
        self.assertTrue((category == 0).all())
        self.assertEqual(spatial.shape, (4, 30))
        self.assertTrue(np.isfinite(numeric).all())
        with self.assertRaises(ValueError):
            pre.transform(query[list(reversed(query.columns))])

    def test_auxiliary_ablations_zero_values_and_presence(self):
        values = np.random.default_rng(4).normal(size=(4, 12, len(SEQUENCE_CHANNELS))).astype(np.float32)
        pre = SequencePreprocessor().fit(values)
        rent = pre.transform(values, "rent")
        coe = pre.transform(values, "rent_coe")
        n = len(SEQUENCE_CHANNELS)
        self.assertTrue((rent[:, :, len(RENT_CHANNELS):n] == 0).all())
        self.assertTrue((rent[:, :, n+len(RENT_CHANNELS):] == 0).all())
        self.assertTrue((coe[:, :, n-1] == 0).all())
        self.assertTrue((coe[:, :, -1] == 0).all())
        np.testing.assert_array_equal(rent[:, :, :len(RENT_CHANNELS)], coe[:, :, :len(RENT_CHANNELS)])

    def test_sequence_scaler_does_not_fit_queries(self):
        fit = np.arange(4*12*len(SEQUENCE_CHANNELS), dtype=np.float32).reshape(4, 12, -1)
        pre = SequencePreprocessor().fit(fit)
        mean = pre.mean.copy()
        pre.transform(fit*100, "rent_coe_stock")
        np.testing.assert_array_equal(mean, pre.mean)

    def test_all_model_processors_backward_and_checkpoint_replay(self):
        pre = FeaturePreprocessor().fit(self.property)
        prop = [torch.from_numpy(x) for x in pre.transform(self.property)]
        for name in ("T0", "T1", "T2", "T3", "C1", "C3"):
            spec = EXPERIMENTS[name]
            seq = torch.randn(4, spec.window, 2*len(SEQUENCE_CHANNELS))
            context = torch.zeros(4, 4)
            torch.manual_seed(42)
            model = RentalTransformer(prop[0].shape[1], pre.cardinalities, spec)
            pred = model(*prop, seq, context)
            pred.square().mean().backward()
            self.assertTrue(torch.isfinite(model.numeric_weight.grad).all())
            model.eval()
            with torch.no_grad():
                expected = model(*prop, seq, context)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            clone = RentalTransformer(prop[0].shape[1], pre.cardinalities, spec)
            clone.load_state_dict(torch.load(stream, weights_only=True))
            clone.eval()
            with torch.no_grad():
                np.testing.assert_array_equal(expected.numpy(), clone(*prop, seq, context).numpy())

    def test_real_horizon_is_not_capped_at_ten(self):
        value = forecast_context(["2025-03"], ["2023-11"])
        self.assertEqual(value[0, 0], 1.)

    def test_sparse_type_uses_neutral_finite_baseline_and_long_horizon_plateau(self):
        from analysis.experiments.aligned_temporal_features import FEATURES, HISTORY_COLUMNS
        template = pd.DataFrame(np.nan, index=[2, 3], columns=HISTORY_COLUMNS+FEATURES)
        raw = pd.DataFrame({"_MONTH": pd.PeriodIndex(["2021-01", "2021-01", "2022-05", "2022-05"], freq="M"),
                            "MONTHLY_RENT": [2000., 2200., 3000., 3200.]})
        def temporal(blind, origin):
            self.assertTrue((blind._MONTH <= origin+10).all())
            f = pd.DataFrame(1., index=blind.index, columns=FEATURES)
            f["TEMP_LOCAL_FORECAST"] = [np.nan, 2500.]
            return f
        with patch("analysis.experiments.spatial_temporal_transformer.origin_history",
                   return_value=pd.DataFrame(1., index=[2, 3], columns=HISTORY_COLUMNS)), \
             patch("analysis.experiments.spatial_temporal_transformer.LocalTemporalModel") as model:
            model.return_value.features.side_effect = temporal
            x, baseline = rebuild_histories(template, raw, ["2021-01", "2021-01"])
        np.testing.assert_array_equal(baseline, [0., 2500.])
        np.testing.assert_array_equal(x.TEMP_FORECAST_HORIZON, [16., 16.])
        np.testing.assert_array_equal(x.TEMP_FLAT_LAST_OBSERVATION_AGE, [7., 7.])

    def test_epoch_selection_depends_only_on_earlier_best_epochs(self):
        self.assertEqual(select_epochs({"best_epoch": 2, "outer_rmse": 0},
                                       {"best_epoch": 5, "outer_rmse": 999}), 4)


class GpuGateTests(unittest.TestCase):
    def test_waits_for_two_consecutive_available_samples(self):
        samples = iter([(2, 0), (12, 0), (12, 90), (12, 0), (12, 0)])
        waits = []
        def read():
            free, util = next(samples)
            return {"total_gib": 16, "free_gib": free, "utilization_pct": util}
        state = wait_for_gpu(snapshot_fn=read, sleep_fn=waits.append, poll_seconds=1)
        self.assertEqual(state["state"], "ready")
        self.assertEqual(len(waits), 4)

    def test_telemetry_error_does_not_start_training(self):
        calls = []
        def read():
            calls.append(1)
            if len(calls) == 1:
                raise subprocess.CalledProcessError(1, "nvidia-smi")
            return {"total_gib": 16, "free_gib": 12, "utilization_pct": 0}
        state = wait_for_gpu(snapshot_fn=read, sleep_fn=lambda _: None, poll_seconds=1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(state["state"], "ready")

    def test_impossible_memory_request_fails(self):
        with self.assertRaises(ValueError):
            wait_for_gpu(min_free_gib=32, snapshot_fn=lambda: {"total_gib": 16}, sleep_fn=lambda _: None)


if __name__ == "__main__":
    unittest.main()

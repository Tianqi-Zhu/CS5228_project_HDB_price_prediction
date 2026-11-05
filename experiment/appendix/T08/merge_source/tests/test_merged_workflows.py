"""Keep TabPFN and the merged zch pipeline available in one checkout."""
import importlib
from pathlib import Path
import unittest


class MergedWorkflowTests(unittest.TestCase):
    def test_duplicate_experiment_names_have_distinct_modules(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("auxiliary_ablation", "baseline", "economic_feature_ablation", "spatiotemporal_hybrid"):
            with self.subTest(name=name):
                tabpfn = importlib.import_module(f"analysis.experiments.{name}")
                zch = importlib.import_module(f"analysis.experiments.baselines.{name}")
                self.assertEqual(Path(tabpfn.__file__), root / "analysis/experiments" / f"{name}.py")
                self.assertEqual(Path(zch.__file__), root / "analysis/experiments/baselines" / f"{name}.py")
                self.assertIsNot(tabpfn, zch)

    def test_report_directories_preserve_both_workflows(self):
        from analysis import config as tabpfn
        from analysis.pipeline import config as zch
        self.assertEqual(tabpfn.ROOT, zch.ROOT)
        self.assertEqual(tabpfn.REPORTS, tabpfn.ROOT / "reports")
        self.assertEqual(zch.EXPERIMENT_REPORTS, zch.ROOT / "output/experiment_runs")

    def test_zch_internal_imports_use_its_own_baseline_features(self):
        from analysis.experiments.baselines import auxiliary_ablation, spatiotemporal_hybrid
        self.assertIs(spatiotemporal_hybrid.AMENITIES, auxiliary_ablation.AMENITIES)
        self.assertIs(spatiotemporal_hybrid.BLOCK, auxiliary_ablation.BLOCK)


if __name__ == "__main__":
    unittest.main()

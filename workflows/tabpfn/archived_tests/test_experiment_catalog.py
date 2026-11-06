"""The experiment index must remain usable without executing training code."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.experiment_catalog import describe, list_experiments


class ExperimentCatalogTests(unittest.TestCase):
    def test_catalog_runs_with_only_the_standard_library(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "analysis.experiment_catalog", "--group", "final", "--json"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
        )
        rows = json.loads(result.stdout)
        self.assertEqual({row["module"].split(".")[-1] for row in rows},
                         {"best_ensemble_submission", "final_submission_selection"})
        self.assertTrue(all(row["executable"] for row in rows))

    def test_describing_a_module_never_executes_its_top_level_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "third_wave_example.py"
            path.write_text('"""Example."""\nraise RuntimeError("must not run")\n'
                            'if __name__ == "__main__":\n    train()\n', encoding="utf-8")
            row = describe(path)
        self.assertEqual(row["group"], "wave03")
        self.assertTrue(row["executable"])
        self.assertEqual(row["summary"], "Example.")

    def test_support_modules_are_distinguished_from_script_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "helper.py"
            path.write_text('def main():\n    pass\n', encoding="utf-8")
            row = describe(path)
        self.assertFalse(row["executable"])
        self.assertIsNone(row["command"])
        self.assertTrue(list_experiments("wave10"))


if __name__ == "__main__":
    unittest.main()

"""Reading reports must link correctly and never alter the source run data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import report_bundle


class ReportBundleTests(unittest.TestCase):
    def test_links_prefer_reading_copies_and_preserve_external_links_and_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'output/experiment_runs/round/README.md'
            target = root / 'experiment/04_history/round.md'
            metric = source.parent / 'scores.csv'
            body = '[scores](scores.csv#annual) [web](https://example.com/a) [section](#results)'
            with patch.object(report_bundle, 'ROOT', root), \
                 patch.object(report_bundle, 'RUNS', root / 'output/experiment_runs'), \
                 patch.dict(report_bundle.READING_FILES, {'03_models/scores.csv': metric}, clear=True):
                value = report_bundle.rewrite_links(body, source, target)
            self.assertIn('](../03_models/scores.csv#annual)', value)
            self.assertIn('](https://example.com/a)', value)
            self.assertIn('](#results)', value)

    def test_refresh_keeps_csv_bytes_and_source_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'output/experiment_runs/test/scores.csv'
            source.parent.mkdir(parents=True)
            raw = b'model,rmse\r\ntrained_model,441.8345348836222\r\n'
            source.write_bytes(raw)
            with patch.object(report_bundle, 'ROOT', root), \
                 patch.dict(report_bundle.READING_FILES, {'03_models/scores.csv': source}, clear=True):
                report_bundle.refresh()
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual((root / 'experiment/03_models/scores.csv').read_bytes(), raw)
            self.assertTrue((root / 'output/checks/records/reading_bundle_sources.json').exists())


if __name__ == '__main__':
    unittest.main()

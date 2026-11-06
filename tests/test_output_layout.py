"""Generated reports stay separate from frozen scientific evidence."""
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from analysis.pipeline.config import ROOT, EDA_REPORTS, STUDY_REPORTS


class OutputLayoutTests(unittest.TestCase):
    def test_default_eda_destinations_agree_and_stay_under_output(self):
        from analysis.eda import overview, run, pdf_report
        from analysis.eda.audits import auxiliary, main_data
        self.assertEqual(EDA_REPORTS, ROOT / 'output/eda')
        self.assertEqual(STUDY_REPORTS, ROOT / 'output/studies')
        self.assertEqual(run.DEFAULT_OUT, EDA_REPORTS)
        self.assertEqual(overview.DEFAULT_OUT, EDA_REPORTS)
        self.assertEqual(main_data.DEFAULT_OUT, EDA_REPORTS / 'supplementary/train_test')
        self.assertEqual(auxiliary.DEFAULT_OUT, EDA_REPORTS / 'supplementary/auxiliary')
        self.assertTrue(pdf_report.DEFAULT_OUT.is_relative_to(ROOT / 'output'))
        self.assertEqual(pdf_report.DEFAULT_EDA, ROOT / 'experiment/main/M01/eda')

    def test_default_eda_dispatch_does_not_overwrite_frozen_main(self):
        from analysis.eda import run
        overview, main_data, auxiliary = Mock(), Mock(), Mock()
        with patch.object(run, 'overview', overview), patch.object(run, 'main_data', main_data), \
             patch.object(run, 'auxiliary', auxiliary), patch.object(run, 'protect_sources', return_value=contextlib.nullcontext()):
            run.main()
        overview.main.assert_called_once_with(output_dir=ROOT / 'output/eda')
        main_data.main.assert_called_once_with(output_dir=ROOT / 'output/eda/supplementary/train_test')
        auxiliary.main.assert_called_once_with(output_dir=ROOT / 'output/eda/supplementary/auxiliary')

    def test_legacy_reading_bundle_cannot_write_into_frozen_evidence(self):
        from analysis.report_bundle import refresh
        for path in (ROOT / 'experiment', ROOT / 'data', ROOT / 'output'):
            with self.subTest(path=path), self.assertRaises(ValueError): refresh(path)

    def test_missing_legacy_sources_do_not_create_partial_reading_bundle(self):
        from analysis import report_bundle
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temp:
            destination = Path(temp) / 'generated'
            with patch.object(report_bundle, 'READING_FILES', {'missing.md': Path(temp) / 'missing.md'}):
                with self.assertRaises(FileNotFoundError): report_bundle.refresh(destination)
            self.assertFalse(destination.exists())

    def test_reading_bundle_links_follow_the_selected_output_directory(self):
        from analysis import report_bundle
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temp:
            source = Path(temp) / 'source'
            source.mkdir()
            (source / 'README.md').write_text('[metrics](metrics.csv#value)\n', encoding='utf-8')
            (source / 'metrics.csv').write_text('value\n1\n', encoding='utf-8')
            destination = Path(temp) / 'generated'
            inputs = {'reading/README.md': source / 'README.md', 'tables/metrics.csv': source / 'metrics.csv'}
            with patch.object(report_bundle, 'READING_FILES', inputs): report_bundle.refresh(destination)
            self.assertEqual((destination / 'reading/README.md').read_text('utf-8'), '[metrics](../tables/metrics.csv#value)\n')
            self.assertTrue((destination / 'tables/metrics.csv').is_file())
            self.assertTrue((destination / 'sources.json').is_file())


if __name__ == '__main__': unittest.main()

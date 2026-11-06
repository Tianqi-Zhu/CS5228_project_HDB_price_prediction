"""Integration checks for MLP on main; full training is a separate acceptance run."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'reproduction/mlp'


class MainMlpIntegrationTests(unittest.TestCase):
    def test_archived_submissions_and_dataset_identity(self):
        spec=importlib.util.spec_from_file_location('mlp_integrity',HERE/'verify.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):result=module.verify()
        self.assertEqual(len(result['submissions']),4)
        self.assertFalse(result['training_performed'])
    def test_frozen_scores_are_not_assigned_to_weighted_candidates(self):
        scores=json.loads((HERE/'artifacts/fullfit/kaggle_results.json').read_text())
        lookup={r['filename']:r['kaggle_displayed_score'] for r in scores['results']}
        self.assertEqual(lookup['submission_mlp_fullfit_cleaned.csv'],437.34455)
        self.assertEqual(lookup['submission_mlp_fullfit_uncleaned.csv'],439.38193)
        weighted=json.loads((HERE/'artifacts/recency/completion.json').read_text())
        self.assertIsNone(weighted['kaggle_score'])
    def test_isolated_runtime_uses_main_data_root(self):
        code='import sys;sys.path.insert(0,sys.argv[1]);from analysis.pipeline.config import ROOT;print(ROOT)'
        output=subprocess.check_output([sys.executable,'-c',code,str(HERE/'runtime')],cwd=ROOT,text=True)
        self.assertEqual(Path(output.strip()),ROOT)
    def test_main_cli_lists_the_integrated_commands(self):
        output=subprocess.check_output([sys.executable,'-m','analysis.main','mlp'],cwd=ROOT,text=True)
        for name in ['fullfit','recency-search','intervals','cleaning','verify','test']:
            self.assertIn(name,output)
    def test_main_pipeline_remains_the_canonical_namespace(self):
        code='import analysis.pipeline.config;print(analysis.pipeline.config.__file__)'
        output=subprocess.check_output([sys.executable,'-c',code],cwd=ROOT,text=True)
        self.assertEqual(Path(output.strip()),ROOT/'analysis/pipeline/config.py')


if __name__=='__main__':unittest.main()

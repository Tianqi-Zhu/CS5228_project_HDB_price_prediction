"""Guard help against training and ensure CLI overrides reach the runner."""
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock,patch
from analysis.main import main


class MainCliTests(unittest.TestCase):
    def test_every_help_exits_before_experiment_import(self):
        original=__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(('analysis.experiments','analysis.eda')):
                raise AssertionError('Help must not import experiment code')
            return original(name,*args,**kwargs)
        for command in ('eda','baseline','compare','ladder','common-models','planned','schemas','evidence','validation-refits','final-pair'):
            with self.subTest(command=command),patch('builtins.__import__',side_effect=guarded),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:main([command,'--help'])
                self.assertEqual(result.exception.code,0)

    def test_baseline_forwards_output_directory(self):
        fake=types.ModuleType('analysis.experiments.baseline');fake.main=Mock()
        with patch.dict(sys.modules,{'analysis.experiments.baseline':fake}):
            main(['baseline','--output-dir','output/cli-check'])
        fake.main.assert_called_once_with(output_dir=Path('output/cli-check'))

    def test_ladder_runs_only_selected_stage(self):
        fake=types.ModuleType('analysis.experiments.report_selection_trials');fake.ladder=Mock()
        with patch.dict(sys.modules,{'analysis.experiments.report_selection_trials':fake}):
            main(['ladder','--stage','june'])
        fake.ladder.assert_called_once_with('june')

    def test_unknown_training_option_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:main(['baseline','--bogus'])
        self.assertEqual(result.exception.code,2)

if __name__=='__main__':unittest.main()

import unittest
class LastPredictionTest(unittest.TestCase):
    def test_export_from_completed_checkpoint(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from pathlib import Path
        from experiments.spatiotemporal_v17.tests.test_train import population
        from experiments.spatiotemporal_v17.train import run
        from experiments.spatiotemporal_v17.last_predictions import export
        b=dict(train=population(8),val=population(4,20),numeric_names=['a','b'],categorical_names=['c'],cardinalities=[2],sequence_names=[],manifest={'test':True})
        with TemporaryDirectory() as d, patch('experiments.spatiotemporal_v17.contract.load_bank',return_value=b):
            run('F0',max_epochs=1,expected=(8,4),output=d,lr=.0001,diagnostics=True)
            export(Path(d),expected=4)
            self.assertEqual(len((Path(d)/'last_predictions.csv').read_text().splitlines()),5)

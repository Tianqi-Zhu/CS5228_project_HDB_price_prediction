import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from experiments.spatiotemporal_v17.compare import collect
class CompareTest(unittest.TestCase):
    def test_unfinished_is_not_ranked(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'F1';p.mkdir(); (p/'result.json').write_text('{"reload_verified":true,"rmse_sgd":2}')
            report=collect(Path(d),['F1','F0']); self.assertEqual(report['complete'],[]); self.assertEqual(len(report['incomplete']),2)

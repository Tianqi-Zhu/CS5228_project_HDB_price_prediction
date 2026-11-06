"""Expose the unchanged function-style cohort regression test to unittest."""
import runpy
import unittest
from pathlib import Path


class AnniversaryMergeTests(unittest.TestCase):
    def test_original_anniversary_history_regression(self):
        source = Path(__file__).with_name('test_tabpfn_anniversary_history.py')
        namespace = runpy.run_path(str(source))
        namespace['test_anniversary_history_uses_only_prior_matching_block_and_type']()


if __name__ == '__main__':
    unittest.main()

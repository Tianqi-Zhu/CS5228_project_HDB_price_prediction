"""Reject false F2 provenance and ID alignment before retrospective scoring."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.validation.final_pair import provenance_hash, rebuild_components


class FinalPairTests(unittest.TestCase):
    def test_line_ending_identity_does_not_accept_value_edits(self):
        original = b'Id,Predicted\r\n0,2000.000000\r\n'
        expected = hashlib.sha256(original).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.csv'
            path.write_bytes(original.replace(b'\r\n', b'\n'))
            self.assertEqual(provenance_hash(path, expected)['matching_representation'], 'restored_CRLF')
            path.write_bytes(path.read_bytes().replace(b'2000', b'2001'))
            with self.assertRaises(ValueError):
                provenance_hash(path, expected)

    def test_missing_duplicate_or_reordered_members_cannot_blend(self):
        sample = pd.DataFrame({'Id': [0, 1], 'Predicted': [0, 0]})
        a = pd.DataFrame({'row_id': [0, 1], 'predicted': [2000., 3000.]})
        for b in [a.iloc[::-1], a.iloc[:1], a.assign(row_id=[0, 0]), a.assign(predicted=[2000, np.nan])]:
            with self.subTest(member=b.to_dict()), self.assertRaises(ValueError):
                rebuild_components(a, b, sample)
        b = a.assign(predicted=[2200., 2800.])
        np.testing.assert_array_equal(rebuild_components(a, b, sample), [2100., 2900.])


if __name__ == '__main__':
    unittest.main()

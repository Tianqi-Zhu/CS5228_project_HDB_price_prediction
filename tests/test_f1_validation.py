"""Frozen validation weights must preserve alignment and support gating."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'reproduction/f1/validate.py'
spec = importlib.util.spec_from_file_location('f1_validation_entry', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class F1ValidationTests(unittest.TestCase):
    def test_support_specific_weights_and_global_fallback(self):
        model = {'names': ['a', 'b'], 'global_weights': [.5, .5],
                 'group_weights': {'True|False': {'weights': [1., 0.]}}}
        result = entry.blend(np.array([[2., 8.], [3., 9.]]), ['a', 'b'], [True, False], [False, True], model)
        np.testing.assert_allclose(result, [2., 6.])

    def test_reordered_or_invalid_weights_fail(self):
        model = {'names': ['a', 'b'], 'global_weights': [.5, .5], 'group_weights': {}}
        with self.assertRaises(ValueError):entry.blend(np.ones((2, 2)), ['b', 'a'], [True]*2, [False]*2, model)
        for weights in [[1., 1.], [1.1, -.1], [float('nan'), .5]]:
            with self.assertRaises(ValueError):entry.blend(np.ones((2, 2)), ['a', 'b'], [True]*2, [False]*2, model | {'global_weights': weights})


if __name__ == '__main__':unittest.main()

import unittest

import numpy as np

from analysis.experiments.recent_time_context import recent_half_positions


class RecentContextTests(unittest.TestCase):
    def test_odd_horizon_keeps_middle_month_and_whole_months(self):
        positions, policy = recent_half_positions(["2021-01", "2021-03", "2021-03", "2021-05"])
        np.testing.assert_array_equal(positions, [1, 2, 3])
        self.assertEqual(policy["cutoff_month"], "2021-03")
        self.assertEqual(policy["retained_calendar_months"], 3)

    def test_even_horizon_uses_time_instead_of_row_count(self):
        positions, policy = recent_half_positions(["2021-04", "2021-01", "2021-01", "2021-01", "2021-02", "2021-03"])
        np.testing.assert_array_equal(positions, [0, 5])
        self.assertEqual(policy["cutoff_month"], "2021-03")
        self.assertEqual(policy["retained_calendar_months"], 2)

    def test_invalid_months_rejected(self):
        for values in ([], ["2021-01", None]):
            with self.assertRaises(ValueError):
                recent_half_positions(values)


if __name__ == "__main__":
    unittest.main()

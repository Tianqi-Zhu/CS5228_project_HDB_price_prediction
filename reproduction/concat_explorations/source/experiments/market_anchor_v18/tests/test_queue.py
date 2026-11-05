import unittest
from experiments.market_anchor_v18.queue import task_list
class QueueTest(unittest.TestCase):
 def test_fixed_budget_and_controls(self):
  tasks=task_list(); self.assertEqual(len(tasks),12)
  self.assertEqual(set(x[1] for x in tasks),{'direct','residual'})
  self.assertEqual(set(x[2] for x in tasks),{'ridge','lightgbm'})

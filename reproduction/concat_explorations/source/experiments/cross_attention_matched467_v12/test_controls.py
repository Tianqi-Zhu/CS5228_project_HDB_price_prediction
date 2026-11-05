import unittest, random
import numpy as np
import torch
from experiments.cross_attention_matched467_v12 import run

class Controls(unittest.TestCase):
    def test_budget_and_schedule(self):
        self.assertEqual(run.EPOCHS, 23)
        p=torch.nn.Parameter(torch.ones(1)); o=torch.optim.AdamW([p],lr=.001)
        s=run.scheduler(o)
        self.assertEqual(s.T_max,60)
    def test_rng_snapshot_preserves_post_build_state(self):
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        torch.nn.Linear(3,4); state=run.rng_get()
        expected=(torch.rand(4),random.random(),np.random.rand())
        run.rng_set(state); actual=(torch.rand(4),random.random(),np.random.rand())
        self.assertTrue(torch.equal(expected[0],actual[0])); self.assertEqual(expected[1:],actual[1:])
    def test_fixed_order_and_no_selection(self):
        self.assertEqual([run.batch_seed(i) for i in range(1,24)],list(range(42001,42024)))
        self.assertEqual(run.epoch_range(0),range(1,24))
        self.assertEqual(run.epoch_range(22),range(23,24))

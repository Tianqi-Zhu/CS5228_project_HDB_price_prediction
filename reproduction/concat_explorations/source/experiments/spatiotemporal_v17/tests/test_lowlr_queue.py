import unittest
from experiments.spatiotemporal_v17.lowlr_queue import task_command
class LowLRTest(unittest.TestCase):
    def test_commands_change_only_learning_rate_and_architecture(self):
        c=task_command('static_1e-4'); d=task_command('cross_3e-5')
        self.assertIn('static',c); self.assertIn('0.0001',c); self.assertIn('cross',d); self.assertIn('3e-05',d)
        with self.assertRaises(ValueError): task_command('bad')

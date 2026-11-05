import unittest
from experiments.spatiotemporal_v17.followup_queue import task_command
class FollowupTest(unittest.TestCase):
    def test_task_commands_isolate_architectures(self):
        c=task_command('F0_concat'); d=task_command('F0_cross')
        self.assertIn('concat',c); self.assertIn('cross',d); self.assertNotEqual(c,d)
        with self.assertRaises(ValueError): task_command('invalid')

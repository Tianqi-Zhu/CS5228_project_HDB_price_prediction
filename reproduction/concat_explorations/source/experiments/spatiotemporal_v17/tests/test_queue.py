import unittest
from experiments.spatiotemporal_v17.queue import prior_state
class QueueTest(unittest.TestCase):
    def test_prior_state_fails_closed(self):
        self.assertEqual(prior_state('completed',False),'ready')
        self.assertEqual(prior_state('running',True),'waiting')
        self.assertEqual(prior_state('running',False),'blocked')
        self.assertEqual(prior_state('failed',False),'blocked')
        self.assertEqual(prior_state('completed',True),'waiting')
    def test_process_probe_distinguishes_absent_and_unknown(self):
        from unittest.mock import patch
        from experiments.spatiotemporal_v17.queue import pid_state, prior_state
        with patch('os.kill',side_effect=ProcessLookupError): self.assertEqual(pid_state(42),'absent')
        with patch('os.kill',side_effect=PermissionError): self.assertEqual(pid_state(42),'unknown')
        with patch('os.kill',return_value=None): self.assertEqual(pid_state(42),'present')
        self.assertEqual(prior_state('running',False,['absent','absent']),'ready_stale')
        self.assertEqual(prior_state('running',False,['absent','unknown']),'blocked')

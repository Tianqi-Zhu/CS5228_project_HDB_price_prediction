import importlib.util
from pathlib import Path
import tempfile
import unittest


class QueueGuards(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec('experiments.overnight_hdb_v14.queue')
        self.assertIsNotNone(spec, 'overnight queue is not implemented')
        from experiments.overnight_hdb_v14 import queue
        return queue

    def test_selection_only_is_mandatory(self):
        queue = self.api()
        command = ['python', '-u', '-m', 'experiments.tabfm_finetune_v13.pipeline_raw', '--selection-only']
        queue.validate_command(command)
        with self.assertRaises(ValueError):
            queue.validate_command(command[:-1])
        with self.assertRaises(ValueError):
            queue.validate_command(command + ['--final-refit'])

    def test_readiness_rejects_changed_source(self):
        queue = self.api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.py'
            path.write_text('approved code\n')
            marker = {'approved': True, 'sources': {str(path): queue.sha256(path)}}
            queue.verify_readiness(marker)
            path.write_text('changed code\n')
            with self.assertRaises(ValueError):
                queue.verify_readiness(marker)


if __name__ == '__main__':
    unittest.main()

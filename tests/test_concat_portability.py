"""Runner import and cross-process locking on the host operating system."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from analysis.experiments.concat_transformer import runtime_path


class ConcatPortabilityTests(unittest.TestCase):
    def test_worker_lock_rejects_second_process_and_releases_on_close(self):
        sys.path.insert(0, str(runtime_path()))
        from cluster.file_lock import flock, LOCK_EX, LOCK_NB
        code = "from pathlib import Path; import sys; from cluster.file_lock import flock,LOCK_EX,LOCK_NB; f=Path(sys.argv[1]).open('a+'); flock(f,LOCK_EX|LOCK_NB)"
        env = dict(os.environ, PYTHONPATH=str(runtime_path()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'worker.lock'
            with path.open('a+') as first:
                flock(first, LOCK_EX | LOCK_NB)
                locked = subprocess.run([sys.executable, '-c', code, str(path)], env=env, capture_output=True)
                self.assertNotEqual(locked.returncode, 0)
            available = subprocess.run([sys.executable, '-c', code, str(path)], env=env, capture_output=True)
            self.assertEqual(available.returncode, 0, available.stderr.decode(errors='replace'))

    def test_archived_final_train_help_can_start_on_host(self):
        result = subprocess.run([sys.executable, '-m', 'cluster.concat_final_v22.train', '--help'],
                                cwd=runtime_path(), capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertIn(b'--smoke', result.stdout)


if __name__ == '__main__':unittest.main()

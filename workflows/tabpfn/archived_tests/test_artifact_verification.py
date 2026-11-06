"""Regression checks for missing or changed exported prediction components."""
from pathlib import Path
import tempfile
import unittest

from analysis.pipeline.data import sha256
from analysis.verification import audit_component_manifest


class ComponentManifestTests(unittest.TestCase):
    def test_directory_rename_requires_the_manifest_path_to_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / 'models/old_name/predictions.npy'
            component.parent.mkdir(parents=True)
            component.write_bytes(b'saved predictions')
            manifest = [{'path': 'models/old_name/predictions.npy', 'sha256': sha256(component)}]
            self.assertTrue(audit_component_manifest(root, manifest)['ok'])
            component.parent.rename(root / 'models/new_name')
            result = audit_component_manifest(root, manifest)
            self.assertFalse(result['ok'])
            self.assertEqual(result['issues'][0]['issue'], 'missing')
            manifest[0]['path'] = 'models/new_name/predictions.npy'
            self.assertTrue(audit_component_manifest(root, manifest)['ok'])

    def test_changed_prediction_bytes_fail_the_recorded_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / 'predictions.npy'
            component.write_bytes(b'original predictions')
            manifest = [{'path': component.name, 'sha256': sha256(component)}]
            component.write_bytes(b'different predictions')
            result = audit_component_manifest(root, manifest)
            self.assertFalse(result['ok'])
            self.assertEqual(result['issues'][0]['issue'], 'hash_mismatch')


if __name__ == '__main__':
    unittest.main()

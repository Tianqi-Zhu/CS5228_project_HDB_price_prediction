"""Resource failures must be actionable without loading weights or training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.resource_preflight import DATA_FILES, ROOT, audit, project_path, sha256, version_matches


class ResourcePreflightTests(unittest.TestCase):
    def data(self, root):
        for name in DATA_FILES:
            path = root / 'data' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('field,MONTHLY_RENT\n1,1000\n' if name == 'train.csv' else 'field\n1\n', encoding='utf-8')

    def final(self, root):
        self.data(root)
        base = Path('output/submissions/components')
        def put(relative, value=b'not a deserializable model'):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            return path
        def write(relative, value):
            return put(relative, json.dumps(value).encode())
        members = []
        for number in range(10):
            relative = (base / f'history_macro_ensemble/models/member{number}/model.joblib').as_posix()
            put(relative)
            members.append({'name': f'member{number}', 'checkpoint': relative})
        write(base / 'history_macro_ensemble/summary.json', {'members': members})
        put(base / 'history_macro_ensemble/features.joblib')
        write(base / 'round_08_ft_seed_average/summary.json', {'seeds': [42, 43, 44, 45]})
        for seed in (42, 43, 44, 45):
            put(base / f'round_08_ft_seed_average/ft_seed{seed}.joblib')
        put(base / 'round_10_final_structure/models/ft_plr_128_3/model.joblib')
        components = []
        for number in range(12):
            relative = base / f'round_10_final_structure/component{number}.npy'
            path = put(relative, b'prediction bytes')
            components.append({'name': f'member{number}', 'path': relative.as_posix(), 'sha256': sha256(path)})
        csv = put(base / 'round_10_final_structure/kaggle_structure_blend.csv', b'Id,Predicted\n0,2000\n')
        put('output/submissions/final/submission.csv', csv.read_bytes())
        write(base / 'round_10_final_structure/summary.json', {
            'component_manifest': components,
            'source_sha256': {'data/train.csv': sha256(root / 'data/train.csv')},
            'path': csv.relative_to(root).as_posix(), 'sha256': sha256(csv)})
        write(base / 'round_10_final_structure/blend_model.json', {'names': [x['name'] for x in components]})
        return components

    def test_missing_files_are_reported_together_with_remedies(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(Path(directory), 'final', check_packages=False)
            self.assertFalse(result['ok'])
            self.assertGreater(len([x for x in result['checks'] if not x['ok']]), 9)
            self.assertTrue(all(x['remedy'] for x in result['checks']))

    def test_query_rent_column_fails_the_input_boundary_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.data(root)
            self.assertTrue(audit(root, check_packages=False)['ok'])
            (root / 'data/test.csv').write_text('field,MONTHLY_RENT\n1,9999\n', encoding='utf-8')
            result = audit(root, check_packages=False)
            self.assertFalse(next(x['ok'] for x in result['checks'] if x['resource'] == 'schema:test.csv'))

    def test_final_checks_all_weights_and_rejects_changed_predictions_without_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            components = self.final(root)
            self.assertTrue(audit(root, 'final', check_packages=False)['ok'])
            path = root / components[3]['path']
            path.write_bytes(b'modified predictions')
            result = audit(root, 'final', check_packages=False)
            self.assertFalse(result['ok'])
            self.assertTrue(any(x['resource'] == components[3]['path'] and x['detail'] == 'sha256_mismatch' for x in result['checks']))
            path.write_bytes(b'prediction bytes')
            (root / 'output/submissions/components/round_08_ft_seed_average/ft_seed44.joblib').unlink()
            self.assertFalse(audit(root, 'final', check_packages=False)['ok'])

    def test_unsafe_manifest_paths_and_empty_components_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.final(root)
            path = root / 'output/submissions/components/round_10_final_structure/summary.json'
            value = json.loads(path.read_text())
            value['component_manifest'] = [{'name': 'unsafe', 'path': '../outside.npy', 'sha256': 'anything'}]
            path.write_text(json.dumps(value))
            self.assertFalse(audit(root, 'final', check_packages=False)['ok'])
            value['component_manifest'] = []
            path.write_text(json.dumps(value))
            self.assertFalse(audit(root, 'final', check_packages=False)['ok'])
            value['component_manifest'] = [{'name': 'no_hash', 'path': 'data/train.csv'}]
            path.write_text(json.dumps(value))
            self.assertTrue(any(x['detail'] == 'missing_or_invalid_sha256' for x in audit(root, 'final', check_packages=False)['checks']))
            with self.assertRaises(ValueError):
                project_path(root, 'D:\\outside.joblib')

    def test_malformed_json_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.final(root)
            path = root / 'output/submissions/components/round_10_final_structure/blend_model.json'
            path.write_text('{broken')
            self.assertFalse(audit(root, 'final', check_packages=False)['ok'])

    def test_cli_can_run_with_site_packages_disabled_and_missing_packages_exit_nonzero(self):
        result = subprocess.run([sys.executable, '-B', '-S', '-m', 'analysis.main', 'preflight', '--json'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 1)
        value = json.loads(result.stdout)
        self.assertFalse(value['ok'])
        self.assertTrue(any(x['resource'] == 'package:numpy' and not x['ok'] for x in value['checks']))

    def test_version_comparison_respects_core_exact_pins_and_tool_minima(self):
        self.assertTrue(version_matches('1.3.0', '>=', '1.3'))
        self.assertFalse(version_matches('1.2.9', '>=', '1.3'))
        self.assertTrue(version_matches('2.11.0+cu128', '==', '2.11.0+cu128'))
        self.assertFalse(version_matches('2.11.0+cpu', '==', '2.11.0+cu128'))


if __name__ == '__main__':
    unittest.main()

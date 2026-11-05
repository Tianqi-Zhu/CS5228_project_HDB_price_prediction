"""Inventory saved files while preserving metric context and artifact roles."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.artifact_catalog import ROOT, build_catalog
from analysis.resource_preflight import sha256


class ArtifactCatalogTests(unittest.TestCase):
    def put(self, root, relative, value=b'not a loadable model'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value).encode() if isinstance(value, dict) else value)
        return path

    def test_index_covers_current_files_instead_of_only_the_legacy_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.put(root, 'output/models/index.json', {'models': []})
            checkpoint = self.put(root, 'output/models/round_10_plr_transformer_final/outer/ft/model.joblib')
            self.put(root, 'output/models/round_10_plr_transformer_final/outer/ft/predictions.csv', b'prediction\n1\n')
            result = self.put(root, 'output/experiment_runs/round_10_plr_transformer_final/outer/ft/results.json',
                              {'stage': 'outer', 'rmse_sgd': 451., 'baseline_scores': {'rmse_sgd': 452.}})
            catalog = build_catalog(root, hashes=True)
            self.assertTrue(catalog['ok'])
            self.assertEqual(catalog['entries_count'], 1)
            entry = catalog['entries'][0]
            self.assertEqual(entry['sha256'], sha256(checkpoint))
            self.assertEqual(entry['metrics_source'], result.relative_to(root).as_posix())
            self.assertEqual([x['json_pointer'] for x in entry['metrics']], ['/rmse_sgd', '/baseline_scores/rmse_sgd'])
            self.assertEqual(entry['metrics'][0]['context']['stage'], 'outer')
            self.assertEqual(len(entry['related_predictions']), 1)
            # Inventorying again must not include its own saved JSON or change entries.
            self.put(root, 'output/models/current_catalog.json', catalog)
            self.assertEqual(build_catalog(root, hashes=True)['entries'], catalog['entries'])

    def test_discarded_models_and_fusion_parameters_are_not_counted_as_new_architectures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.put(root, 'output/models/archive_weight_scale_debug/june/old/model.joblib')
            self.put(root, 'output/models/round_09_grouped_blending/outer/blend/model.json', {'weights': [1.]})
            catalog = build_catalog(root)
            self.assertEqual(catalog['counts_by_kind'], {'weight_file': 1, 'fusion_parameters': 1})
            discarded = next(x for x in catalog['entries'] if x['usage'] == 'discarded')
            self.assertEqual(discarded['experiment'], 'archive_weight_scale_debug')

    def test_final_usage_traces_saved_recipe_but_does_not_assign_ensemble_rmse_to_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = 'output/submissions/components/'
            member = base + 'history_macro_ensemble/models/tree/model.joblib'
            self.put(root, member)
            self.put(root, base + 'history_macro_ensemble/summary.json', {'members': [{'checkpoint': member}], 'local_recent_rmse': 442.9})
            self.put(root, base + 'history_macro_ensemble/features.joblib')
            self.put(root, base + 'round_08_ft_seed_average/summary.json', {'seeds': [42]})
            self.put(root, base + 'round_08_ft_seed_average/ft_seed42.joblib')
            self.put(root, base + 'round_10_final_structure/summary.json', {'local_recent_rmse': 441.8})
            self.put(root, base + 'round_10_final_structure/models/ft_plr_128_3/model.joblib')
            self.put(root, base + 'round_10_final_structure/blend_model.json', {'global_weights': [1.]})
            catalog = build_catalog(root)
            self.assertTrue(catalog['ok'], catalog['issues'])
            entry = next(x for x in catalog['entries'] if x['path'] == member)
            self.assertEqual(entry['usage'], 'final')
            self.assertEqual(entry['metrics'], [])
            self.assertEqual(entry['export_context']['metrics'][0]['value'], 442.9)
            self.assertEqual(catalog['counts_by_kind']['feature_cache'], 1)
            (root / member).unlink()
            self.assertTrue(any(x['path'] == member and x['issue'] == 'recorded_final_dependency_missing' for x in build_catalog(root)['issues']))

    def test_cli_listing_needs_no_site_packages_and_filtered_write_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            member = 'output/submissions/components/history_macro_ensemble/models/tree/model.joblib'
            self.put(fixture, member)
            self.put(fixture, 'output/submissions/components/history_macro_ensemble/summary.json',
                     {'members': [{'checkpoint': member}]})
            # Exercise the real unified CLI with a controlled resource root,
            # independent of whether ignored local artifacts were distributed.
            program = ('from pathlib import Path; import sys, functools; '
                       'import analysis.artifact_catalog as a; '
                       'a.build_catalog=functools.partial(a.build_catalog,root=Path(sys.argv[1])); '
                       'from analysis.main import main; main(["artifact-list","--usage","final","--json"])')
            result = subprocess.run([sys.executable, '-B', '-S', '-c', program, directory],
                                    cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            catalog = json.loads(result.stdout)
            self.assertEqual([x['path'] for x in catalog['entries']], [member])
            self.assertTrue(all(x['usage'] == 'final' for x in catalog['entries']))
        rejected = subprocess.run([sys.executable, '-B', '-S', '-m', 'analysis.artifact_catalog', '--usage', 'final', '--write'],
                                  cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(rejected.returncode, 2)


if __name__ == '__main__':
    unittest.main()

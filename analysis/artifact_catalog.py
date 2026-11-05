"""Inventory current weight files and fusion parameters without loading models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from analysis.resource_preflight import ROOT, project_path, sha256

ROOTS = ('output/models', 'output/submissions/components')
WEIGHT_EXTENSIONS = {'.joblib', '.pt', '.cbm', '.zip', '.pkl'}
STAGES = {'screen', 'inner', 'outer', 'june', 'confirm', 'smoke', 'final', 'all'}


def read_object(path, issues):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except (OSError, ValueError) as error:
        issues.append({'path': str(path), 'issue': str(error)})
        return {}


def metric_values(value, pointer='', inherited=None):
    """Keep all named metrics and their context; do not merge validation windows."""
    rows = []
    if isinstance(value, dict):
        context = dict(inherited or {})
        for name in ('stage', 'model', 'variant', 'name', 'validation_start', 'validation_end',
                     'origin', 'rows', 'n', 'n_validation', 'meta_fit_last_month'):
            if name in value and isinstance(value[name], (str, int, float)):
                context[name] = value[name]
        for name, item in value.items():
            location = pointer + '/' + str(name)
            if any(token in name.lower() for token in ('rmse', 'mae')) and isinstance(item, (int, float)) and not isinstance(item, bool):
                rows.append({'metric': name, 'value': item, 'json_pointer': location, 'context': context})
            elif isinstance(item, (dict, list)):
                rows += metric_values(item, location, context)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows += metric_values(item, pointer + '/' + str(index), inherited)
    return rows


def final_dependencies(root, issues):
    """Trace the recorded full-data final recipe, not a score-based selection."""
    base = 'output/submissions/components/'
    paths = set()
    history = root / (base + 'history_macro_ensemble/summary.json')
    if history.is_file():
        for member in read_object(history, issues).get('members', []):
            if isinstance(member, dict) and isinstance(member.get('checkpoint'), str):
                try:
                    paths.add(project_path(root, member['checkpoint']).relative_to(root).as_posix())
                except ValueError as error:
                    issues.append({'path': member['checkpoint'], 'issue': str(error)})
    seeds = root / (base + 'round_08_ft_seed_average/summary.json')
    if seeds.is_file():
        for seed in read_object(seeds, issues).get('seeds', []):
            if isinstance(seed, int):
                paths.add(base + f'round_08_ft_seed_average/ft_seed{seed}.joblib')
    if (root / (base + 'round_10_final_structure/summary.json')).is_file():
        paths.update({base + 'history_macro_ensemble/features.joblib',
                      base + 'round_10_final_structure/models/ft_plr_128_3/model.joblib',
                      base + 'round_10_final_structure/blend_model.json'})
    return paths


def build_catalog(root=ROOT, hashes=False):
    root = Path(root).resolve()
    issues = []
    final = final_dependencies(root, issues)
    entries = []
    metric_cache = {}
    exports = {}
    for relative_root in ROOTS:
        base = root / relative_root
        for path in sorted(base.rglob('*')):
            if not path.is_file() or (path.suffix not in WEIGHT_EXTENSIONS and path.name not in ('model.json', 'blend_model.json')):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                issues.append({'path': str(path), 'issue': 'outside_project'})
                continue
            relative = path.relative_to(root).as_posix()
            parts = path.relative_to(base).parts
            experiment = parts[0]
            kind = 'weight_file'
            if path.name in ('features.joblib', 'prepared.joblib'):
                kind = 'feature_cache'
            elif path.suffix == '.json':
                kind = 'fusion_parameters'
                read_object(path, issues)
            usage = ('discarded' if experiment == 'archive_weight_scale_debug' else
                     'final' if relative in final else
                     'submission_component' if relative_root.endswith('components') else 'development')
            stage = next((part for part in parts[1:-1] if part in STAGES), None)
            folder = root / 'output/experiment_runs' / experiment
            metrics_source = None
            metrics = []
            export = None
            if relative_root == 'output/models':
                candidate = root / 'output/experiment_runs' / path.parent.relative_to(base)
                while candidate.is_relative_to(folder):
                    result = candidate / 'results.json'
                    if result.is_file():
                        if result not in metric_cache:
                            metric_cache[result] = metric_values(read_object(result, issues))
                        metrics = metric_cache[result]
                        metrics_source = result.relative_to(root).as_posix()
                        break
                    if candidate == folder:
                        break
                    candidate = candidate.parent
            else:
                summary = base / experiment / 'summary.json'
                if summary.is_file():
                    if summary not in exports:
                        exports[summary] = {'path': summary.relative_to(root).as_posix(),
                            'metrics': metric_values(read_object(summary, issues)),
                            'scope': 'Export/ensemble context; not validation RMSE of this individual full-data checkpoint.'}
                    export = exports[summary]
            predictions = [p.relative_to(root).as_posix() for p in sorted(path.parent.glob('*'))
                           if p.is_file() and p.name in ('predictions.csv', 'predictions.npy', 'ft_predictions.npy')]
            source_folder = root / 'analysis/experiments/optimization' / experiment
            modules = [p.relative_to(root).as_posix() for p in sorted(source_folder.glob('*.py'))]
            entries.append({'path': relative, 'kind': kind, 'usage': usage, 'experiment': experiment,
                            'stage': stage, 'bytes': path.stat().st_size,
                            'sha256': sha256(path) if hashes else None,
                            'metrics_source': metrics_source, 'metrics': metrics,
                            'export_context': export, 'related_predictions': predictions,
                            'source_modules': modules})
    recorded = {item['path'] for item in entries}
    for missing in sorted(final - recorded):
        issues.append({'path': missing, 'issue': 'recorded_final_dependency_missing'})
    return {'schema_version': 1, 'ok': not issues, 'roots': list(ROOTS),
            'hashes_computed': hashes, 'entries_count': len(entries),
            'counts_by_kind': dict(Counter(item['kind'] for item in entries)),
            'counts_by_usage': dict(Counter(item['usage'] for item in entries)),
            'entries': entries, 'issues': issues,
            'interpretation': [
                'Counts are saved files, not unique model architectures or statistically independent experiments.',
                'Development, full-data submission, fusion parameters and discarded debug runs have separate roles.',
                'Metrics retain original JSON pointers and context; no cross-window RMSE ranking is generated.',
                'Export context describes an ensemble/recipe and must not be assigned to each individual checkpoint.',
                'Legacy output/models/index.json is an unchanged historical snapshot. No model file is deserialized.',
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', dest='as_json')
    parser.add_argument('--hashes', action='store_true', help='Read files to compute SHA-256; does not load models')
    parser.add_argument('--write', action='store_true', help='Write the complete current catalog to output/models/current_catalog.json')
    parser.add_argument('--usage', choices=('all', 'final', 'development', 'submission_component', 'discarded'), default='all')
    args = parser.parse_args(argv)
    if args.write and args.usage != 'all':
        parser.error('--write requires --usage all so the saved catalog stays complete')
    result = build_catalog(hashes=args.hashes)
    if args.usage != 'all':
        result = result | {'filter': args.usage, 'entries': [item for item in result['entries'] if item['usage'] == args.usage]}
        result['entries_count'] = len(result['entries'])
        result['counts_by_kind'] = dict(Counter(item['kind'] for item in result['entries']))
        result['counts_by_usage'] = dict(Counter(item['usage'] for item in result['entries']))
    if args.write:
        target = project_path(ROOT, 'output/models/current_catalog.json')
        if not result['ok']:
            parser.error('Catalog has issues; the existing saved catalog was not overwritten')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Artifact files: {result['entries_count']}; kinds={result['counts_by_kind']}; uses={result['counts_by_usage']}")
        if args.write:
            print('Saved output/models/current_catalog.json')
        if args.usage != 'all':
            for item in result['entries']:
                print(item['kind'] + '  ' + item['path'])
        for issue in result['issues']:
            print('ISSUE ' + str(issue))
        print('File counts are not architecture counts. Metrics retain their original windows; no model is loaded or trained.')
    if not result['ok']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()

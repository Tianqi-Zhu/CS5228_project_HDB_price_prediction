"""Read-only resource checks using the standard library; never load model pickles."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('core', 'reports', 'final', 'wave06', 'wave07', 'all')
DATA_FILES = (
    'train.csv', 'test.csv', 'example-submission.csv', 'auxiliary/sg-hdb-block.csv',
    'auxiliary/sg-mrt-stations.csv', 'auxiliary/sg-schools.csv',
    'auxiliary/sg-shopping-malls.csv', 'auxiliary/sg-coe-prices.csv',
    'auxiliary/sg-stock-prices.csv',
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def project_path(root, relative):
    """Resolve cross-platform manifest paths without following them outside root."""
    if not isinstance(relative, str) or not relative or re.match(r'^[A-Za-z]:', relative):
        raise ValueError('Expected a project-relative path')
    path = (Path(root) / relative.replace('\\', '/')).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path leaves project root')
    return path


def package_versions(root, profile):
    # Some core neural packages in the existing checkout are provided by author
    # loaders' isolated targets rather than global site-packages. Report where
    # their metadata was found; this does not simulate imports or change sys.path.
    folders = ['deep-deps', 'model-deps']
    if profile in ('wave07', 'all'):
        folders = ['transformer-deps', 'foundation-deps'] + folders
    search = [str(root / '.cache' / folder) for folder in folders] + list(sys.path)
    versions = {}
    for distribution in metadata.distributions(path=search):
        name = distribution.metadata.get('Name')
        if name:
            versions.setdefault(re.sub(r'[-_.]+', '-', name).lower(),
                                {'version': distribution.version, 'location': str(distribution.locate_file(''))})
    return versions


def constraints(path):
    """Read the simple direct-package constraints used by this repository."""
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.split('#', 1)[0].strip()
        # environment.yml pip list entries and optional plain requirements.
        line = re.sub(r'^-\s+', '', line)
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(==|>=)([0-9][A-Za-z0-9.+-]*)', line)
        if match:
            result.append(match.groups())
    return result


def version_matches(actual, operator, expected):
    if operator == '==':
        return actual == expected
    numbers = lambda value: tuple(int(part) for part in re.match(r'^\d+(?:\.\d+)*', value).group().split('.'))
    left, right = numbers(actual), numbers(expected)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def audit(root=ROOT, profile='core', check_packages=True):
    root = Path(root).resolve()
    if profile not in PROFILES:
        raise ValueError(profile)
    checks = []
    seen = set()

    def add(name, ok, detail, remedy):
        checks.append({'resource': name, 'ok': bool(ok), 'detail': detail, 'remedy': remedy})

    def file(relative, remedy, expected_hash=None):
        key = (relative, expected_hash)
        if key in seen:
            return None
        seen.add(key)
        try:
            path = project_path(root, relative)
        except ValueError as error:
            add(str(relative), False, str(error), 'Fix the manifest; paths must stay inside the repository.')
            return None
        if not path.is_file():
            add(relative, False, 'missing_file', remedy)
            return None
        if expected_hash and sha256(path) != expected_hash:
            add(relative, False, 'sha256_mismatch', 'Restore the recorded input/artifact version or regenerate dependent results.')
            return None
        add(relative, True, 'sha256_verified' if expected_hash else 'file_exists', remedy)
        return path

    def manifest(relative, remedy):
        path = file(relative, remedy)
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(value, dict):
                raise ValueError('Expected an object')
            return value
        except (ValueError, OSError) as error:
            add(relative + ':json', False, str(error), 'Restore a valid JSON manifest.')
            return None

    def entries(items, path_key, remedy):
        if not isinstance(items, list) or not items:
            add('manifest:' + path_key, False, 'missing_or_empty_entries', 'Regenerate the manifest with its resource list.')
            return
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get(path_key), str):
                add('manifest:' + path_key, False, 'invalid_entry', 'Repair the malformed manifest entry.')
                continue
            if path_key == 'path' and not re.fullmatch(r'[0-9a-fA-F]{64}', str(item.get('sha256', ''))):
                add(item[path_key], False, 'missing_or_invalid_sha256', 'Restore the recorded component SHA-256 manifest.')
                continue
            file(item[path_key], remedy, item.get('sha256') if path_key == 'path' else None)

    if check_packages:
        add('python', sys.version_info[:2] == (3, 11), sys.version.split()[0], 'Use Python 3.11 from environment.yml.')
        versions = package_versions(root, profile)
        configs = ['environment.yml']
        if profile in ('wave06', 'all'):
            configs.append('requirements/optional_deep_models.txt')
        if profile in ('wave07', 'all'):
            configs.append('requirements/optional_pretrained_transformers.txt')
        for relative in configs:
            path = file(relative, 'Restore the dependency configuration from this checkout.')
            if path is None:
                continue
            for name, operator, expected in constraints(path):
                provider = versions.get(re.sub(r'[-_.]+', '-', name).lower(), {})
                actual = provider.get('version')
                add('package:' + name, actual is not None and version_matches(actual, operator, expected),
                    {'expected': operator + expected, 'installed_metadata': actual,
                     'provider': provider.get('location')},
                    'Follow ' + relative + ' and requirements/README.md; optional packages use their documented isolated target.')

    for name in DATA_FILES:
        file('data/' + name, 'Restore the original competition CSV; see data/README.md.')
    for name, target_expected in (('train.csv', True), ('test.csv', False)):
        path = root / 'data' / name
        if path.is_file():
            with path.open(encoding='utf-8-sig', newline='') as stream:
                header = next(csv.reader(stream), [])
            add('schema:' + name, ('MONTHLY_RENT' in header) == target_expected,
                {'target_present': 'MONTHLY_RENT' in header}, 'Restore the original train/test schema.')

    if profile in ('reports', 'all'):
        from analysis.report_bundle import READING_FILES
        for source in READING_FILES.values():
            file(source.relative_to(ROOT).as_posix(), 'Generate the corresponding earlier experiment before experiment-refresh; see docs/guides/running_experiments.md.')

    if profile in ('final', 'all'):
        base = 'output/submissions/components/'
        history = manifest(base + 'history_macro_ensemble/summary.json', 'Restore full-data components or run best_ensemble_submission with its earlier experiment dependencies.')
        if history is not None:
            members = history.get('members')
            add('final:history_member_count', isinstance(members, list) and len(members) == 10,
                len(members) if isinstance(members, list) else None, 'Restore the ten-member history export manifest.')
            entries(history.get('members'), 'checkpoint', 'Restore/retrain the recorded full-data base model.')
        file(base + 'history_macro_ensemble/features.joblib', 'Restore the full-data feature bundle or rebuild the history export.')
        seed = manifest(base + 'round_08_ft_seed_average/summary.json', 'Run eighth_wave_submission after preparing earlier results.')
        if seed is not None:
            seeds = seed.get('seeds')
            valid_seeds = isinstance(seeds, list) and len(seeds) == 4 and all(isinstance(x, int) for x in seeds) and set(seeds) == {42, 43, 44, 45}
            add('final:ft_seed_list', valid_seeds, seeds, 'Restore the four-seed export manifest.')
            if valid_seeds:
                for number in seeds:
                    file(base + f'round_08_ft_seed_average/ft_seed{number}.joblib', 'Restore/retrain the four full-data FT models.')
        final = manifest(base + 'round_10_final_structure/summary.json', 'Prepare ninth/tenth-wave models and run tenth_wave_report; see the running guide.')
        if final is not None:
            entries(final.get('component_manifest'), 'path', 'Restore or regenerate the recorded full-data component prediction.')
            sources = final.get('source_sha256')
            if not isinstance(sources, dict) or not sources:
                add('final:source_sha256', False, 'missing_source_hashes', 'Restore the final export source manifest.')
            else:
                for relative, expected in sources.items():
                    file(relative, 'Restore the recorded competition data version.', expected)
            if isinstance(final.get('path'), str):
                file(final['path'], 'Regenerate the component submission.', final.get('sha256'))
            file('output/submissions/final/submission.csv', 'Run final_submission_selection after verifying candidate CSVs.', final.get('sha256'))
        file(base + 'round_10_final_structure/models/ft_plr_128_3/model.joblib', 'Restore/retrain the full-data PLR-FT model.')
        blend = manifest(base + 'round_10_final_structure/blend_model.json', 'Restore/refit the selected grouped fusion.')
        if blend is not None and final is not None:
            names = [item.get('name') for item in final.get('component_manifest', []) if isinstance(item, dict)]
            ordered = blend.get('names')
            add('final:component_order', isinstance(ordered, list) and ordered == names and len(names) == 12,
                {'blend_names': ordered, 'manifest_names': names}, 'Restore matching fusion parameters and ordered component manifest.')

    if profile in ('wave06', 'all'):
        for relative in ('TALENT/model/models/modernNCA.py', 'TALENT/model/lib/tabr/utils.py',
                         'TALENT/configs/default/modernNCA.json'):
            file('.cache/deep-sources/' + relative, 'Prepare the recorded TALENT author source; see the sixth-wave report.')
    if profile in ('wave07', 'all'):
        checkpoints = manifest('output/experiment_runs/round_07_pretrained_transformers/checkpoint_manifest.json', 'Prepare the recorded public Transformer checkpoints using seventh_wave_foundations.')
        if checkpoints is not None:
            for name in ('tabdpt13', 'nori15', 'nori6m'):
                item = checkpoints.get(name, {})
                file(item.get('path'), 'Prepare the recorded public checkpoint; see the seventh-wave report.', item.get('sha256'))
        tabicl = manifest('.cache/foundation-models/tabicl_manifest.json', 'Prepare the public TabICL v2 checkpoint using second_wave_foundation.')
        if tabicl is not None:
            file('.cache/foundation-models/tabicl-regressor-v2-20260212.ckpt', 'Prepare the recorded TabICL checkpoint.', tabicl.get('sha256'))
        for relative in ('foundation-deps/tabicl/__init__.py',
                         'transformer-deps/synthefy_nori/configs/default_inference.json',
                         'transformer-deps/synthefy_nori/configs/model_v3_inference.json'):
            file('.cache/' + relative, 'Prepare the documented author package/support files for seventh-wave models.')

    return {'profile': profile, 'root': str(root), 'ok': all(item['ok'] for item in checks),
            'checks_passed': sum(item['ok'] for item in checks), 'checks': checks,
            'limits': 'Static file, package-metadata, recorded-hash and manifest checks only. Package metadata includes documented isolated targets; their author loaders must still activate the appropriate paths. No model imports, unpickling, GPU test, download, installation or training. Passing does not certify all transitive dependencies or a fresh environment.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='core')
    parser.add_argument('--json', action='store_true', dest='as_json')
    args = parser.parse_args(argv)
    result = audit(profile=args.profile)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Resource preflight: {args.profile}; {result['checks_passed']}/{len(result['checks'])} passed")
        for item in result['checks']:
            if not item['ok']:
                print(f"MISSING/INVALID {item['resource']}: {item['detail']}\n  {item['remedy']}")
        print(result['limits'])
    if not result['ok']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()

"""Distinguish ordered raw feature schemas, model inputs and encoded dimensions."""
from pathlib import Path
import argparse
import hashlib
import json

from analysis.pipeline.config import ROOT, FEATURE_SETS
from analysis.validation.prepare_inputs import sha

INPUT_META = ROOT / 'experiment/main/M00/inputs/metadata.json'
REFITS = ROOT / 'reproduction/f1/validation_evidence'
ADJUSTED = ['COHORT24_ADJUSTED_RENT', 'COHORT24_RATIO_VS_BLOCK', 'COHORT24_SHRUNK_RATIO',
            'PAST_TYPE_LEVEL', 'COHORT24_RELATIVE_SUPPORT']


def adjusted_schema(ladder):
    """sixth_wave_deep.prepare forces adjusted=True before dropping ANN36."""
    result = [dict(c) for c in ladder if not c['name'].startswith('ANN_BLOCK_36_')]
    result += [{'name': c, 'dtype': 'float32'} for c in ADJUSTED]
    if len(result) != 62 or len({c['name'] for c in result}) != 62:
        raise ValueError('Expected 62 distinct adjusted raw inputs')
    return result


def resolved_record_schema(rec, meta):
    if rec['model'] in ('single_PLE', 'single_PLR'):
        return adjusted_schema(meta['feature_schemas'][rec['window'] + '_ladder_62'])
    return rec['schema']


def bridge(before, after):
    a, b = set(before), set(after)
    return {'added': sorted(b - a), 'removed': sorted(a - b),
            'same_ordered_schema': before == after, 'strictly_nested': a < b}


def catalog():
    meta = json.loads(INPUT_META.read_text('utf-8'))
    schemas, uses, corrections = {}, [], []

    def add(key, columns, source, scope):
        names = [c['name'] if isinstance(c, dict) else c for c in columns]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate feature name')
        if key in schemas and schemas[key]['ordered_names'] != names:
            raise ValueError('Same schema ID has different ordered inputs')
        schemas[key] = {'ordered_names': names, 'feature_count': len(names),
                        'ordered_names_sha256': hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest(),
                        'source': source, 'scope': scope}
        if columns and isinstance(columns[0], dict): schemas[key]['dtypes'] = columns
        return names

    c7 = add('C7', FEATURE_SETS['C_property_time_town'], 'analysis/pipeline/config.py:FEATURE_SETS',
             'Raw columns; one-hot encoded dimension differs by fitted vocabulary')
    for level in (19, 27, 47, 62):
        y, j = (meta['feature_schemas'][w + '_ladder_' + str(level)] for w in ('Y', 'J'))
        assert y == j
        add('ladder' + str(level), y, 'experiment/main/M00/inputs/metadata.json', 'Matched LightGBM/CatBoost ladder raw inputs')
    adjusted = add('adjusted62', adjusted_schema(meta['feature_schemas']['Y_ladder_62']),
                   'sixth_wave_deep.prepare -> third_wave_training.inputs; fresh PLR payload',
                   'PLE/PLR raw input contract, not encoded token/missing-indicator count')
    for rec in meta['predictions']:
        if rec['model'] in ('single_PLE', 'single_PLR'):
            old = [c['name'] for c in rec['schema']]
            corrections.append({'window': rec['window'], 'model': rec['model'],
                                'archived_metadata_count': len(old), 'resolved_count': len(adjusted),
                                'reason': 'Old metadata construction omitted adjusted=True forced by trainer.prepare',
                                **bridge(old, adjusted), 'predictions_or_metrics_changed': False})
    for window in ('Y', 'J'):
        path = REFITS / window / 'validation_run.json'
        report = json.loads(path.read_text('utf-8'))
        for member, columns in report['component_schemas'].items():
            key = 'F1:' + member
            add(key, columns, path.relative_to(ROOT).as_posix(), 'Actual saved learner input, before internal encoding')
            record = next(r for r in report['model_records'] if r['name'] == member)
            uses.append({'window': window, 'member': member, 'schema_id': key,
                         'raw_feature_count': len(columns), 'epochs_or_iterations': record['count'],
                         'fit_rows': record['fit_rows'], 'checkpoint_sha256': record['checkpoint_sha256'],
                         'generating_commit': report['generating_commit']})
        assert report['component_schemas']['tenth/ft_plr_128_3'] == adjusted
    concat_path = ROOT / 'reproduction/concat_transformer/artifacts/feature_manifest.json'
    concat = json.loads(concat_path.read_text('utf-8'))
    assert set(concat['selected_columns']) == set(adjusted)
    add('concat_raw62', concat['selected_columns'], concat_path.relative_to(ROOT).as_posix(),
        'Same raw column set as adjusted62; preprocessing order is recorded separately')
    add('concat_encoded99', concat['numeric_names'] + concat['categorical_names'],
        concat_path.relative_to(ROOT).as_posix(), '95 numeric inputs including missing flags + 4 category codes; not 99 raw features')
    pairs = [('C7', 'ladder19'), ('ladder19', 'ladder27'), ('ladder27', 'ladder47'),
             ('ladder47', 'ladder62'), ('ladder62', 'adjusted62'), ('adjusted62', 'concat_raw62')]
    transitions = [{'before': a, 'after': b, **bridge(schemas[a]['ordered_names'], schemas[b]['ordered_names'])}
                   for a, b in pairs]
    assert not transitions[0]['strictly_nested']
    assert all(t['strictly_nested'] for t in transitions[1:4])
    assert len(transitions[4]['added']) == len(transitions[4]['removed']) == 5
    return {'all_passed': True, 'schemas': schemas, 'F1_member_uses': uses,
            'transitions': transitions, 'metadata_corrections': corrections,
            'archived_metadata_sha256': sha(INPUT_META), 'old_inputs_preserved': True,
            'comparison_rule': 'Use ordered names, target, protocol and budget; equal column counts alone do not define a matched control',
            'F2': 'Pending actual ordered 87-column schema; broad87 F1 member is not proof of F2 identity'}


def run(output_dir=None):
    from analysis.validation.main import dump, table
    result = catalog()
    signature = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
    destination = Path(output_dir or ROOT / 'output/feature_catalog' / signature).resolve()
    if not destination.is_relative_to(ROOT / 'output'):
        raise ValueError('New catalog output must be below output/')
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / 'schemas.json', result)
    table(destination / 'registry.csv', [{'schema_id': key, **{k: value for k, value in schema.items()
          if k not in ('ordered_names', 'dtypes')}} for key, schema in result['schemas'].items()])
    table(destination / 'F1_member_inputs.csv', result['F1_member_uses'])
    table(destination / 'transitions.csv', [{k: json.dumps(v) if isinstance(v, list) else v for k, v in t.items()}
                                           for t in result['transitions']])
    print('Ordered feature schemas verified:', destination)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == '__main__': main()

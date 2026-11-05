"""Second prespecified screen: stable relative targets and recency weighting."""
from __future__ import annotations
import argparse
from analysis.experiments import frontier_search as search

TRIALS = []
for half_life in (6, 12, 24):
    for target in ('raw', 'ratio'):
        TRIALS.append({'name': f'recent{half_life}_{target}', 'family': 'lgb',
            'snapshot': False, 'neighbors': False, 'target': target,
            'leaves': 31, 'half_life': half_life})
for snapshot in (False, True):
    for leaves in (15, 31):
        TRIALS.append({'name': f'relative_{"snapshot" if snapshot else "normal"}_{leaves}',
            'family': 'lgb', 'snapshot': snapshot, 'neighbors': False,
            'target': 'ratio', 'leaves': leaves, 'features': 'relative',
            'min_child': 150, 'l2': 40.})
for snapshot in (False, True):
    TRIALS.append({'name': f'compact_{"snapshot" if snapshot else "normal"}',
        'family': 'lgb', 'snapshot': snapshot, 'neighbors': False,
        'target': 'residual', 'leaves': 15, 'features': 'no_entities',
        'min_child': 150, 'l2': 40.})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen', 'outer'])
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args()
    search.write_json(search.OUT / 'refinement_plan.json', {'trials': TRIALS,
        'recorded_before_outer_evaluation': True})
    search.TRIALS = TRIALS
    # Keep the first-phase plan intact.
    original_write = search.write_json
    def write(path, value):
        if path.name != 'plan.json':
            original_write(path, value)
    search.write_json = write
    search.run(args.stage, args.names)


if __name__ == '__main__':
    main()

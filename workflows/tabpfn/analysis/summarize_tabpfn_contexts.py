"""Verify completed context runs and write one comparison report."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.config import DATA, ROOT
from analysis.data import sha256
from analysis.experiments.tabpfn_comparison import FOLDS, TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+',
                        default=[Path('reports/tabpfn'), Path('reports/tabpfn_50k')])
    parser.add_argument('--output', type=Path, default=Path('reports/tabpfn'))
    args = parser.parse_args()
    source = pd.read_csv(DATA / 'train.csv')
    expected = source.index[source.RENT_APPROVAL_DATE.ge(FOLDS[0][0]) &
                            source.RENT_APPROVAL_DATE.lt(FOLDS[-1][1])]
    rows, checks, candidates, run_descriptions = [], [], [], []
    reference, shared = None, None
    labels = {'catboost': 'CatBoost', 'tabpfn_spatial': 'TabPFN spatial',
              'tabpfn_history': 'TabPFN + history',
              'catboost_tabpfn_forward_blend': 'Forward blend'}
    for directory in args.runs:
        result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        settings = result['settings']
        environment_path = directory / 'runtime_environment.json'
        flags = json.loads(environment_path.read_text())['environment'] if environment_path.exists() else {}
        execution = ('sequential estimator execution' if flags.get('TABPFN_MAX_BATCHED_ESTIMATOR_ROWS') == '0'
                     else 'checkpoint-default estimator batching')
        run_descriptions.append(f"{settings['context_rows']:,}-row context: {settings['batch_size']}-row prediction batches, {execution}.")
        pred = pd.read_csv(directory / 'validation_predictions.csv')
        assert pred.row_id.is_unique and set(pred.row_id) == set(expected)
        np.testing.assert_array_equal(pred[TARGET], source.loc[pred.row_id, TARGET])
        for name, digest in settings['sources_sha256'].items():
            assert sha256(ROOT / name) == digest, name
        provenance = [settings[k] for k in ['sources_sha256', 'implementation_sha256',
                                           'shared_code_sha256', 'versions', 'model_version',
                                           'n_estimators', 'seed', 'features', 'folds',
                                           'selection_rule', 'catboost_settings']]
        checkpoint_hashes = {t['checkpoint_sha256'] for t in result['timings'] if 'checkpoint_sha256' in t}
        assert len(checkpoint_hashes) == 1
        provenance.append(sorted(checkpoint_hashes))
        if shared is None:
            shared = provenance
            reference = pred.set_index('row_id')[[TARGET, 'catboost']].sort_index()
        else:
            assert provenance == shared, 'Runs must use the same sources, implementation and packages'
            np.testing.assert_allclose(pred.set_index('row_id')[[TARGET, 'catboost']].sort_index(),
                                       reference, rtol=0, atol=1e-6)
        for start, end in FOLDS:
            part = pred[pred.validation_start.eq(start)]
            assert part['_MONTH'].ge(start).all() and part['_MONTH'].lt(end).all()
            manifest = json.loads((directory / f'completed_{start}.json').read_text(encoding='utf-8'))
            assert manifest['prediction_sha256'] == sha256(directory / f'predictions_{start}.csv')
        pooled = pd.read_csv(directory / 'pooled_scores.csv')
        folds = pd.read_csv(directory / 'fold_scores.csv')
        for score in pooled.itertuples():
            actual = np.sqrt(np.mean((pred[score.model] - pred[TARGET]) ** 2))
            assert abs(actual - score.rmse_sgd) < 1e-6
            if score.model == 'catboost' and directory != args.runs[0]:
                continue
            fold_values = folds[folds.model.eq(score.model)].set_index('validation_start').rmse_sgd
            rows.append({'model': labels[score.model],
                         'context_rows': 'all history' if score.model == 'catboost' else str(settings['context_rows']),
                         'pooled_rmse_sgd': score.rmse_sgd, 'mae_sgd': score.mae_sgd,
                         'bias_sgd': score.bias_sgd,
                         **{start: float(fold_values.loc[start]) for start, _ in FOLDS}})
            if score.model == result['selected_candidate']:
                candidates.append((score.rmse_sgd, labels[score.model], settings['context_rows']))
        checks.append({'run': str(directory), 'rows': len(pred), 'source_hashes': 'passed',
                       'unique_source_row_ids': 'passed', 'disjoint_windows': 'passed',
                       'source_labels': 'passed', 'prediction_hashes': 'passed', 'pooled_rmse': 'passed'})
    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'context_comparison.csv', index=False, float_format='%.8f')
    (args.output / 'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    best = min(candidates)
    cat_rmse = float(table.loc[table.model.eq('CatBoost'), 'pooled_rmse_sgd'].iloc[0])
    alternative = table.loc[~table.model.eq('CatBoost')].sort_values('pooled_rmse_sgd').iloc[0]
    difference = alternative.pooled_rmse_sgd - cat_rmse
    observation = (f"The lowest observed TabPFN-based pooled RMSE is {alternative.pooled_rmse_sgd:.2f} "
                   f"({alternative['model']}, {int(alternative.context_rows):,}-row context), "
                   f"{abs(difference):.2f} SGD {'higher' if difference >= 0 else 'lower'} than CatBoost. "
                   'The selection rule also requires consistent improvement across forecast origins.')
    columns = ['model', 'context_rows', 'pooled_rmse_sgd', *[s for s, _ in FOLDS]]
    lines = ['| Model | Context rows | Pooled RMSE | 2023-01 | 2023-07 | 2024-04 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in table[columns].itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    links = '\n'.join(f'- [{json.loads((p / "settings.json").read_text())["context_rows"]:,}-row context report]({p.resolve().relative_to(args.output.resolve()).as_posix()}/report.md)'
                      if p.resolve().is_relative_to(args.output.resolve()) else
                      f'- [{json.loads((p / "settings.json").read_text())["context_rows"]:,}-row context report](../{p.name}/report.md)'
                      for p in args.runs)
    text = f'''# TabPFN context-size comparison

Selected candidate: **{best[1]}**{'' if best[1] == 'CatBoost' else f' with {best[2]:,} context rows'}.

{observation}

Each completed run uses the same {len(expected):,} transaction validation rows across three disjoint fixed-origin windows, the same TabPFN-3.5 checkpoint, {settings['n_estimators']} estimators and seed {settings['seed']}. Spatial inputs have 29 columns; rental histories add 12 strictly prior summaries. The target is monthly rent in SGD and the scoring metric is RMSE. Only the completed runs listed below are included.

{chr(10).join(lines)}

Promotion requires at least a 1% pooled RMSE improvement and improvement on every fold. The forward blend may tie its uncalibrated first fold, but must improve both later folds. These rules were fixed before TabPFN scoring. The blend weight for each later fold uses only earlier completed validation windows.

{' '.join(run_descriptions)} Execution settings, timing and memory are recorded in each run; the model inputs, ensemble count and predictive-mean output remain the same. No COE or stock observations enter either model, and validation rents never update history features.

The validation windows have already been examined in this project. These are development comparisons, not an untouched final test. This experiment does not evaluate full-history TabPFN contexts, fine-tuning, log-target variants, or a new submission.

## Detailed results and verification

{links}

`context_comparison.csv` includes RMSE, MAE and mean bias. `verification.json` records checks of unique source rows, labels, source/prediction hashes, disjoint windows and independently recomputed pooled RMSE. The CatBoost control predictions agree across runs.

Regenerate with `.venv/Scripts/python.exe analysis/summarize_tabpfn_contexts.py` after both runs complete.
'''
    (args.output / 'comparison_report.md').write_text(text, encoding='utf-8')
    print(table.to_string(index=False))
    print(f'Selected candidate: {best[1]}; all artifact checks passed.')


if __name__ == '__main__':
    main()

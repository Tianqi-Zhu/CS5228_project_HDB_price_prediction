"""Verify temporal adaptation artifacts, explain transfer and render diagnostics."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import DATA, ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.tabpfn_adaptation import MODELS, ORIGINS, FOLDS
from analysis.experiments.tabpfn_adaptation_features import HEAD_COLUMNS
from analysis.experiments.tabpfn_residual_heads import chronological_head_split


OUT = REPORTS / 'tabpfn_adaptation'


def main():
    settings = json.loads((OUT / 'settings.json').read_text(encoding='utf-8'))
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    cache = ROOT / '.cache/tabpfn_adaptation' / signature[:16]
    source = pd.read_csv(DATA / 'train.csv')
    for name, digest in settings['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in settings['code_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    episodes = []
    for origin in ORIGINS:
        path = cache / f'episode_{origin}.npz'
        manifest = json.loads(path.with_suffix('.json').read_text())
        assert manifest['signature'] == signature and manifest['sha256'] == sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        row_ids = arrays['row_id']
        assert len(np.unique(row_ids)) == len(row_ids)
        np.testing.assert_array_equal(source.loc[row_ids, 'MONTHLY_RENT'], arrays['actual'])
        np.testing.assert_array_equal(source.loc[row_ids, 'RENT_APPROVAL_DATE'], arrays['month'])
        assert (arrays['origin'] == origin).all() and (arrays['month'] >= origin).all()
        context_end = manifest['audit']['context_last_month']
        assert context_end < origin
        assert source.loc[row_ids, 'RENT_APPROVAL_DATE'].gt(context_end).all()
        horizon = pd.PeriodIndex(arrays['month'], freq='M').asi8 - pd.Period(origin, 'M').ordinal + 1
        np.testing.assert_array_equal(horizon, arrays['horizon'])
        np.testing.assert_array_equal(horizon, arrays['numeric'][:, HEAD_COLUMNS.index('FORECAST_HORIZON')])
        assert set(horizon) <= {1, 3, 6, 12}
        assert np.isfinite(arrays['embedding']).all() and arrays['embedding'].shape[1] == 1024
        episodes.append(arrays)
    head_data = {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
    pred = pd.read_csv(OUT / 'validation_predictions.csv')
    expected = source.index[source.RENT_APPROVAL_DATE.ge(FOLDS[0][0]) &
                            source.RENT_APPROVAL_DATE.lt(FOLDS[-1][1])]
    assert pred.row_id.is_unique and set(pred.row_id) == set(expected)
    np.testing.assert_array_equal(pred.MONTHLY_RENT, source.loc[pred.row_id, 'MONTHLY_RENT'])
    pooled = pd.read_csv(OUT / 'pooled_scores.csv').set_index('model')
    folds = pd.read_csv(OUT / 'fold_scores.csv')
    for model in MODELS:
        assert np.isfinite(pred[model]).all()
        rmse = np.sqrt(np.mean((pred[model] - pred.MONTHLY_RENT) ** 2))
        assert abs(rmse - pooled.loc[model, 'rmse_sgd']) < 1e-6
    for start, end in FOLDS:
        part = pred[pred.validation_start.eq(start)]
        assert part.month.ge(start).all() and part.month.lt(end).all()
        manifest = json.loads((OUT / f'completed_{start}.json').read_text())
        assert manifest['signature'] == signature and manifest['sha256'] == sha256(OUT / f'predictions_{start}.csv')
    selections = json.loads((OUT / 'head_selection.json').read_text())
    transfer = []
    for selection in selections:
        start, name = selection['validation_start'], selection['model']
        keep = head_data['month'] < start
        months = head_data['month'][keep]
        train, cal = chronological_head_split(months, start)
        assert int(train.sum()) == selection['training_episodes']
        assert int(cal.sum()) == selection['inner_rows']
        assert max(months.tolist()) == selection['latest_head_query_month'] < start
        baseline = head_data['base'][keep][cal]
        actual = head_data['actual'][keep][cal]
        inner_base = float(np.sqrt(np.mean((baseline - actual) ** 2)))
        scores = folds[folds.validation_start.eq(start)].set_index('model').rmse_sgd
        artifact = OUT / f'head_{start}_{name}.joblib'
        assert artifact.is_file()
        transfer.append({'validation_start': start, 'model': name,
                         'inner_base_rmse': inner_base,
                         'inner_head_rmse': selection['inner_rmse_sgd'],
                         'inner_gain_sgd': inner_base - selection['inner_rmse_sgd'],
                         'outer_base_rmse': float(scores.loc['pfn_recent']),
                         'outer_head_rmse': float(scores.loc[name]),
                         'outer_gain_sgd': float(scores.loc['pfn_recent'] - scores.loc[name]),
                         'artifact_sha256': sha256(artifact)})
    pd.DataFrame(transfer).to_csv(OUT / 'head_transfer.csv', index=False, float_format='%.8f')
    write_checks = {'status': 'passed', 'validation_rows': len(pred),
                    'historical_query_tasks': len(head_data['actual']),
                    'origins': len(episodes), 'head_fits': len(selections),
                    'checks': ['source/code hashes', 'cached array hashes', 'context before query',
                               'query/source labels and dates', 'forecast horizons',
                               'unique/disjoint outer rows', 'pooled RMSE recomputation',
                               'inner/outer chronological head partitions', 'head artifact hashes']}
    (OUT / 'verification.json').write_text(json.dumps(write_checks, indent=2), encoding='utf-8')
    results = json.loads((OUT / 'results.json').read_text())
    (OUT / 'gradient_pilot_status.json').write_text(json.dumps({
        'status': 'not_triggered' if not results['gradient_pilot_models'] else 'required',
        'qualifying_heads': results['gradient_pilot_models'],
        'reason': 'Pilot requires an embedding head to improve matched pfn_recent pooled and on at least two folds.'
    }, indent=2), encoding='utf-8')

    def figure(models, labels, reference, path, title):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout='constrained')
        colors = ['#435b79', '#d29a2d', '#178879', '#9260a3', '#c65c62', '#527fa8']
        scores = pooled.loc[models, 'rmse_sgd'].to_numpy()
        axes[0].bar(np.arange(len(models)), scores, color=colors[:len(models)], width=.65)
        axes[0].set_xticks(np.arange(len(models)), labels, fontsize=9)
        axes[0].set_ylim(0, max(scores) * 1.14)
        axes[0].set_ylabel('Pooled RMSE (SGD; lower is better)')
        for i, value in enumerate(scores):
            axes[0].text(i, value + 6, f'{value:.2f}', ha='center', fontsize=9)
        wide = folds.pivot(index='validation_start', columns='model', values='rmse_sgd').sort_index()
        for model, label, color in zip(models, labels, colors):
            if model != reference:
                axes[1].plot(wide.index, wide[model] - wide[reference], marker='o',
                             linewidth=1.8, color=color, label=label.replace('\n', ' '))
        axes[1].axhline(0, color='#555555', linewidth=1)
        axes[1].set_ylabel(f'RMSE minus {reference} (SGD)')
        axes[1].set_xlabel('Fixed forecast origin')
        axes[1].legend(frameon=False, fontsize=8)
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
        fig.suptitle(title, fontsize=13)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    figure(['catboost', 'pfn_original', 'pfn_identifiers', 'pfn_recent', 'pfn_seasonal_only'],
           ['CatBoost', 'Original', '+ IDs', '+ Recent', 'Seasonal\nonly'], 'catboost',
           OUT / 'feature_ablation.png', 'Feature ablation: one frozen TabPFN estimator, 10,000 context rows')
    figure(['pfn_recent', 'ridge_numeric', 'ridge_embedding', 'mlp_numeric', 'mlp_embedding'],
           ['Frozen\nbackbone', 'Ridge\nnumeric', 'Ridge\nembedding', 'MLP\nnumeric', 'MLP\nembedding'],
           'pfn_recent', OUT / 'head_comparison.png', 'Frozen backbone plus learned residual heads')
    report = OUT / 'report.md'
    text = report.read_text(encoding='utf-8').split('\n## Verified diagnostics\n')[0]
    gain = float(pooled.loc['pfn_original', 'rmse_sgd'] - pooled.loc['pfn_recent', 'rmse_sgd'])
    cat_gain = float(pooled.loc['catboost', 'rmse_sgd'] - pooled.loc['pfn_recent', 'rmse_sgd'])
    text += f'''
## Verified diagnostics

Recent-feature TabPFN improves matched original-feature TabPFN by **S${gain:.2f} pooled RMSE**. Its pooled gain over CatBoost is **S${cat_gain:.2f}**, but it worsens the second and third outer folds and misses the prespecified 1%/all-fold promotion criterion. CatBoost remains the selected candidate; recent-feature TabPFN has the lowest observed pooled error in this run.

Every learned residual head worsens all three folds relative to its own frozen recent-feature backbone. Embedding Ridge and MLP have particularly large errors at the first origin. This is evidence against these heads and this chronological training design; it does not establish the ceiling of other adaptations. `head_transfer.csv` compares inner-selection performance with outer performance; temporal/context distribution shift is a plausible explanation, not an isolated causal finding.

The gradient pilot gate failed, so backbone weights were **not fine-tuned**. That stage was conditional in the plan. No conclusion about fine-tuned TabPFN accuracy is claimed.

![Feature ablation](feature_ablation.png)

![Frozen residual heads](head_comparison.png)

`verification.json` records successful checks for {len(pred):,} outer validation rows, {len(head_data['actual']):,} historical query tasks, context/query chronology, inner partitions, and source/cache/prediction/head hashes. Imputation/scaling and head selection use only pre-outer data. Peak memory in timing records is PyTorch allocated memory, excluding driver reservations and display memory.

Regenerate diagnostics with `.venv/Scripts/python.exe analysis/plot_tabpfn_adaptation.py` after the experiment completes.
'''
    report.write_text(text, encoding='utf-8')
    print(json.dumps(write_checks, indent=2))


if __name__ == '__main__':
    main()

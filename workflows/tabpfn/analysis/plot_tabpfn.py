"""Plot a completed TabPFN comparison and attach figures to its report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {'catboost': 'CatBoost', 'tabpfn_spatial': 'TabPFN spatial',
          'tabpfn_history': 'TabPFN + history',
          'catboost_tabpfn_forward_blend': 'Forward blend'}
COLORS = ['#435b79', '#d69826', '#168578', '#9254a2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reports/tabpfn'))
    args = parser.parse_args()
    pooled = pd.read_csv(args.input / 'pooled_scores.csv').set_index('model')
    folds = pd.read_csv(args.input / 'fold_scores.csv')
    monthly = pd.read_csv(args.input / 'monthly_scores.csv')
    models = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    x = np.arange(len(models))
    values = pooled.loc[models, 'rmse_sgd'].to_numpy()
    axes[0].bar(x, values, color=COLORS, width=.65)
    for index, value in enumerate(values):
        axes[0].text(index, value + 8, f'{value:.2f}', ha='center', fontsize=10)
    axes[0].set_xticks(x, [LABELS[m].replace(' ', '\n', 1) for m in models])
    axes[0].set_ylim(0, max(values) * 1.15)
    axes[0].set_ylabel('Pooled RMSE (SGD; lower is better)')
    axes[0].set_title('All three disjoint validation folds')
    wide = folds.pivot(index='validation_start', columns='model', values='rmse_sgd').sort_index()
    for model, color in zip(models[1:], COLORS[1:]):
        axes[1].plot(wide.index, wide[model] - wide.catboost, marker='o',
                     color=color, linewidth=2, label=LABELS[model])
    axes[1].axhline(0, color=COLORS[0], linewidth=1)
    axes[1].set_ylabel('RMSE difference from CatBoost (SGD)')
    axes[1].set_xlabel('Fixed forecast origin')
    axes[1].set_title('Negative values improve on CatBoost')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(args.input / 'model_comparison.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.7), sharex=True, layout='constrained')
    for model, color in zip(models, COLORS):
        part = monthly.loc[monthly.model.eq(model)].sort_values('month')
        dates = pd.to_datetime(part.month)
        axes[0].plot(dates, part.rmse_sgd, label=LABELS[model], color=color, linewidth=1.7)
        axes[1].plot(dates, part.bias_sgd, label=LABELS[model], color=color, linewidth=1.7)
    axes[0].set_ylabel('Monthly RMSE (SGD)')
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].set_title('Fixed-origin forecasts; histories never update from validation rents')
    axes[1].set_ylabel('Mean predicted minus actual rent (SGD)')
    axes[1].axhline(0, color='#666666', linewidth=.8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
        for origin in ['2023-07', '2024-04']:
            ax.axvline(pd.Timestamp(origin), color='#999999', linestyle=':', linewidth=1)
    fig.savefig(args.input / 'monthly_errors.png', dpi=160)
    plt.close(fig)

    report = args.input / 'report.md'
    content = report.read_text(encoding='utf-8').split('\n## Diagnostic figures\n')[0]
    content = content.split('\n## Runtime notes\n')[0].replace('| nan ', '| — ')
    content += ('\n## Runtime notes\n\n'
                'CatBoost timings combine fitting and prediction; TabPFN reports them separately. '
                'Peak memory is PyTorch maximum allocated memory; driver reservations and display memory are additional.\n')
    environment = args.input / 'runtime_environment.json'
    if environment.exists():
        flags = json.loads(environment.read_text(encoding='utf-8'))['environment']
        content += '\nThis run uses sequential estimator execution to avoid GPU memory paging. Reproduce the environment before running the command above:\n\n```powershell\n'
        content += '\n'.join(f"$env:{name} = '{value}'" for name, value in flags.items())
        content += '\n```\n'
    if (args.input / 'memory_probe.json').exists():
        content += ('\nAn initial 2,000-row prediction batch exhausted GPU memory. '
                    'The 500-row configuration passed a memory probe without changing contexts or ensemble size; '
                    'see `memory_probe.json` and `attempt_batch2000/`.\n')
    content += ('\n## Diagnostic figures\n\n'
                '![Pooled and fold comparisons](model_comparison.png)\n\n'
                '![Monthly RMSE and mean bias](monthly_errors.png)\n\n'
                f'Regenerate with `.venv/Scripts/python.exe analysis/plot_tabpfn.py --input {args.input.as_posix()}`.\n')
    report.write_text(content, encoding='utf-8')
    print(f'Figures saved in {args.input}')


if __name__ == '__main__':
    main()

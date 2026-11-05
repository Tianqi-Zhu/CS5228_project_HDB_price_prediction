"""Interpret saved ablations and check the earlier-selected COE-only LSTM.

No architecture/window is selected on the later period. The additional COE-only
sensitivity and transfer tables are follow-up diagnostics, not prespecified tests.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

from analysis.data import sha256
from analysis.experiments.temporal_sequence_ablation import (
    OUT, TARGETS, Spec, block_interval, evaluate_spec, evaluation_cases, forecast,
    specifications, table,
)


def main():
    out = OUT
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    panel = pd.read_csv(out / 'monthly_input_panel.csv', index_col='month')
    panel.index = pd.PeriodIndex(panel.index, freq='M')
    pred = pd.read_csv(out / 'forecast_predictions.csv')
    chosen = pd.read_csv(out / 'earlier_selections.csv')
    comparisons = pd.read_csv(out / 'selected_comparisons.csv')
    spec_map = {s.name: s for s in specifications()}
    cases = evaluation_cases(panel.index)
    rows = []
    for selection in chosen[chosen.selection.str.startswith('cell:')].itertuples():
        local = pred[(pred.target == selection.target) & (pred.scenario == selection.scenario) &
                     pred.phase.eq('outer') & pred.model.eq(selection.model)]
        rows.append(dict(target=selection.target, scenario=selection.scenario,
                         method=local.method.iloc[0], group=local.group.iloc[0],
                         model=selection.model, window=int(local.window.iloc[0]),
                         development_normalized_mse=selection.development_normalized_mse,
                         outer_rmse_sgd=float(np.sqrt(np.mean((local.predicted-local.actual)**2)))))
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out / 'input_processing_rmse_sgd.csv', index=False)

    # Extend the existing row correction using saved sufficient error moments;
    # there is no need to refit the identical property models.
    monthly = pd.read_csv(out / 'row_transfer_monthly.csv')
    templates = monthly[monthly.candidate.eq('persistence')].copy()
    unique = pred.drop_duplicates(['target', 'model', 'origin', 'month']).set_index(['target', 'model', 'origin', 'month'])
    extra, extra_months, sensitivity = [], [], []
    for scenario in ('rolling', 'fixed_origin'):
        target = 'adjusted_mean_rent'
        for label, selection in [('rent_stock', 'group:rent_stock'), ('rent_coe', 'group:rent_coe'),
                                 ('rent_ridge', 'cell:ridge_lags:rent'),
                                 ('rent_lstm', 'cell:lstm:rent'),
                                 ('rent_stock_lstm', 'cell:lstm:rent_stock'),
                                 ('rent_coe_lstm', 'cell:lstm:rent_coe')]:
            c = chosen[(chosen.target == target) & (chosen.scenario == scenario) & chosen.selection.eq(selection)].iloc[0]
            part = templates[templates.scenario.eq(scenario)].copy()
            values = []
            path_cache = {}
            for row in part.itertuples():
                m = pd.Period(row.month, 'M')
                fold_origin = pd.Period('2023-06' if row.month < '2024-01' else
                                        '2023-12' if row.month < '2024-06' else '2024-05', 'M')
                origin = m-1 if scenario == 'rolling' else fold_origin
                key = (target, c.model, str(origin), row.month)
                if key in unique.index:
                    mean = float(unique.loc[key, 'predicted'])
                else:
                    if origin not in path_cache:
                        path_cache[origin] = forecast(panel, target, spec_map[c.model], panel.index.get_loc(origin))[0]
                    mean = path_cache[origin][m.ordinal-origin.ordinal-1]
                values.append(mean-panel.loc[fold_origin, target])
            part['correction'] = values
            part['candidate'], part['model'] = label, c.model
            dev = part[part.phase.eq('development')]
            denominator = np.sum(dev.n*dev.correction**2)
            gamma = float(np.clip(np.sum(dev.n*dev.correction*dev.mean_residual)/denominator, 0, 1)) if denominator else 0.
            part['gamma'] = gamma
            extra_months.append(part)
            for phase, local in part.groupby('phase'):
                shift = gamma*local.correction
                mse = local.baseline_mse-2*shift*local.mean_residual+shift**2
                baseline = float(np.sqrt(np.average(local.baseline_mse, weights=local.n)))
                corrected = float(np.sqrt(np.average(mse, weights=local.n)))
                low, high = block_interval(local.month.to_numpy(), local.baseline_mse.to_numpy(),
                                           mse.to_numpy(), weights=local.n.to_numpy())
                extra.append(dict(scenario=scenario, candidate=label, model=c.model, phase=phase,
                                  gamma=gamma, baseline_rmse_sgd=baseline, corrected_rmse_sgd=corrected,
                                  delta_rmse_sgd=corrected-baseline, delta_ci_low=low, delta_ci_high=high,
                                  improved_months=int((mse < local.baseline_mse).sum()), rows=int(local.n.sum())))
        c = chosen[(chosen.target == target) & (chosen.scenario == scenario) &
                   chosen.selection.eq('cell:lstm:rent_coe')].iloc[0]
        for label, options in [('seed43', dict(seed=43)), ('seed44', dict(seed=44)),
                               ('extra_month_delay', dict(delay=1))]:
            subset = cases[cases.phase.eq('outer') & cases.scenario.eq(scenario)]
            result = evaluate_spec(panel, target, spec_map[c.model], subset, **options)
            result.to_csv(out / f'followup_rent_coe_{scenario}_{label}.csv', index=False)
            sensitivity.append(dict(target=target, scenario=scenario, check=label, model=c.model,
                                    outer_rmse_sgd=float(np.sqrt(np.mean((result.predicted-result.actual)**2)))))
    extended = pd.concat([pd.read_csv(out / 'row_transfer_scores.csv'), pd.DataFrame(extra)], ignore_index=True)
    extended.to_csv(out / 'row_transfer_extended_scores.csv', index=False)
    pd.concat([monthly, *extra_months], ignore_index=True).to_csv(out / 'row_transfer_extended_monthly.csv', index=False)
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out / 'followup_rent_coe_sensitivity.csv', index=False)
    delay_rows = []
    original_sensitivity = pd.read_csv(out / 'sensitivity_scores.csv')
    for target in TARGETS:
        for scenario in ('rolling', 'fixed_origin'):
            subset = cases[cases.phase.eq('outer') & cases.scenario.eq(scenario)]
            result = evaluate_spec(panel, target, Spec('persistence'), subset, delay=1)
            baseline = float(np.sqrt(np.mean((result.predicted-result.actual)**2)))
            candidate = float(original_sensitivity[(original_sensitivity.target == target) &
                              (original_sensitivity.scenario == scenario) &
                              original_sensitivity.check.eq('extra_month_delay')].outer_rmse_sgd.iloc[0])
            delay_rows.append(dict(target=target, scenario=scenario,
                                   delayed_persistence_rmse_sgd=baseline,
                                   delayed_joint_lstm_rmse_sgd=candidate,
                                   delta_rmse_sgd=candidate-baseline))
    matched_delay = pd.DataFrame(delay_rows)
    matched_delay.to_csv(out / 'matched_delay_comparison.csv', index=False)

    adjusted = ablation[ablation.target.eq('adjusted_mean_rent') & ablation.method.eq('lstm')]
    pivot = adjusted.pivot(index='group', columns='scenario', values='outer_rmse_sgd').reset_index()
    rename = {'rent': 'Rent only', 'rent_stock': 'Rent + stocks', 'rent_coe': 'Rent + COE',
              'rent_stock_coe': 'Rent + stocks + COE'}
    pivot['group'] = pivot.group.map(rename)
    row_outer = extended[extended.phase.eq('outer') & ~extended.candidate.eq('rent_only')]
    lines = ['# What the temporal feature ablation shows', '',
             '**Keep rental-history persistence as the conservative default. COE is a more promising sequence '
             'input than stock returns here, but no tested neural processor establishes a stable overall win.** '
             'All 86 target/specification combinations and 8,858 forecast cases completed; 17 availability/index '
             'tests passed. The later period has been examined by previous experiments.', '',
             'The previous national-index analysis already compared Singapore mean-rent changes with a broad '
             'Singapore stock proxy: correlation 0.005 for the same month and -0.253 for the preceding month '
             '(95% month-block interval -0.517 to 0.067). It did not use LSTM.', '',
             '## Which inputs help the LSTM?', '',
             'Later June 2024–March 2025 RMSE of the property-adjusted monthly mean (SGD). Each row uses its '
             'own window chosen on earlier data. Rolling pools 1/3/6-month forecasts; fixed-origin forecasts '
             'all ten months from May 2024 information. Compare information sets within each scenario.', '',
             table(pivot), '',
             'Persistence scores **37.75 rolling / 45.39 fixed-origin**. The rent+COE LSTM uses six months of '
             'history for rolling forecasts and twelve for the fixed path; the joint LSTM uses three months. '
             'Adding stocks does not improve on the rent+COE LSTM in either scenario. This comparison permits '
             'each input set its earlier-selected window; all fixed-window results are in all_scores.csv.', '',
             'The joint LSTM improves the later fixed path from 45.39 to **31.83**, while rent+COE reaches '
             '**23.24**. However, persistence won earlier model selection, and the joint LSTM worsens rolling '
             'RMSE to **61.50**. Simple rent momentum/EWMA also perform well on the later fixed path '
             '(21.24/23.02), despite losing earlier selection. Choosing among these after inspecting this '
             'period would overstate the evidence.', '',
             '## Does it improve individual-rental prediction?', '',
             table(row_outer[['scenario', 'candidate', 'model', 'gamma', 'corrected_rmse_sgd', 'delta_rmse_sgd']]), '',
             'The uncorrected property/amenity tree scores **471.63** SGD RMSE. All correction scales above '
             'are learned from earlier errors only. Monthly-updated rent persistence scores **469.62**, '
             'beating the joint LSTM correction at **470.27**. The fixed-origin joint LSTM improves to '
             '**470.91** (a 0.73 SGD gain). These are controlled tree results; no TabPFN adapter or submission '
             'was refitted. The additional same-processor rent/stock/COE transfer rows are follow-up '
             'diagnostics added after viewing the aggregate ablation; models/windows still use earlier '
             'selection and gamma still uses earlier rental errors.', '',
             'The matched fixed-origin rent-only LSTM correction scores **469.40**, better than the '
             'joint LSTM at **470.91**. Stock-plus-rent lag regression scores **469.39**, versus '
             '**469.67** for rent-only lag regression: a small exploratory difference. The best earlier '
             'row-correction candidate in the extended comparison is rent+stock LSTM; it scores **470.06** '
             'fixed-origin and **470.18** rolling later, without beating the simpler controls in this '
             'later period. Better national-mean forecasting does not guarantee better individual-rental '
             'predictions. Gamma shrinkage can also make a biased national forecast a useful small '
             'tree-bias correction; it does not establish that stocks predict rents.', '',
             '## Robustness and practical decision', '',
             'The joint fixed-path gain persists for seeds 43/44 (25.92/22.35) and shrinks with the equal-weight '
             'stock proxy (43.64). With an extra full month of publication delay it deteriorates to **99.35**. '
             'Input availability is therefore a material part of the result. Under the same delay, adjusted-mean '
             'persistence scores 54.52 on the fixed path, outperforming the delayed joint LSTM. '
             'The raw-mean fixed path remains favorable for the LSTM under delay, so even this sensitivity '
             'depends on which rental index is used.', '', table(matched_delay), '',
             'Additional earlier-selected rent+COE LSTM sensitivity:', '', table(sensitivity), '',
             'There are **51 independent calendar months at most**, with serial dependence, rather than '
             '150,000 independent macro sequences. A tiny LSTM cannot establish a general advantage from '
             'these data. Treat COE plus short/long rent histories as the next candidate to confirm on new '
             'months, and keep stock inputs optional until they beat that control consistently. Do not '
             'replace the existing final ensemble from this study alone.', '',
             'The detailed report uses a separate horizon-normalized MSE ranking. Its averages weight each '
             'horizon relative to that horizon\'s persistence error; they can exceed 1 even when pooled SGD '
             'RMSE improves. The SGD table above avoids conflating these two objectives.', '',
             '## Files and reproduction', '',
             '- [Full methods and results](report.md)',
             '- [All input/processor RMSE comparisons](input_processing_rmse_sgd.csv)',
             '- [Extended individual-rental transfer](row_transfer_extended_scores.csv)',
             '- [Verification](verification.json)', '',
             'Run `.venv/Scripts/python.exe analysis/compare_temporal_sequence_ablation.py`, then '
             '`.venv/Scripts/python.exe analysis/summarize_temporal_sequence_ablation.py`. '
             'The second command derives these findings and performs the explicitly labeled follow-up checks. '
             'Original input files and both final submission CSVs remain unchanged.', '']
    (out / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'findings_metadata.json').write_text(json.dumps(dict(
        summary_code_sha256=sha256(Path(__file__)),
        core_results_sha256=sha256(out / 'results.json'),
        supplementary_checks_are_followup_diagnostics=True,
        selections_use_development_only=True), indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for scenario, local in adjusted.groupby('scenario'):
        local = local.set_index('group').reindex(['rent', 'rent_stock', 'rent_coe', 'rent_stock_coe'])
        positions = np.arange(4) + (-.17 if scenario == 'rolling' else .17)
        axes[0].barh(positions, local.outer_rmse_sgd, height=.3,
                     label='Monthly updates (1/3/6m)' if scenario == 'rolling' else 'Fixed May 2024 cutoff')
    axes[0].set_yticks(np.arange(4), [rename[g] for g in ('rent', 'rent_stock', 'rent_coe', 'rent_stock_coe')])
    axes[0].set(xlabel='Monthly mean RMSE (SGD)', title='LSTM input ablation; earlier-selected windows')
    axes[0].legend(loc='lower right', fontsize=8)
    actual = pred[(pred.target == 'adjusted_mean_rent') & pred.scenario.eq('fixed_origin') & pred.phase.eq('outer')]
    observed = actual.drop_duplicates('month').sort_values('month')
    axes[1].plot(pd.to_datetime(observed.month), observed.actual, color='black', label='Observed')
    for label, selection in [('Persistence', 'rent_only'), ('Rent + COE LSTM', 'cell:lstm:rent_coe'),
                             ('Joint LSTM', 'joint_lstm')]:
        name = chosen[(chosen.target == 'adjusted_mean_rent') & chosen.scenario.eq('fixed_origin') &
                      chosen.selection.eq(selection)].model.iloc[0]
        local = actual[actual.model.eq(name)].sort_values('month')
        axes[1].plot(pd.to_datetime(local.month), local.predicted, label=label)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[1].set(ylabel='Adjusted mean rent (SGD)', title='Ten months from one fixed cutoff')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(out / '02_input_ablation_and_forecast.png', dpi=160)
    plt.close(fig)
    print('Wrote findings, input RMSE table, extended transfer and COE sensitivity checks.', flush=True)


if __name__ == '__main__':
    main()

"""Readable findings and figures from completed, immutable feature ablations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.compare_tabpfn_reliability_macro import OUT, verify_sources
from analysis.experiments.temporal_sequence_ablation import table


def main():
    verify_sources()
    scores=pd.read_csv(OUT/'scores.csv')
    comparisons=pd.read_csv(OUT/'adjacent_feature_comparisons.csv')
    selection=json.loads((OUT/'selection.json').read_text())
    main=scores[scores.origin.eq('2024-06')].copy()
    selected=main[main.member.eq(selection['selected'])].iloc[0]
    long=pd.read_csv(OUT/'long_horizon_scores.csv')
    policy=json.loads((OUT/'long_horizon_selection.json').read_text())['policy']
    available=pd.read_csv(OUT/'feature_coverage.csv')
    macro=available[(available.member=='coe_increment') & available.feature.eq('NATIONAL_COE_INCREMENTAL_GROWTH')]
    extra=main[main.member.ne('baseline')]
    if (extra.delta_vs_baseline_sgd >= 0).all():
        recommendation=('All appended input sets worsen later RMSE. Retain the original77-feature inputs and current '
                        'submission. The rent-only forecast won earlier selection but failed to transfer to the later fold.')
    elif selection['selected']=='baseline':
        recommendation='No appended input set beat the control in earlier pooled selection. Keep the original77-feature inputs.'
    else:
        recommendation=(f'Earlier validation selected {selection["selected"]}; its later change is '
                        f'{selected.delta_vs_baseline_sgd:+.3f} SGD RMSE. Treat this as development evidence, subject to adapter and chronology checks.')
    lines=['# Findings: TabPFN reliability and forecast-derived COE inputs','',recommendation,'',
           'The frozen two-estimator TabPFN control, complete context rows, original rental histories and additive residual '
           'baseline are fixed. This isolates nested input additions. The numerical-best MSE-only DiagScale model is a '
           'separate architecture/view control, not the frozen baseline in this table.','',
           '## Individual-rental results','',
           'June2024–March2025, all28,995 rentals; SGD RMSE. Negative changes improve the frozen77-feature control.','',
           table(main[['member','features','rmse_sgd','delta_vs_baseline_sgd']]),'',
           'Inputs: reliability adds five missing/sparse-history/unknown-block flags; rent_forecast adds a national '
           'rent-only LSTM growth forecast and unavailable flag; coe_increment adds the difference between matched '
           'rent+COE and rent-only forecast growth. Both LSTMs use12 months,8 hidden units,100 fixed epochs and seed42.','',
           '## Does each addition help?','',
           table(comparisons[comparisons.origin.eq('2024-06')][['candidate','reference','delta_rmse_sgd','ci_low','ci_high']]),'',
           'Intervals resample paired three-month blocks. They omit fitting/selection uncertainty and multiple-search '
           'adjustment. The ten later months have been used by many earlier experiments. Neither a small favorable '
           'point estimate nor an interval is untouched confirmation.','',
           '## Earlier selection and sparse-history coverage','',
           table(pd.DataFrame([dict(member=k,pooled_earlier_rmse_sgd=v) for k,v in selection['earlier_rmse'].items()])), '',
           f'The nested July2023 choice based on September2022 alone is **{selection["nested_July2023_selected"]}**. '
           'Only earlier query targets enter these selections. The specification was informed by prior development studies; '
           'the old historical folds are not independent discoveries.', '', table(macro[['origin','split','missing','rows']]),'',
           'Missing macro forecasts are deliberately retained with an availability flag. A direct head requires three '
           'mature monthly labels. Early matrices have very few distinct forecast values; the unchanged native TabPFN '
           'preprocessor can infer a low-cardinality forecast column as categorical and warn when incoming values differ. '
           'The matrices themselves retain identical float64 appended-column dtypes. This is a real startup limitation '
           'of the tested pipeline, not evidence that a richer-history numerical forecast can never help. '
           'The unavailable flag and forecast value are added together, so their separate contributions are not isolated.','',
           'Each input is generated at the row\'s existing historical forecast origin with an additional one-month '
           'publication buffer, for both rents and COE. Future-value poisoning refits replayed exactly. '
           'Only source observations available by that cutoff train an LSTM forecast. '
           'The reference adjustment itself uses2021 only and is applied to forecasts after that year ends. '
           'Forecast growth caps at head10 if the publication buffer creates an eleven-month macro distance.','',
           '## DiagScale confirmation','']
    diag=OUT/'diagscale_confirmation.json'
    if diag.exists():
        d=json.loads(diag.read_text())
        lines += [table(pd.DataFrame([dict(inputs='original77',**d['baseline_metrics']),
                                      dict(inputs=d['selected'],**d['candidate_metrics'])])), '',
                  'This keeps the original preDecember2023 adapter fixed with one view; it does not retrain an adapter '
                  'on the augmented input matrices. No later target selected this input set.','']
    else:
        lines += ['The earlier-only gate did not choose an appended feature set, so DiagScale confirmation was not '
                  'triggered. Its recorded S$455.071 development score and final submission remain unchanged.','']
    lines += ['## Separate 11–16-month baseline check','',
              f'Earlier 11–16-month area/type mean-rent errors select **{policy}**.','',
              table(long[long.phase.eq('later_reused')][['band','policy','unique_target_months','rmse_sgd','bias_sgd']]),'',
              'Plateau10 holds the temporal forecast at its ten-month level. Damped16 extends the existing damped '
              'trend and local-premium decay. Persistence carries the latest shrunken local level. All policies fit '
              'only data available at each historical cutoff. Later cases reuse June2024–March2025 targets, and '
              'overlapping origins can reuse the same rentals. These are count-weighted area/type cell-mean errors; '
              'they must not be compared numerically with individual-rental TabPFN RMSE. '
              'Changing the baseline inside a refitted residual TabPFN remains a separate, unmeasured experiment.','',
              '## Files and reproduction','',
              '- [Full methods and all folds](report.md)',
              '- [Feature coverage](feature_coverage.csv)',
              '- [Causality replay](causality_audit.json)',
              '- [Verification](verification.json)','',
              'Run `.venv/Scripts/python.exe analysis/compare_tabpfn_reliability_macro.py --stage all`, then '
              '`.venv/Scripts/python.exe analysis/summarize_tabpfn_reliability_macro.py`. '
              'Completed inference jobs resume using checked source, feature and prediction hashes. '
              'No raw data, previous experiment artifacts, pretrained weights, existing adapters or final submission '
              'CSVs are overwritten.','']
    (OUT/'findings.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ref=comparisons[(comparisons.origin=='2024-06') & comparisons.reference.eq('baseline')]
    for i,row in enumerate(ref.itertuples()):
        axes[0].hlines(i,row.ci_low,row.ci_high,color='#267B8A',linewidth=2)
        axes[0].plot(row.delta_rmse_sgd,i,'o',color='#267B8A')
    axes[0].set_yticks(range(len(ref)),ref.candidate)
    axes[0].axvline(0,color='black',linewidth=1)
    axes[0].set(xlabel='Change in individual-rental RMSE (SGD)',title='Appended inputs vs original77; lower is better')
    local=long[long.phase.eq('later_reused') & long.band.eq('11-16')]
    axes[1].bar(local.policy,local.rmse_sgd,color='#267B8A')
    axes[1].set(ylabel='Area/type mean RMSE (SGD)',title='Separate long-horizon baseline,11–16 months')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUT/'01_feature_and_long_horizon_results.png',dpi=160)
    plt.close(fig)
    print(recommendation)


if __name__=='__main__':main()

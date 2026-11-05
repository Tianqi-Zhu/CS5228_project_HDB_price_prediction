"""Readable, auditable reporting for the declared A/B/C/D/E ensemble study."""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.data import sha256
from analysis.experiments.financial_sector_rents import table
from analysis.compare_local_economy_landscape import write_json
import analysis.compare_tabpfn_abcde as run


def report(choice,scores,weights,comparisons,monthly):
    # Reporting-only contrasts of already declared predictions. No new weights
    # or candidate family is fitted after seeing later results.
    all_predictions=pd.read_csv(run.OUT/'predictions_all_origins.csv',float_precision='round_trip')
    late_predictions=all_predictions[all_predictions.origin=='2024-06']
    extra=[]
    for candidate,reference in [('ABDE','A'),('ABCDE_equal','A'),('ABCDE_equal','BDE_existing'),
                                 ('ABCDE_equal','CDE_equal'),('ABCDE_equal','BDE_equal'),('ABCDE_equal','ABDE')]:
        if ((comparisons.candidate==candidate)&(comparisons.reference==reference)).any():continue
        extra.append(dict(candidate=candidate,reference=reference,
            delta_rmse=run.score(late_predictions,late_predictions[candidate])['rmse_sgd']-run.score(late_predictions,late_predictions[reference])['rmse_sgd'],
            **run.interval(late_predictions,candidate,reference)))
    if extra:
        comparisons=pd.concat([comparisons,pd.DataFrame(extra)],ignore_index=True)
        comparisons.to_csv(run.OUT/'paired_comparisons.csv',index=False)
    main=scores[scores.origin=='2024-06'];nested=scores[scores.origin=='2023-07']
    mw=weights[weights.origin=='2024-06'].copy()
    selected=main[main.model=='selected'].iloc[0]
    reference=main[main.model=='BDE_existing'].iloc[0]
    a=main[main.model=='A_single'].iloc[0]
    pair=comparisons[comparisons.candidate=='selected']
    early_selected=nested[nested.model=='selected'].iloc[0]
    early_bde=nested[nested.model=='BDE_existing'].iloc[0]
    early_a=nested[nested.model=='A_single'].iloc[0]
    eligible=bool(selected.rmse_sgd<min(reference.rmse_sgd,a.rmse_sgd)
                  and early_selected.rmse_sgd<min(early_bde.rmse_sgd,early_a.rmse_sgd)
                  and pair.ci_high.lt(0).all())
    # This gate concerns a separate candidate file only. Even a passing reused
    # development backtest is not fresh confirmation of future/Kaggle gains.
    write_json(run.OUT/'candidate_gate.json',dict(earlier_selected=choice['selected'],eligible_for_separate_candidate=eligible,
        both_main_references_beaten=bool(selected.rmse_sgd<min(reference.rmse_sgd,a.rmse_sgd)),
        both_nested_references_beaten=bool(early_selected.rmse_sgd<min(early_bde.rmse_sgd,early_a.rmse_sgd)),
        block_intervals_below_zero=bool(pair.ci_high.lt(0).all()),no_existing_submission_replaced=True))
    core_names=['A_single','BDE_existing','CDE','AC','ABDE','ACDE','ABCDE','selected']
    seen=main[main.model.isin(core_names)].sort_values('rmse_sgd')
    winner=main.loc[main.rmse_sgd.idxmin()]
    practical=main[main.model=='ABDE'].iloc[0]
    time_rows=[]
    for origin in run.ORIGINS[:-1]:
        meta=json.loads((run.OUT/f'oof_provenance_{origin}.json').read_text())
        time_rows.append(dict(origin=origin,adapter_reused=origin=='2022-09',context_rows=meta['context_rows'],query_rows=meta['query_rows'],
            new_adapter_fit_seconds=0. if origin=='2022-09' else meta['fit_seconds'],prediction_seconds=meta['prediction_seconds']))
    pd.DataFrame(time_rows).to_csv(run.OUT/'new_compute_cost.csv',index=False)
    error=pd.read_csv(run.OUT/'error_correlations.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    show=monthly[(monthly.origin=='2024-06')&monthly.model.isin(['ABCDE','CDE','ABDE','ABCDE_equal'])]
    for name,p in show.groupby('model'):
        ax.plot(pd.to_datetime(p.month),p.delta_vs_existing_BDE,marker='o',label=name)
    ax.axhline(0,color='black',lw=1);ax.legend(ncol=3)
    ax.set(title='Monthly prediction error relative to the existing BDE ensemble',ylabel='RMSE difference (SGD; negative is better)')
    fig.savefig(run.OUT/'01_monthly_ensemble_differences.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    im=ax.imshow(error,cmap='viridis',vmin=.99,vmax=1.)
    ax.set_xticks(range(5),run.MEMBERS);ax.set_yticks(range(5),run.MEMBERS)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{error.iloc[i,j]:.4f}',ha='center',va='center',color='black' if error.iloc[i,j]>.996 else 'white')
    fig.colorbar(im,ax=ax,label='Signed prediction-error correlation')
    ax.set_title('A–E error overlap on reused June2024–March2025 rentals')
    fig.savefig(run.OUT/'02_error_correlations.png',dpi=160);plt.close(fig)
    equality_checks=[]
    # Nested-set calibration guarantees, independently evaluated on the same
    # earlier pooled cases. This is not a claim about later RMSE ordering.
    pool=pd.concat([run.load_predictions(o) for o in run.ORIGINS[:-1]],ignore_index=True)
    for big,small in [('ABCDE','CDE'),('ABCDE','BDE'),('ABCDE','ACDE'),('ACDE','CDE'),('ABDE','BDE')]:
        wb=np.asarray(choice['main_weights'][big]);ws=np.asarray(choice['main_weights'][small])
        mb=run.score(pool,pool[run.MEMBERS].to_numpy()@wb)['rmse_sgd']**2
        ms=run.score(pool,pool[run.MEMBERS].to_numpy()@ws)['rmse_sgd']**2
        assert mb<=ms+1e-7
        equality_checks.append(dict(larger=big,subset=small,earlier_larger_mse=mb,earlier_subset_mse=ms,verified=True))
    pd.DataFrame(equality_checks).to_csv(run.OUT/'nested_simplex_checks.csv',index=False)
    lines=['# A/B/C/D/E weighted TabPFN ensembles','',
        f"**Earlier-only family selection: {choice['selected']}.** Its reused later RMSE is S${selected.rmse_sgd:.6f}, versus S${reference.rmse_sgd:.6f} for existing BDE and S${a.rmse_sgd:.6f} for A alone. "
        f"The lowest observed later score among all reported candidates is {winner.model} at S${winner.rmse_sgd:.6f}; this observed ranking does not replace the earlier selection.",'',
        f"The most promising learned-weight candidate on the later period is ABDE at S${practical.rmse_sgd:.6f}. "
        "The fixed equal-weight ABCDE sensitivity has the lowest observed score overall. The unregularized simplex fit over ABCDE collapses to about7.3%A/92.7%C and omits B,D,E, illustrating weight instability between historical periods. Neither observed winner was the family chosen by the earlier-transfer rule, so neither is promoted as an independently validated winner.",'',
        '## Analysis before new weight fitting','',
        'A=77-feature MSE-only DiagScale (one inference view); B=77-feature plain frozen Fast (two views); C=77-feature relative-history Fast (two views); D=84-feature seasonal Fast (two views); E=89-feature nearby-comparable Fast (two views). All use the same pretrained Fast backbone, original residual targets and original temporal baselines, adding the baseline exactly once before averaging.','',
        'ABCDE contains CDE and BDE as feasible subsets: zero A/B weights reproduces CDE, and zero A/C reproduces BDE. Therefore the exact nonnegative sum-one optimizer cannot have higher CALIBRATION MSE for ABCDE on identical cases. There is no corresponding guarantee on later months. Unstable extra weights can worsen transfer. Equal weighting also has no nesting guarantee, because adding a model changes every other weight.','',
        'For an existing prediction p and an added prediction a, let e=p-y and d=a-p. The blend p+alpha*d changes MSE by 2*alpha*E[e*d]+alpha^2*E[d^2]. A small positive addition helps precisely when E[e*d]<0. Individual RMSE and pairwise error correlation alone do not establish this condition for a multi-model blend.','',
        'The saved, previously inspected main scores favored A individually. Its error correlation with C (~0.9934) was lower than with B (~0.9965), motivating AC and ACDE; B might still add bias/error cancellation. This analysis motivated the declared groups A,BDE,CDE,AC,ABDE,ACDE,ABCDE before NEW weights were fitted. It was informed by already-inspected development labels, so this is retrospective exploration, not a new untouched experiment. No main-period oracle weights or exhaustive subset search are fitted.','',
        '## Earlier-only calibration and selection','',
        'The same original full-context historical windows are reused: September2022–June2023 (60,079 context,2,000 sampled queries) and July2023–April2024 (87,937 context,2,000 sampled queries). Queries are200 label-independent cases per month, weighted to each month’s rental population. All five models share exactly the same query IDs,months,labels and population weights.','',
        'September has no earlier weight labels and uses declared equal weights as cold start. July weights fit September predictions only; July’s nested transfer RMSE chooses the family, with ties favoring fewer active models. After selection is saved, June2024 weights refit each declared group on the two disjoint earlier windows, with last targetApril2024. June2024–March2025 supplies28,995 complete later cases. It does not fit weights, choose gamma/intercepts, select clipping, or choose a new family.','',
        table(pd.DataFrame(choice['nested_results']).drop(columns='weights'),['group','rmse_sgd','mae_sgd','bias_sgd','active_members']),'',
        'The exact existing five-member active-face simplex optimizer is reused; it enumerates candidate stationary faces for the convex optimum, rather than a hyperparameter grid. No negative weights, intercept or target calibration are introduced. The earlier-data nested-simplex inequality is independently checked in nested_simplex_checks.csv.','',
        '## Historical A adapters and leakage checks','',
        'The final adapter trained beforeDecember2023 cannot be backdated to predict September2022/July2023 queries. Instead, the existing one-epoch September2022 adapter is reused only after exact complete X_train,y_train and residual-baseline replay against the original historical bundle. A new July2023 adapter is fitted only on labels throughJune2023, using the fixed existing MSE-only recipe: one epoch,lr0.01,weight decay0.01,seed42,1024-row tasks,20% supervised queries,two training views and one inference view. No query evaluation, early stopping or checkpoint selection is permitted. All other backbone tensors match the frozen checkpoint digest.','',
        'Historical A forecasts cover the same ten-month origins/horizons as B–E, rather than mixing the older four-month OOF forecasts with incompatible fixed-origin predictions. Existing main A and every B–E forecast are reused unchanged. The main A adapter is the previously selected preDecember2023 model; historical A follows the fixed one-epoch recipe rather than selecting epochs on its own query windows. That recipe was developed in prior project work, so historical simulation is conditional on that already chosen algorithm.','',
        table(pd.DataFrame(time_rows),['origin','adapter_reused','context_rows','query_rows','new_adapter_fit_seconds','prediction_seconds']),'',
        '## Main reused development comparison','',
        table(main,['model','rows','rmse_sgd','mae_sgd','bias_sgd','delta_vs_existing_BDE']),'',
        'BDE is reoptimized by the unchanged original earlier-only method and independently reproduces existing BDE predictions/weights. BDE_existing also retains the old earliest-origin baseline fallback, unlike the declared equal cold starts; the two BDE versions match at July2023 andJune2024. *_equal rows are fixed equal-weight sensitivities, not selected models.','',
        '## Main weights','',table(mw,['group','A','B','C','D','E']),'',
        'Weights describe earlier calibration fit, not main-label optimization. Zero-weight members may be omitted. Costs in internal inference views are A=1 and B/C/D/E=2 each: all-positive ABCDE costs9 views, ACDE/ABDE7, CDE/BDE6, AC3. Sequential deployment peak memory is the largest member peak, not a sum across all members. Four components already have full-data test predictions; C still needs a separate full-data inference if it is retained.','',
        '## Paired comparisons','',table(comparisons,['candidate','reference','delta_rmse','ci_low','ci_high','months','block_months']),'',
        'Intervals jointly resample contiguous three-month blocks of the ten later months, retaining all paired rental observations within each sampled month. They condition on fitted models and omit weight/family-selection uncertainty and project-wide repeated holdout reuse. Do not interpret28,995 rentals as28,995 independent time observations. monthly_scores.csv and subgroup_scores.csv retain monthly,town and flat-type errors.','',
        '![](01_monthly_ensemble_differences.png)','',
        '![](02_error_correlations.png)','',
        '## Decision and limits','',
        f"Separate-candidate gate: {eligible}. It requires the earlier-selected family to beat both A and existing BDE on July nested transfer and later RMSE, with main paired-block upper95 differences below zero versus both. No existing submission is changed. Failure does not establish that all mixtures are useless; it means this declared retrospective experiment does not justify promotion under the gate.",'',
        'Existing feature and adapter hyperparameters were developed using these already-examined dates. No genuinely untouched later labeled holdout exists. Model errors are highly correlated, so small aggregate changes can be unstable even with convex weights. A Kaggle score cannot be measured without hidden test labels.','',
        '## Reproduce','',
        'Run `.venv/Scripts/python.exe analysis/compare_tabpfn_abcde.py --stage all`. It reuses validated caches and resumes completed A forecasts. `--stage select` fits/locks earlier-only weights; `--stage evaluate` recreates the report from predictions. Settings,source/code hashes,pre_analysis.txt,A adapter/context provenance,selection.json,weights.csv,all row predictions and verification.json are retained in this directory. Feature bundles/adapters are under.cache/tabpfn_abcde_ensemble/. Original scripts,reports,adapters and submissions are preserved.','']
    (run.OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    selection=json.loads((run.OUT/'selection.json').read_text())
    report(selection,*[pd.read_csv(run.OUT/name) for name in ['scores.csv','weights.csv','paired_comparisons.csv','monthly_scores.csv']])

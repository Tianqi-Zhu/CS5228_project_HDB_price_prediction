"""Nested chronological fitting of sparse static and AR(1) latent corrections."""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from analysis.compare_tabpfn_hierarchical_latent import OUT,CACHE,ROOT,HISTORICAL,OUTERS,raw_frame,load_bundle,check_sources
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable,write_json,hashlib_array
from analysis.experiments.hierarchical_latent_residual import (
    HierarchicalResidual,DynamicHierarchicalResidual,latest_available_residuals,apply_correction,weighted_metrics,
)

ALPHAS=[0.,.25,.5,.75,1.]


def temporal_evidence(pool):
    monthly=pool.groupby('month').apply(lambda p:float(np.average(p.residual,weights=p.fit_weight)),include_groups=False)
    acf=float(monthly.autocorr()) if len(monthly)>2 else 0.
    return dict(months=len(monthly),lag1_monthly_mean_acf=acf if np.isfinite(acf) else None,
        enabled=bool(len(monthly)>=8 and np.isfinite(acf) and abs(acf)>=.1),
        interpretation='Descriptive historical temporal evidence; subsequent forecast validation determines usefulness')


def read_cases():
    frames=[]
    for origin in HISTORICAL:
        path=OUT/f'historical_predictions_{origin}.csv'; meta=json.loads((OUT/f'historical_provenance_{origin}.json').read_text())
        assert sha256(path)==meta['predictions_sha256']
        assert meta['no_adapters'] and meta['gradients_prohibited'] and meta['parameters_unchanged'] and meta['n_estimators']==2
        assert meta['parameter_digest_before']==meta['parameter_digest_after']
        b=load_bundle(origin); f=pd.read_csv(path,float_precision='round_trip')
        np.testing.assert_array_equal(f.row_id,b['X_query'].index)
        np.testing.assert_array_equal(f.actual,b['y_query'])
        assert f.row_id.is_unique and len(f)==len(b['X_query']) and (f.origin==origin).all()
        assert set(f.row_id).isdisjoint(b['X_train'].index)
        assert meta['context_max_month']<origin and f.month.min()>=origin
        np.testing.assert_allclose(f.actual-f.E0,f.residual,atol=1e-10,rtol=0)
        frames.append(f)
    result=pd.concat(frames,ignore_index=True)
    assert np.isfinite(result.E0).all()
    return result


def outer_frame(cases,origin,raw):
    if origin!='2024-06': return cases.loc[cases.origin==origin].reset_index(drop=True).copy()
    replay=json.loads((OUT/'baseline_replay.json').read_text())
    assert replay['verified'] and sha256(OUT/'baseline_predictions.csv')==replay['baseline_predictions_sha256']
    f=pd.read_csv(OUT/'baseline_predictions.csv',float_precision='round_trip')
    for c,source in [('building','building'),('submarket','submarket'),('town','TOWN_CLEAN'),('flat_type','FLAT_TYPE_CLEAN')]:
        f[c]=raw.loc[f.row_id,source].to_numpy()
    f['origin']=origin; f['weight']=1.
    f['residual']=f.actual-f.E0
    f['horizon']=pd.PeriodIndex(f.month,freq='M').asi8-pd.Period(origin,'M').ordinal+1
    return f


def training_diagnostics(pool,origin):
    directory=OUT/'diagnostics'/origin; directory.mkdir(parents=True,exist_ok=True)
    monthly=[]
    for month,g in pool.groupby('month'):
        monthly.append(dict(month=month,n=len(g),mean_residual=float(np.average(g.residual,weights=g.fit_weight)),
            residual_std=float(g.residual.std()),**weighted_metrics(g.actual,g.E0,g.weight)))
    pd.DataFrame(monthly).to_csv(directory/'monthly_residuals.csv',index=False)
    subgroup=pool.groupby(['submarket','month']).agg(n=('row_id','size'),mean_residual=('residual','mean'),std=('residual','std')).reset_index()
    subgroup.to_csv(directory/'submarket_monthly_residuals.csv',index=False)
    for field in ['building','submarket']:
        stats=pool.groupby(field).agg(n=('row_id','size'),mean_residual=('residual','mean'),variance=('residual','var')).reset_index()
        stats.to_csv(directory/f'{field}_residuals.csv',index=False)
    horizon=[]
    for h,g in pool.groupby('horizon'):
        horizon.append(dict(horizon=int(h),n=len(g),**weighted_metrics(g.actual,g.E0,g.weight)))
    pd.DataFrame(horizon).to_csv(directory/'historical_horizon_residuals.csv',index=False)
    evidence=temporal_evidence(pool); write_json(directory/'temporal_evidence.json',evidence)
    return evidence


def validation_views(cases,outer):
    result=[]
    for origin in HISTORICAL:
        if origin>=outer: continue
        fit=latest_available_residuals(cases,origin)
        val=cases.loc[(cases.origin==origin)&(cases.month<outer)].reset_index(drop=True).copy()
        if len(fit)<200 or len(val)<200: continue
        assert fit.month.max()<origin<=val.month.min() and set(fit.row_id).isdisjoint(val.row_id)
        result.append((origin,fit,val,temporal_evidence(fit)))
    return result


def fit_model(pool,origin,config,dynamic):
    if dynamic:
        return DynamicHierarchicalResidual(config['block_penalty'],config['group_penalty'],config['rho'],config['state_penalty']).fit(pool,origin)
    return HierarchicalResidual(config['block_penalty'],config['group_penalty']).fit(pool,origin)


def tune(cases,outer,kind,static_config=None):
    views=validation_views(cases,outer)
    if kind=='E1': configs=[dict(block_penalty=b,group_penalty=g) for b in [25.,100.] for g in [25.,100.]]
    else:
        configs=[dict(block_penalty=static_config['block_penalty'],group_penalty=static_config['group_penalty'],rho=r,state_penalty=p)
                 for r in [0.,.5,.9] for p in [25.,100.]]
    if not views:
        config=dict(block_penalty=100.,group_penalty=100.) if kind=='E1' else dict(**static_config,rho=.5,state_penalty=100.)
        return dict(config=config,alpha=0.,cv_rmse_sgd=None,baseline_cv_rmse_sgd=None,cold_start=True,
            reason='No complete earlier chronological inner validation; correction disabled prospectively'),[],[]
    started=time.perf_counter(); records=[]; fold_records=[]
    base_sum=sum(float(np.sum(v.weight.to_numpy()*(v.E0-v.actual).to_numpy()**2)) for _,_,v,_ in views)
    total_weight=sum(float(v.weight.sum()) for _,_,v,_ in views)
    for config in configs:
        sums={alpha:0. for alpha in ALPHAS}
        for validation_origin,fit,val,evidence in views:
            dynamic=kind=='E2' and evidence['enabled']
            model=fit_model(fit,validation_origin,config,dynamic)
            correction=model.predict_components(val).total_correction.to_numpy()
            for alpha in ALPHAS:
                predicted=apply_correction(val.E0,correction,alpha)
                error=predicted-val.actual.to_numpy()
                sums[alpha]+=float(np.sum(val.weight.to_numpy()*error**2))
                fold_records.append(dict(outer_origin=outer,model=kind,validation_origin=validation_origin,
                    config=json.dumps(config,sort_keys=True),alpha=alpha,fit_rows=len(fit),validation_rows=len(val),
                    fit_last_label_month=fit.month.max(),validation_last_label_month=val.month.max(),dynamic_enabled=dynamic,
                    rmse_sgd=float(np.sqrt(np.average(error**2,weights=val.weight)))))
        for alpha in ALPHAS:
            records.append(dict(outer_origin=outer,model=kind,config=config,alpha=alpha,
                cv_rmse_sgd=float(np.sqrt(sums[alpha]/total_weight)),baseline_cv_rmse_sgd=float(np.sqrt(base_sum/total_weight))))
    def rank(r):
        c=r['config']
        return (r['cv_rmse_sgd'],r['alpha'],-c['block_penalty'],-c['group_penalty'],-c.get('state_penalty',0),c.get('rho',0))
    selected=min(records,key=rank)
    selected=dict(**selected,cold_start=False,selection_seconds=time.perf_counter()-started,
        validation_origins=[v[0] for v in views],validation_case_rows=sum(len(v[2]) for v in views),
        maximum_selection_label_month=max(v[2].month.max() for v in views))
    assert selected['maximum_selection_label_month']<outer
    return selected,records,fold_records


def choose_family(e1,e2):
    if e1['cold_start']: return 'E0'
    scores={'E0':e1['baseline_cv_rmse_sgd'],'E1':e1['cv_rmse_sgd'],'E2':e2['cv_rmse_sgd']}
    threshold=min(scores.values())*1.001
    return next(name for name in ['E0','E1','E2'] if scores[name]<=threshold)


def fit_origin(cases,origin,raw):
    directory=OUT/'origins'/origin; directory.mkdir(parents=True,exist_ok=True)
    pool=latest_available_residuals(cases,origin)
    query=outer_frame(cases,origin,raw)
    assert pool.month.max()<origin and set(pool.row_id).isdisjoint(query.row_id)
    pool.to_csv(directory/'fitting_residuals.csv',index=False,float_format='%.17g')
    evidence=training_diagnostics(pool,origin)
    e1,grid1,fold1=tune(cases,origin,'E1')
    # E1 is implemented and diagnostics are saved before any E2 state fit.
    if evidence['enabled']:
        e2,grid2,fold2=tune(cases,origin,'E2',e1['config'])
    else:
        e2=dict(e1); e2['config']=dict(e1['config']); e2['dynamic_gate_disabled']=True; grid2=[]; fold2=[]
    family=choose_family(e1,e2)
    selection=dict(origin=origin,E1=e1,E2=e2,dynamic_gate=evidence,selected_family=family,
        family_tie_policy='Prefer E0 then E1 then E2 within0.1% of best inner RMSE',
        residual_fit_rows=len(pool),residual_fit_last_label_month=pool.month.max(),residual_fit_ids_sha256=hashlib_array(pool.row_id),
        evaluation_ids_sha256=hashlib_array(query.row_id),outer_targets_used=False)
    immutable(directory/'selection.json',selection)
    pd.DataFrame([{**r,'config':json.dumps(r['config'],sort_keys=True)} for r in grid1+grid2]).to_csv(directory/'hyperparameter_scores.csv',index=False)
    pd.DataFrame(fold1+fold2).to_csv(directory/'inner_fold_scores.csv',index=False)
    context_counts=raw.loc[raw.month<origin,'building'].value_counts()
    query['context_building_count']=query.building.map(context_counts).fillna(0).astype(int)
    timings=[]; models={}
    for name,config,dynamic in [('E1',e1,False),('E2',e2,evidence['enabled'])]:
        model=fit_model(pool,origin,config['config'],dynamic)
        path=CACHE/f'{origin}_{name}.joblib'; joblib.dump(model,path,compress=3)
        started=time.perf_counter(); components=model.predict_components(query); overhead=time.perf_counter()-started
        correction=components.total_correction.to_numpy()
        query[name]=apply_correction(query.E0,correction,config['alpha'])
        query[name+'_alpha1']=apply_correction(query.E0,correction,1.)
        np.testing.assert_array_equal(apply_correction(query.E0,correction,0.),query.E0)
        for c in components: query[f'{name}_{c}']=components[c].to_numpy()
        query[f'{name}_alpha']=config['alpha']
        assert np.isfinite(query[name]).all() and np.isfinite(query[name+'_alpha1']).all()
        timings.append(dict(origin=origin,model=name,fit_rows=len(pool),fit_seconds=model.fit_seconds_,
            predict_overhead_seconds=overhead,solver_iterations=model.solver_iterations_,solver_stop=model.solver_stop_,
            system_rows=model.system_shape_[0],system_columns=model.system_shape_[1],system_nnz=model.system_nnz_,
            training_reconstruction_rmse_sgd=model.training_rmse_,model_sha256=sha256(path)))
        models[name]=model
    query['selected_prediction']=query[family]
    query.to_csv(directory/'predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(timings).to_csv(directory/'timings.csv',index=False)
    static=models['E1']
    pd.DataFrame(dict(building=static.buildings_,effect=static.coef_[1:1+static.n_buildings_],
        residual_count=static.building_counts_,effective_weight=static.building_weights_)).to_csv(directory/'building_effects.csv',index=False)
    pd.DataFrame(dict(submarket=static.groups_,effect=static.coef_[1+static.n_buildings_:1+static.n_buildings_+static.n_groups_],
        residual_count=static.group_counts_,effective_weight=static.group_weights_)).to_csv(directory/'submarket_effects.csv',index=False)
    if evidence['enabled']:
        dynamic=models['E2']; state_rows=[]
        for group,states in zip(dynamic.groups_,dynamic.states_):
            for month,value in zip(pd.period_range(dynamic.state_first_month_,dynamic.state_last_month_,freq='M'),states):
                state_rows.append(dict(submarket=group,month=str(month),state=float(value),fitted_historical=True))
            for h in range(1,11):
                state_rows.append(dict(submarket=group,month=str(pd.Period(dynamic.state_last_month_,'M')+h),
                    state=float(states[-1]*dynamic.rho**h),fitted_historical=False))
        pd.DataFrame(state_rows).to_csv(directory/'dynamic_states_and_forecasts.csv',index=False)
    print('EVALUATED',origin,[(n,weighted_metrics(query.actual,query[n],query.weight)) for n in ['E0','E1','E2']],flush=True)
    return query,selection


def persistence_diagnostics(cases,origin,config):
    pool=latest_available_residuals(cases,origin)
    midpoint='2023-05'; frames={}; effects={}; groups={}
    for period,mask in [('early',pool.month<midpoint),('late',pool.month>=midpoint)]:
        f=pool.loc[mask].copy(); frame_model=HierarchicalResidual(config['block_penalty'],config['group_penalty']).fit(f,origin)
        frames[period]=f
        effects[period]=pd.DataFrame(dict(building=frame_model.buildings_,effect=frame_model.coef_[1:1+frame_model.n_buildings_],n=frame_model.building_counts_)).set_index('building')
        c=frame_model.predict_components(f.assign(month=origin))
        f['residual_after_building']=f.residual.to_numpy()-c.building_effect.to_numpy()-frame_model.coef_[0]
        groups[period]=f.groupby('submarket').agg(effect=('residual_after_building','mean'),n=('row_id','size'))
    correlations=[]
    for name,items,minimum in [('building',effects,2),('submarket',groups,5)]:
        joined=items['early'].join(items['late'],lsuffix='_early',rsuffix='_late',how='inner')
        joined=joined.loc[(joined.n_early>=minimum)&(joined.n_late>=minimum)]
        joined.to_csv(OUT/f'{name}_persistence.csv')
        correlations.append(dict(group=name,common_groups=len(joined),minimum_cases_per_period=minimum,
            pearson=float(joined.effect_early.corr(joined.effect_late)) if len(joined)>2 else None,
            spearman=float(joined.effect_early.corr(joined.effect_late,method='spearman')) if len(joined)>2 else None,
            earlier_months='May2022-April2023',later_months='May2023-May2024',
            interpretation='Independent target-month periods; descriptive persistence, not forecast performance'))
    write_json(OUT/'persistence_diagnostics.json',correlations)


def run():
    check_sources(); replay=json.loads((OUT/'baseline_replay.json').read_text()); assert replay['verified']
    code=[Path(__file__),ROOT/'analysis/experiments/hierarchical_latent_residual.py',ROOT/'analysis/summarize_tabpfn_hierarchical_latent.py']
    immutable(OUT/'latent_plan.json',dict(code_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        static_grid=dict(block_penalty=[25.,100.],group_penalty=[25.,100.]),dynamic_grid=dict(rho=[0.,.5,.9],state_penalty=[25.,100.]),
        alpha=ALPHAS,outer_origins=OUTERS,cold_start_alpha=0.,family_tolerance=.001,
        solver='Sparse augmented least squares, LSMR tolerance1e-10; no dense covariance or design',
        E2_prior='lambda_d*((1-rho^2)*d_first^2 + sum_t(d_t-rho*d_previous)^2)',
        no_outer_target_selection=True))
    cases=read_cases(); cases.to_csv(OUT/'all_historical_forecast_cases.csv',index=False,float_format='%.17g')
    raw=raw_frame(); predictions=[]; selections={}
    for origin in OUTERS:
        f,s=fit_origin(cases,origin,raw); predictions.append(f); selections[origin]=s
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions_all_origins.csv',index=False,float_format='%.17g')
    persistence_diagnostics(cases,'2024-06',selections['2024-06']['E1']['config'])
    for path,h in json.loads((OUT/'latent_plan.json').read_text())['code_sha256'].items(): assert sha256(ROOT/path)==h
    from analysis.summarize_tabpfn_hierarchical_latent import summarize
    summarize()


if __name__=='__main__': run()

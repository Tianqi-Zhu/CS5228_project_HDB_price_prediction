"""Train support-aware residual stacking from pre-origin out-of-time forecasts.

June fitting may use April/May forecasts, whose base models were fit before
April. June-or-later rents are never used to fit the June meta-model.
All periods and component selection are reused development evidence.
"""
from __future__ import annotations
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from analysis.experiments.frontier_search import ROOT,score,write_json,month_number
from analysis.experiments.third_wave_training import make_data,adjusted_baseline
from analysis.experiments.fourth_wave_ensemble import old_best

OUT=ROOT/'output/experiment_runs/round_08_feature_pruning_seeds'
ART=ROOT/'output/models/round_08_feature_pruning_seeds'
SPECS=[{'name':f'ridge_{features}_{alpha}','family':'ridge','features':features,'alpha':alpha}
    for features in ('predictions','support','typed') for alpha in (10.,100.,1000.)]
SPECS += [{'name':f'convex_intercept_{ridge}','family':'convex','ridge':ridge,'features':'predictions'} for ridge in (0.,.001)]
SPECS += [{'name':f'lgb_support_{rounds}','family':'lgb','features':'support','rounds':rounds} for rounds in (100,300)]


def registry():
    result={}
    best=json.loads((ROOT/'output/experiment_runs/00_report_and_final/best_ensemble.json').read_text(encoding='utf-8'))
    for member in best['leaf_members']:
        result[member['name']]={stage:str((ROOT/'output/models'/member['experiment_root']/member['kind']/stage/member['artifact_name']/'predictions.csv').relative_to(ROOT)) for stage in ('screen','outer','june')}
    for family in ('lgb','cat','xgb'):
        for kind in ('base','shape','building','combined','no36','only24'):
            name=f'{family}_{kind}'
            result['fifth/'+name]={stage:f'output/models/round_05_feature_interactions/matrix/{stage}/{name}/predictions.csv' for stage in ('screen','outer','june')}
    sixth=json.loads((ROOT/'output/experiment_runs/round_06_deep_models/ensemble_selection.json').read_text(encoding='utf-8'))
    for name in sixth['representatives'].values():
        result['sixth/'+name]={stage:f'output/models/round_06_deep_models/{stage}/{name}/predictions.csv' for stage in ('screen','outer','june')}
    seventh=json.loads((ROOT/'output/experiment_runs/round_07_pretrained_transformers/ensemble_selection.json').read_text(encoding='utf-8'))
    names=(set(seventh['representatives'].values())|set(seventh['weights']))-{'previous_history','previous_deep'}
    for name in sorted(names):
        result['seventh/'+name]={stage:f'output/models/round_07_pretrained_transformers/{stage}/{name}/predictions.csv' for stage in ('screen','outer','june')}
    for paths in result.values():
        for path in paths.values():
            assert (ROOT/path).exists(),path
    return result


def data(stage,members):
    bundle=make_data(stage);baseline=old_best(stage,bundle)
    frame=pd.DataFrame({'PRED_BASE':baseline/1000.})
    for i,(name,paths) in enumerate(members.items()):
        part=pd.read_csv(ROOT/paths[stage]).sort_values('source_row')
        np.testing.assert_array_equal(part.source_row,bundle['row_val'])
        np.testing.assert_array_equal(part.actual,bundle['y_val'])
        frame[f'D_{i:02}']=(part.prediction.to_numpy(float)-baseline)/1000.
    ann=bundle['ann_val'];extra=bundle['extra_val'];raw=bundle['x_val']
    for lag in (12,24,36):
        n=ann[f'ANN_BLOCK_{lag}_R1_N'].to_numpy(float)
        frame[f'SUPPORT{lag}']=(n>0).astype(float)
        frame[f'LOG_N{lag}']=np.log1p(n)/5.
    horizon=month_number(pd.Series(bundle['date_val']))-bundle['origin']+1
    frame['HORIZON']=np.minimum(horizon,6)/6.
    base,_=adjusted_baseline(ann,extra,raw)
    frame['BASE_VS_COHORT']=(baseline-base)/1000.
    frame['AREA']=raw.FLOOR_AREA_SQM.to_numpy(float)/100.
    frame['PAST_LEVEL']=extra.MARKET_3M.to_numpy(float)/1000.
    for column in ('FLAT_TYPE_CLEAN','TOWN_CLEAN'):
        frame[column]=raw[column].astype('string').fillna('__MISSING__').to_numpy(str)
    return {'x':frame,'baseline':baseline,'actual':bundle['y_val'],'row':bundle['row_val'],
        'month':np.asarray(bundle['date_val']),'hashes':bundle['hashes']}


class MetaFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,kind='predictions'):
        self.kind=kind
    def fit(self,x,y=None):
        self.deltas_=[column for column in x if column.startswith('D_')]
        self.types_=sorted(x.FLAT_TYPE_CLEAN.unique()) if self.kind=='typed' else []
        return self
    def transform(self,x):
        if self.kind=='predictions':
            return x[['PRED_BASE']+self.deltas_].copy()
        result=x.copy()
        for support in ('SUPPORT24','SUPPORT36'):
            for column in self.deltas_:
                result[f'{column}_{support}']=x[column]*x[support]
        if self.kind=='typed':
            pieces=[result]
            for i,level in enumerate(self.types_):
                mask=x.FLAT_TYPE_CLEAN.eq(level).to_numpy(float)
                pieces.append(pd.DataFrame({f'{column}_TYPE{i}':x[column].to_numpy()*mask for column in self.deltas_},index=x.index))
            result=pd.concat(pieces,axis=1)
        return result


class ConvexResidual(BaseEstimator):
    def __init__(self,ridge=.001):
        self.ridge=ridge
    def fit(self,x,y,sample_weight=None):
        self.columns_=[column for column in x if column.startswith('D_')]
        matrix=x[self.columns_].to_numpy(float)
        matrix=np.column_stack([matrix,np.ones(len(x))])
        weights=np.ones(len(x))/len(x) if sample_weight is None else sample_weight/np.sum(sample_weight)
        def objective(w):
            error=matrix@w-y
            return np.sum(weights*error**2)+self.ridge*(w@w),2*matrix.T@(weights*error)+2*self.ridge*w
        count=len(self.columns_)
        fitted=minimize(objective,np.zeros(count+1),jac=True,method='SLSQP',
            bounds=[(0.,.35)]*count+[(-.15,.15)],
            constraints=[{'type':'ineq','fun':lambda w:.75-w[:-1].sum(),
                'jac':lambda w:np.r_[-np.ones(count),0.]}],
            options={'maxiter':1500,'ftol':1e-12})
        if not fitted.success:
            raise RuntimeError(fitted.message)
        self.coef_=fitted.x[:-1];self.intercept_=float(fitted.x[-1]);return self
    def predict(self,x):
        return x[self.columns_].to_numpy(float)@self.coef_+self.intercept_


def fit(spec,x,target,weights):
    # Keep regularization comparable across origins with unequal row counts.
    # Pool weights encode relative origin importance, not the loss scale.
    weights=np.asarray(weights,dtype=float)
    if len(weights)!=len(x) or not np.all(np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Expected finite nonnegative sample weights with positive total.')
    weights=weights*(len(x)/weights.sum())
    if spec['family']=='convex':
        return ConvexResidual(spec['ridge']).fit(x,target,sample_weight=weights)
    preprocess=ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median',keep_empty_features=True),make_column_selector(dtype_include=np.number)),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),make_column_selector(dtype_exclude=np.number))])
    if spec['family']=='ridge':
        estimator=Ridge(alpha=spec['alpha'])
    else:
        from lightgbm import LGBMRegressor
        estimator=LGBMRegressor(n_estimators=spec['rounds'],num_leaves=7,min_child_samples=200,
            learning_rate=.03,reg_lambda=20.,random_state=42,n_jobs=4,verbosity=-1)
    model=Pipeline([('features',MetaFeatures(spec['features'])),('preprocess',preprocess),('model',estimator)])
    model.fit(x,target,model__sample_weight=weights)
    return model


def forecast(model,x,baseline):
    # Bound a learned correction; this cap is fixed before later evaluation.
    return baseline+np.clip(model.predict(x)*1000.,-250.,250.)


def pre_origin_pool(earlier,annual,origin):
    pieces=[(earlier,np.asarray(earlier['month'])<origin)]
    if origin>'2024-04':
        pieces.append((annual,np.asarray(annual['month'])<origin))
    xs=[];ys=[];weights=[];months=[]
    for item,mask in pieces:
        assert np.all(np.asarray(item['month'])[mask]<origin)
        n=int(np.sum(mask))
        xs.append(item['x'].loc[mask]);ys.append((item['actual'][mask]-item['baseline'][mask])/1000.)
        weights.append(np.full(n,1./len(pieces)/n));months.extend(item['month'][mask])
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),np.concatenate(weights),months


def run():
    threadpool_limits(4)
    members=registry()
    write_json(OUT/'plan.json',{'specs':SPECS,'registry':members,
        'selection_train':['2023-10','2024-01'],'selection_validation':['2024-02','2024-03'],
        'meta_fit_outer':['2023-10','2024-03'],'meta_fit_june':[['2023-10','2024-03'],['2024-04','2024-05']],
        'origin_weighting':'Equal total weight per preceding forecast origin; no later labels.',
        'correction_cap_sgd':250.,'notes':'Base configuration/periods were reused development evidence. Meta tuning split is not a new independent test.'})
    earlier=data('screen',members)
    train=earlier['month']<'2024-02';tune=~train
    selection=[]
    for spec in SPECS:
        path=OUT/'screen'/spec['name']/'results.json'
        if path.exists():
            selection.append(json.loads(path.read_text(encoding='utf-8')));continue
        started=time.perf_counter()
        target=(earlier['actual']-earlier['baseline'])/1000.
        model=fit(spec,earlier['x'].loc[train],target[train],np.ones(np.sum(train)))
        prediction=forecast(model,earlier['x'].loc[tune],earlier['baseline'][tune])
        destination=ART/'screen'/spec['name'];destination.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,destination/'model.joblib',compress=1)
        pd.DataFrame({'source_row':earlier['row'][tune],'month':earlier['month'][tune],
            'actual':earlier['actual'][tune],'prediction':prediction}).to_csv(destination/'predictions.csv',index=False)
        record={'spec':spec,'stage':'screen','rows':len(prediction),**score(earlier['actual'][tune],prediction),
            'baseline_scores':score(earlier['actual'][tune],earlier['baseline'][tune]),'seconds':time.perf_counter()-started}
        write_json(path,record);selection.append(record)
        print('META SCREEN',spec['name'],record['rmse_sgd'],flush=True)
    selection_path=OUT/'selection.json'
    if not selection_path.exists():
        best=min(selection,key=lambda item:item['rmse_sgd'])
        write_json(selection_path,{'selected':best['spec']['name'],'score':best,
            'decision_before_later_evaluation':True,'query_rents_used_for_later_fit':False})
    annual=data('outer',members)
    for stage,query,origin in [('outer',annual,'2024-04'),('june',None,'2024-06')]:
        if query is None:
            query=data(stage,members)
        x,target,weights,months=pre_origin_pool(earlier,annual,origin)
        for spec in SPECS:
            path=OUT/stage/spec['name']/'results.json'
            if path.exists(): continue
            started=time.perf_counter();model=fit(spec,x,target,weights)
            prediction=forecast(model,query['x'],query['baseline'])
            destination=ART/stage/spec['name'];destination.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,destination/'model.joblib',compress=1)
            pd.DataFrame({'source_row':query['row'],'month':query['month'],'actual':query['actual'],
                'prediction':prediction}).to_csv(destination/'predictions.csv',index=False)
            record={'spec':spec,'stage':stage,'rows':len(prediction),'meta_fit_rows':len(x),
                'meta_fit_last_month':max(months),'origin':origin,**score(query['actual'],prediction),
                'baseline_scores':score(query['actual'],query['baseline']),'seconds':time.perf_counter()-started,
                'source_sha256':query['hashes']}
            write_json(path,record)
            print('META RESULT',stage,spec['name'],record['rmse_sgd'],flush=True)


if __name__=='__main__':
    from analysis.experiments.eighth_wave_meta import run as run_registered
    run_registered()

"""Learn small changes to blending from pre-origin model forecasts only."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data

OUT=ROOT/'output/experiment_runs/round_09_grouped_blending'
ART=ROOT/'output/models/round_09_grouped_blending'
SPECS=[]
for objective in ('mse','month_centered'):
    for ridge in (.0003,.003,.03):
        SPECS.append({'name':f'global_{objective}_{ridge}','objective':objective,
            'ridge':ridge,'gate':'none','update':True})
        SPECS.append({'name':f'frozen_{objective}_{ridge}','objective':objective,
            'ridge':ridge,'gate':'none','update':False})
for gate in ('support24','support36','type','disagreement'):
    for objective in ('mse','month_centered'):
        SPECS.append({'name':f'{gate}_{objective}','objective':objective,
            'ridge':.003,'group_ridge':.03,'gate':gate,'update':True})
for ridge in (.003,.03):
    SPECS.append({'name':f'signed_{ridge}','objective':'mse','ridge':ridge,
        'gate':'none','update':True,'signed':True})
FOLLOWUP=[]
for gate,ridges in [('support24',(.003,.01,.1)),('support_pair',(.003,.03,.1)),
    ('type',(.01,.1)),('disagreement',(.003,))]:
    for group_ridge in ridges:
        FOLLOWUP.append({'name':f'followup_{gate}_centered_{group_ridge}',
            'objective':'month_centered','ridge':.003,'group_ridge':group_ridge,
            'gate':gate,'update':True})


def read_prediction(path,bundle):
    table=pd.read_csv(path).sort_values('source_row')
    np.testing.assert_array_equal(table.source_row,bundle['row_val'])
    np.testing.assert_array_equal(table.actual,bundle['y_val'])
    return table.prediction.to_numpy(float)


def data(stage):
    bundle=make_data(stage)
    specification=json.loads((ROOT/'output/experiment_runs/00_report_and_final/best_ensemble.json').read_text(encoding='utf-8'))
    names=[];values=[];prior=[]
    for member in specification['leaf_members']:
        names.append(member['name']);prior.append(member['weight'])
        path=ROOT/'output/models'/member['experiment_root']/member['kind']/stage/member['artifact_name']/'predictions.csv'
        values.append(read_prediction(path,bundle))
    original_ft=next(i for i,name in enumerate(names) if name.endswith('/ft_transformer'))
    newft=[]
    for seed in (42,43,44,45):
        if stage=='screen' and seed==42:
            newft.append(values[original_ft])
        else:
            name=f'ft_no36_seed{seed}' if seed!=42 else 'ft_no36'
            newft.append(read_prediction(ROOT/'output/models/round_08_feature_pruning_seeds'/stage/name/'predictions.csv',bundle))
    names.append('ft_no36_seedmean');values.append(np.mean(newft,axis=0))
    prior.append(prior[original_ft]);prior[original_ft]=0.
    matrix=np.column_stack(values)
    context=pd.DataFrame({'support24':(bundle['ann_val'].ANN_BLOCK_24_R1_N.to_numpy()>0).astype(str),
        'support36':(bundle['ann_val'].ANN_BLOCK_36_R1_N.to_numpy()>0).astype(str),
        'type':bundle['x_val'].FLAT_TYPE_CLEAN.astype(str).to_numpy(),
        'disagreement':np.std(newft,axis=0)})
    context['support_pair']=context.support24+'|'+context.support36
    return {'matrix':matrix,'names':names,'prior':np.asarray(prior),'context':context,
        'actual':np.asarray(bundle['y_val']),'row':bundle['row_val'],'month':np.asarray(bundle['date_val']),
        'hashes':bundle['hashes']}


def subset(item,mask):
    return item|{key:(item[key].loc[mask].reset_index(drop=True) if key=='context' else item[key][mask])
        for key in ('matrix','actual','row','month','context')}


def fitting_pool(earlier,annual,origin,update=True):
    first=subset(earlier,earlier['month']<origin)
    pieces=[first]
    if update and origin>'2024-04':
        pieces.append(subset(annual,annual['month']<origin))
    for part in pieces:
        if not len(part['actual']) or not np.all(part['month']<origin):
            raise ValueError('Meta fitting requires nonempty strictly pre-origin forecasts.')
    result=first|{key:np.concatenate([p[key] for p in pieces]) for key in ('matrix','actual','row','month')}
    result['context']=pd.concat([p['context'] for p in pieces],ignore_index=True)
    result['weights']=np.concatenate([np.full(len(p['actual']),1./len(pieces)/len(p['actual'])) for p in pieces])
    return result


def solve(matrix,target,months,weights,prior,spec):
    errors=(np.asarray(matrix)-np.asarray(target)[:,None])/1000.
    weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
    if spec['objective']=='month_centered':
        errors=errors.copy()
        for month in np.unique(months):
            mask=months==month
            errors[mask]-=np.average(errors[mask],axis=0,weights=weights[mask])
    covariance=errors.T@(weights[:,None]*errors)
    ridge=spec['ridge'];prior=np.asarray(prior)
    def objective(w):
        delta=w-prior
        return w@covariance@w+ridge*(delta@delta),2*covariance@w+2*ridge*delta
    lower=-.05 if spec.get('signed') else 0.
    fitted=minimize(objective,prior,jac=True,method='SLSQP',bounds=[(lower,.6)]*len(prior),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(len(w))}],
        options={'maxiter':1000,'ftol':1e-12})
    if not fitted.success:raise RuntimeError(fitted.message)
    return fitted.x


def group_labels(context,gate,threshold=None):
    if gate=='none':return np.full(len(context),'all')
    if gate=='disagreement':return np.where(context[gate].to_numpy()<=threshold,'low','high')
    return context[gate].to_numpy(str)


def fit(item,spec):
    weights=item.get('weights',np.ones(len(item['actual'])))
    global_weights=solve(item['matrix'],item['actual'],item['month'],weights,item['prior'],spec)
    threshold=float(item['context'].disagreement.median()) if spec['gate']=='disagreement' else None
    labels=group_labels(item['context'],spec['gate'],threshold)
    groups={}
    if spec['gate']!='none':
        for level in sorted(set(labels)):
            mask=labels==level
            if np.sum(mask)<500:continue
            group_spec=spec|{'ridge':spec['group_ridge']}
            learned=solve(item['matrix'][mask],item['actual'][mask],item['month'][mask],weights[mask],global_weights,group_spec)
            groups[str(level)]={'weights':learned.tolist(),'rows':int(np.sum(mask))}
    return {'spec':spec,'global_weights':global_weights.tolist(),'group_weights':groups,
        'threshold':threshold,'names':item['names'],'fit_rows':len(item['actual']),
        'fit_months':sorted(set(item['month'])),'fit_last_month':str(max(item['month']))}


def predict(model,item):
    if model['names']!=item['names']:raise ValueError('Component order mismatch')
    prediction=item['matrix']@np.asarray(model['global_weights'])
    labels=group_labels(item['context'],model['spec']['gate'],model['threshold'])
    for level,group in model['group_weights'].items():
        mask=labels==level
        prediction[mask]=item['matrix'][mask]@np.asarray(group['weights'])
    return prediction


def persist(stage,spec,item,model,prediction):
    directory=ART/stage/spec['name'];directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'model.json',model)
    restored=json.loads((directory/'model.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(predict(restored,item),prediction,atol=1e-9,rtol=0)
    pd.DataFrame({'source_row':item['row'],'month':item['month'],'actual':item['actual'],
        'prediction':prediction}).to_csv(directory/'predictions.csv',index=False)
    record={'stage':stage,'name':spec['name'],'spec':spec,'rows':len(prediction),
        'fit_last_month':model['fit_last_month'],'fit_rows':model['fit_rows'],
        'groups':len(model['group_weights']),**score(item['actual'],prediction)}
    write_json(OUT/stage/spec['name']/'results.json',record)
    return record


def run():
    threadpool_limits(4)
    write_json(OUT/'plan.json',{'specs':SPECS,'component_count':11,
        'prior':'Original ten-leaf weights with FT replaced by no36 four-seed mean.',
        'selection_train':['2023-10','2024-01'],'selection_validation':['2024-02','2024-03'],
        'later_fit':'Outer uses pre-April screen forecasts; June optionally adds April/May outer forecasts with equal origin mass.',
        'families':['convex anchored weights','monthly centered error covariance','support/type/disagreement gates','bounded signed weights'],
        'no_new_base_model_fit':True,'no_query_label_fitting':True,
        'note':'All base models, windows, and component selection were reused development evidence. Later scores are not independent tests.'})
    earlier=data('screen');train=earlier['month']<'2024-02'
    training=subset(earlier,train);tuning=subset(earlier,~train)
    rows=[]
    for spec in SPECS:
        model=fit(training,spec);pred=predict(model,tuning)
        rows.append(persist('screen',spec,tuning,model,pred))
    selected=min(rows,key=lambda r:r['rmse_sgd'])
    path=OUT/'selection.json'
    if not path.exists():write_json(path,{'selected':selected,'decision_before_later_evaluation':True})
    print('SELECTED',selected['name'],selected['rmse_sgd'],flush=True)
    annual=data('outer')
    for stage,origin in [('outer','2024-04'),('june','2024-06')]:
        query=annual if stage=='outer' else data(stage)
        for name,prior in [('previous_history',np.r_[earlier['prior'][:-1],0.]),('previous_ft_seedmean',earlier['prior'])]:
            if name=='previous_history':
                prior=prior.copy();index=next(i for i,n in enumerate(query['names']) if n.endswith('/ft_transformer'))
                prior[index]=query['prior'][-1]
            pred=query['matrix']@prior
            rows.append({'stage':stage,'name':name,'rows':len(pred),**score(query['actual'],pred)})
        for spec in SPECS:
            pool=fitting_pool(earlier,annual,origin,spec['update'])
            model=fit(pool,spec);assert model['fit_last_month']<origin
            pred=predict(model,query)
            result=persist(stage,spec,query,model,pred);rows.append(result)
            print('BLEND',stage,spec['name'],result['rmse_sgd'],flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    pivot=table[table.stage!='screen'].pivot(index='name',columns='stage',values='rmse_sgd').sort_values('june')
    pivot.to_csv(OUT/'comparison.csv');print(pivot.to_string(),flush=True)


def followup():
    threadpool_limits(4)
    write_json(OUT/'followup_plan.json',{'specs':FOLLOWUP,
        'note':'Additional group shrinkage and joint-support exploration after first 22 configurations. Later periods reused; not an independent confirmation.'})
    earlier=data('screen');annual=data('outer');train=earlier['month']<'2024-02'
    rows=pd.read_csv(OUT/'metrics.csv').to_dict('records')
    for stage,origin in [('screen','2024-02'),('outer','2024-04'),('june','2024-06')]:
        query=subset(earlier,~train) if stage=='screen' else annual if stage=='outer' else data(stage)
        for spec in FOLLOWUP:
            pool=subset(earlier,train) if stage=='screen' else fitting_pool(earlier,annual,origin)
            model=fit(pool,spec);pred=predict(model,query)
            record=persist(stage,spec,query,model,pred);rows.append(record)
            print('FOLLOWUP',stage,spec['name'],record['rmse_sgd'],flush=True)
    table=pd.DataFrame(rows).drop_duplicates(['stage','name'],keep='last');table.to_csv(OUT/'metrics.csv',index=False)
    pivot=table[table.stage!='screen'].pivot(index='name',columns='stage',values='rmse_sgd').sort_values('june')
    pivot.to_csv(OUT/'comparison.csv');print(pivot.to_string(),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup',action='store_true')
    args=parser.parse_args()
    (followup if args.followup else run)()

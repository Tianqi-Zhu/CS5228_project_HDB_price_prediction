"""One predeclared time-adaptive blend, selected before the June forecast origin.

Equal weight to mean loss on Oct-Mar earlier forecasts and Apr-May forecasts
from the April origin. Fixed ridge 0.002; no tuning using June-or-later labels.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import score,write_json
from analysis.experiments.third_wave_training import OUT,ART,make_data
from analysis.experiments.third_wave_cross_ensemble import baseline


def candidates():
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    items={'round_03_tree_transformer_ensemble':{'kind':'round_03_tree_transformer_ensemble'},'round_02_anniversary_history':{'kind':'round_02_anniversary_history'}}
    items.update({name:item for name,item in selection['members'].items() if item['kind']!='round_02_anniversary_history'})
    for name in ('xgb3','catgpu8_control'):
        items['models/'+name]={'kind':'models','name':name}
    cross=json.loads((OUT/'cross_type_selection.json').read_text())
    for name in cross['weights']:
        if name!='round_03_tree_transformer_ensemble':
            items['cross_type/'+name]={'kind':'cross_type','name':name}
    return items


def values(stage,bundle,items):
    result=[]
    for item in items.values():
        if item['kind']=='round_03_tree_transformer_ensemble':
            prediction=baseline(stage,bundle)
        elif item['kind']=='round_02_anniversary_history':
            prediction=bundle['fallback']
        else:
            frame=pd.read_csv(ART/item['kind']/stage/item['name']/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            np.testing.assert_array_equal(frame.actual,bundle['y_val'])
            prediction=frame.prediction.to_numpy(float)
        result.append(prediction)
    return np.column_stack(result)


def select():
    path=OUT/'adaptive_selection.json'
    if path.exists():
        return
    items=candidates()
    earlier=make_data('screen')
    recent=make_data('outer')
    mask=np.asarray(recent['date_val'])<'2024-06'
    if np.any(np.asarray(recent['date_val'])[mask]>='2024-06'):
        raise AssertionError('Post-origin meta-training labels')
    x=values('screen',earlier,items)/1000.
    z=values('outer',recent,items)[mask]/1000.
    y=earlier['y_val']/1000.
    target=recent['y_val'][mask]/1000.
    # Query labels from June onward are never passed to the optimizer.
    initial=np.zeros(len(items));initial[0]=1.
    ridge=.002
    def objective(w):
        old_error=x@w-y
        new_error=z@w-target
        shift=w-initial
        loss=.5*np.mean(old_error**2)+.5*np.mean(new_error**2)+ridge*(shift@shift)
        gradient=x.T@old_error/len(y)+z.T@new_error/len(target)+2*ridge*shift
        return loss,gradient
    fitted=minimize(objective,initial,jac=True,method='SLSQP',
        bounds=[(0,1.),(0,1.)]+[(0,.35)]*(len(items)-2),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1000,'ftol':1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    weights=np.where(fitted.x>1e-5,fitted.x,0.);weights/=weights.sum()
    result={'members':{name:item|{'weight':float(weight)} for (name,item),weight in zip(items.items(),weights) if weight>0},
        'selection_ranges':[['2023-10','2024-03'],['2024-04','2024-05']],
        'forecast_origin':'2024-06','ridge':ridge,'recent_loss_fraction':.5,
        'earlier_scores':score(y*1000,x@weights*1000),
        'april_may_scores':score(target*1000,z@weights*1000),
        'meta_fit_rows':len(y)+len(target),
        'note':'A single predeclared adaptive variant, with weights selected only from pre-June forecasts. Structure discovery occurred during development; this is not an independent test.'}
    write_json(path,result)
    print('ADAPTIVE LOCK',json.dumps(result,indent=2),flush=True)


def evaluate():
    selection=json.loads((OUT/'adaptive_selection.json').read_text())
    bundle=make_data('june')
    matrix=values('june',bundle,selection['members'])
    weights=np.array([item['weight'] for item in selection['members'].values()])
    prediction=matrix@weights
    path=ART/'adaptive'/'june'
    path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':prediction}).to_csv(path/'predictions.csv',index=False)
    result={'stage':'june','scores':score(bundle['y_val'],prediction),
        'previous_third_wave':score(bundle['y_val'],baseline('june',bundle)),
        'rows':len(prediction),'selection_ranges':selection['selection_ranges']}
    write_json(OUT/'adaptive_june.json',result)
    print('ADAPTIVE RESULT',result,flush=True)


if __name__=='__main__':
    select()
    evaluate()

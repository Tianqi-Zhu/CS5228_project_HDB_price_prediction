"""Freeze a third-wave convex blend using only pre-April development labels."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import OUT,ART,make_data

MODULES={'training':'analysis.experiments.third_wave_training',
    'models':'analysis.experiments.third_wave_models','features':'analysis.experiments.third_wave_features',
    'mixture':'analysis.experiments.third_wave_mixture'}


def select():
    destination=OUT/'ensemble_selection.json'
    if destination.exists():
        print('Third-wave selection already locked',flush=True)
        return
    bundle=make_data('screen')
    registered={'round_02_anniversary_history':{'kind':'round_02_anniversary_history'}}
    values=[bundle['fallback']]
    diagnostics=[]
    for kind,module in MODULES.items():
        for path in sorted((OUT/kind/'screen').glob('*/results.json')):
            result=json.loads(path.read_text())
            name=result['spec']['name']
            predicted=pd.read_csv(ART/kind/'screen'/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(predicted.source_row,bundle['row_val'])
            np.testing.assert_array_equal(predicted.actual,bundle['y_val'])
            diagnostics.append({'kind':kind,'name':name,**score(predicted.actual,predicted.prediction)})
            if result['rmse_sgd']>535.:
                continue
            registered[f'{kind}/{name}']={'kind':kind,'name':name,'module':module}
            values.append(predicted.prediction.to_numpy(float))
    matrix=np.column_stack(values)/1000.
    actual=bundle['y_val']/1000.
    def objective(weights):
        error=matrix@weights-actual
        return np.mean(error**2),2*matrix.T@error/len(error)
    initial=np.zeros(len(registered))
    initial[0]=1.
    result=minimize(objective,initial,jac=True,method='SLSQP',
        bounds=[(0,1.)]+[(0,.35)]*(len(registered)-1),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1500,'ftol':1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    weights=np.where(result.x>1e-5,result.x,0.)
    weights/=weights.sum()
    members={key:item|{'weight':float(weight)} for (key,item),weight in zip(registered.items(),weights) if weight>0}
    locked={'members':members,'selection_months':['2023-10','2024-03'],
        'earlier_scores':score(bundle['y_val'],matrix@weights*1000),
        'previous_scores':score(bundle['y_val'],bundle['fallback']),
        'diagnostics':diagnostics,'note':'No intercept, nonnegative unit-sum weights. Each new member is capped at 0.35; old blend may retain all weight. Outer periods have already been viewed during earlier development.'}
    write_json(destination,locked)
    print(json.dumps({k:v for k,v in locked.items() if k!='diagnostics'},indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    bundle=make_data(stage)
    # Group refits to avoid preparing the same origin repeatedly.
    for kind,module in MODULES.items():
        names=[item['name'] for item in selection['members'].values() if item['kind']==kind and
            not (ART/kind/stage/item['name']/'predictions.csv').exists()]
        if names:
            subprocess.run([sys.executable,'-u','-m',module,stage,'--names',*names],check=True)
    prediction=np.zeros(len(bundle['y_val']))
    components=[]
    for name,item in selection['members'].items():
        if item['kind']=='round_02_anniversary_history':
            value=bundle['fallback']
        else:
            frame=pd.read_csv(ART/item['kind']/stage/item['name']/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            np.testing.assert_array_equal(frame.actual,bundle['y_val'])
            value=frame.prediction.to_numpy(float)
        prediction+=item['weight']*value
        components.append({'name':name,'weight':item['weight'],**score(bundle['y_val'],value)})
    path=ART/'ensemble'/stage
    path.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':prediction})
    frame.to_csv(path/'predictions.csv',index=False)
    monthly=[]
    for month,part in frame.groupby('month'):
        previous=bundle['fallback'][frame.month.to_numpy()==month]
        monthly.append({'month':month,'rows':len(part),'previous_rmse':score(part.actual,previous)['rmse_sgd'],
            **score(part.actual,part.prediction)})
    result={'stage':stage,'rows':len(prediction),'scores':score(bundle['y_val'],prediction),
        'previous_scores':score(bundle['y_val'],bundle['fallback']),'components':components,'monthly':monthly}
    write_json(OUT/f'ensemble_{stage}.json',result)
    print('THIRD WAVE ENSEMBLE',stage,result['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

"""Lock a small regularized extension of the earlier-selected third-wave blend."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.third_wave_cross_ensemble import baseline
from analysis.experiments.report_selection_trials import OUT,ART


def select():
    destination=OUT/'optimization_selection.json'
    if destination.exists():
        return
    bundle=make_data('screen')
    previous=baseline('screen',bundle)
    names=['third_wave_locked']
    values=[previous]
    for path in sorted((OUT/'optimization/screen').glob('*/results.json')):
        record=json.loads(path.read_text())
        frame=pd.read_csv(ART/'optimization/screen'/record['name']/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        names.append(record['name'])
        values.append(frame.prediction.to_numpy(float))
    matrix=np.column_stack(values)/1000.
    actual=bundle['y_val']/1000.
    initial=np.zeros(len(names));initial[0]=1.
    ridge=.001
    def objective(weights):
        error=matrix@weights-actual
        shift=weights-initial
        return np.mean(error**2)+ridge*(shift@shift),2*matrix.T@error/len(error)+2*ridge*shift
    fitted=minimize(objective,initial,jac=True,method='SLSQP',
        bounds=[(0.,1.)]+[(0.,.25)]*(len(names)-1),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1000,'ftol':1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    weights=np.where(fitted.x>1e-5,fitted.x,0.);weights/=weights.sum()
    result={'weights':{name:float(weight) for name,weight in zip(names,weights) if weight>0},
        'selection':['2023-10','2024-03'],'ridge':ridge,
        'earlier_scores':score(bundle['y_val'],matrix@weights*1000),
        'previous_scores':score(bundle['y_val'],previous),
        'notes':'Weights chosen only on earlier chronological forecasts; no intercept; cap 0.25 per new model. Existing later periods remain development data.'}
    write_json(destination,result)
    print(json.dumps(result,indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'optimization_selection.json').read_text())
    names=[name for name in selection['weights'] if name!='third_wave_locked' and
        not (ART/'optimization'/stage/name/'predictions.csv').exists()]
    if names:
        subprocess.run([sys.executable,'-u','-m','analysis.experiments.report_selection_trials',
            'optimize','--stage',stage,'--names',*names],check=True)
    bundle=make_data(stage)
    previous=baseline(stage,bundle)
    prediction=np.zeros(len(previous))
    components=[]
    for name,weight in selection['weights'].items():
        if name=='third_wave_locked':
            values=previous
        else:
            frame=pd.read_csv(ART/'optimization'/stage/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            np.testing.assert_array_equal(frame.actual,bundle['y_val'])
            values=frame.prediction.to_numpy(float)
        prediction+=weight*values
        components.append({'name':name,'weight':weight,**score(bundle['y_val'],values)})
    destination=ART/'optimized_ensemble'/stage
    destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':prediction}).to_csv(destination/'predictions.csv',index=False)
    result={'stage':stage,'scores':score(bundle['y_val'],prediction),
        'previous_scores':score(bundle['y_val'],previous),'rows':len(prediction),'components':components}
    write_json(OUT/f'optimization_{stage}.json',result)
    print('REPORT OPTIMIZATION',stage,result['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

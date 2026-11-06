"""Earlier-window selection of recent Transformer representatives and blends."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_ensemble import old_best
from analysis.experiments.seventh_wave_transformers import OUT,ART,SPECS


def previous(stage,bundle,name):
    if name=='previous_history':
        return old_best(stage,bundle)
    if stage=='screen':
        # The sixth-wave weights were already selected on exactly these rows.
        from analysis.experiments.sixth_wave_deep import OUT as SIX_OUT,ART as SIX_ART
        record=json.loads((SIX_OUT/'ensemble_selection.json').read_text())
        result=np.zeros(len(bundle['y_val']))
        for member,weight in record['weights'].items():
            if member=='previous_best':
                result+=weight*old_best(stage,bundle)
            else:
                frame=pd.read_csv(SIX_ART/stage/member/'predictions.csv').sort_values('source_row')
                np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
                result+=weight*frame.prediction.to_numpy(float)
        return result
    frame=pd.read_csv(ROOT/f'output/models/round_06_deep_models/ensemble/{stage}/predictions.csv').sort_values('source_row')
    np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
    return frame.prediction.to_numpy(float)


def select():
    path=OUT/'ensemble_selection.json'
    if path.exists():
        print('Transformer selection already locked',flush=True);return
    bundle=make_data('screen')
    names=['previous_history','previous_deep'];values=[previous('screen',bundle,name) for name in names]
    candidates=[]
    for spec in SPECS:
        name=spec['name'];record=json.loads((OUT/'screen'/name/'results.json').read_text())
        frame=pd.read_csv(ART/'screen'/name/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        names.append(name);values.append(frame.prediction.to_numpy(float))
        candidates.append({'name':name,'family':'nori' if spec['family'].startswith('nori') else spec['family'],
            'rmse_sgd':record['rmse_sgd']})
    matrix=np.column_stack(values)/1000;target=bundle['y_val']/1000
    initial=np.zeros(len(names));initial[0]=1
    ridge=.001
    def objective(w):
        error=matrix@w-target;delta=w-initial
        return np.mean(error**2)+ridge*(delta@delta),2*matrix.T@error/len(target)+2*ridge*delta
    result=minimize(objective,initial,jac=True,method='SLSQP',bounds=[(0,1)]+[(0,.4)]*(len(names)-1),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1000,'ftol':1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    weights=np.where(result.x>1e-5,result.x,0.);weights/=weights.sum()
    representatives={}
    for item in sorted(candidates,key=lambda item:item['rmse_sgd']):
        representatives.setdefault(item['family'],item['name'])
    record={'weights':{name:float(weight) for name,weight in zip(names,weights) if weight>0},
        'representatives':representatives,'screen_scores':score(bundle['y_val'],matrix@weights*1000),
        'candidates':candidates,'selection_months':['2023-10','2024-03'],'ridge':ridge,
        'notes':'Earlier full-window selection only. Reused development periods and prior structure selection prevent independent-test claims. Cap .4 for each new candidate.'}
    write_json(path,record);print(json.dumps(record,indent=2),flush=True)


def evaluate(stage):
    record=json.loads((OUT/'ensemble_selection.json').read_text())
    names=sorted((set(record['weights'])|set(record['representatives'].values()))-{'previous_history','previous_deep'})
    missing=[name for name in names if not (OUT/stage/name/'results.json').exists()]
    if missing:
        subprocess.run([sys.executable,'-u','-m','analysis.experiments.seventh_wave_transformers',stage,'--names',*missing],check=True)
    bundle=make_data(stage);combined=np.zeros(len(bundle['y_val']))
    for name,weight in record['weights'].items():
        if name.startswith('previous_'):
            prediction=previous(stage,bundle,name)
        else:
            frame=pd.read_csv(ART/stage/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            prediction=frame.prediction.to_numpy(float)
        combined+=weight*prediction
    destination=ART/'ensemble'/stage;destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],'actual':bundle['y_val'],
        'prediction':combined}).to_csv(destination/'predictions.csv',index=False)
    summary={'stage':stage,'rows':len(combined),'scores':score(bundle['y_val'],combined),
        'previous_history':score(bundle['y_val'],previous(stage,bundle,'previous_history')),
        'previous_deep':score(bundle['y_val'],previous(stage,bundle,'previous_deep'))}
    write_json(OUT/f'ensemble_{stage}.json',summary)
    print('TRANSFORMER ENSEMBLE',stage,json.dumps(summary['scores']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june']);args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

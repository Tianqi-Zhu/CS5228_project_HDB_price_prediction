"""Earlier-window selection of cross-type features as a separate extension."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import score,write_json
from analysis.experiments.third_wave_training import OUT,ART,make_data


def baseline(stage,bundle):
    if stage!='screen':
        frame=pd.read_csv(ART/'ensemble'/stage/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        return frame.prediction.to_numpy(float)
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    combined=np.zeros(len(bundle['y_val']))
    for item in selection['members'].values():
        if item['kind']=='round_02_anniversary_history':
            values=bundle['fallback']
        else:
            frame=pd.read_csv(ART/item['kind']/'screen'/item['name']/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            values=frame.prediction.to_numpy(float)
        combined+=item['weight']*values
    return combined


def select():
    path=OUT/'cross_type_selection.json'
    if path.exists():
        print('Cross-type weights already locked',flush=True)
        return
    bundle=make_data('screen')
    previous=baseline('screen',bundle)
    names=['round_03_tree_transformer_ensemble']
    values=[previous]
    for path_model in sorted((OUT/'cross_type/screen').glob('*/results.json')):
        result=json.loads(path_model.read_text())
        if result['rmse_sgd']>535:
            continue
        name=result['spec']['name']
        frame=pd.read_csv(ART/'cross_type/screen'/name/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        names.append(name)
        values.append(frame.prediction.to_numpy(float))
    matrix=np.column_stack(values)/1000
    actual=bundle['y_val']/1000
    initial=np.zeros(len(names));initial[0]=1
    def objective(w):
        error=matrix@w-actual
        return np.mean(error**2),2*matrix.T@error/len(error)
    fitted=minimize(objective,initial,jac=True,method='SLSQP',
        bounds=[(0,1)]+[(0,.35)]*(len(names)-1),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1000,'ftol':1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    weights=np.where(fitted.x>1e-5,fitted.x,0.);weights/=weights.sum()
    selection={'weights':{name:float(w) for name,w in zip(names,weights) if w>0},
        'earlier_scores':score(bundle['y_val'],matrix@weights*1000),
        'previous_scores':score(bundle['y_val'],previous),
        'selection_months':['2023-10','2024-03'],
        'note':'Separate extension; previous third-wave weights are unchanged. New weights selected only on the earlier window.'}
    write_json(path,selection)
    print(json.dumps(selection,indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'cross_type_selection.json').read_text())
    names=[name for name in selection['weights'] if name!='round_03_tree_transformer_ensemble' and
        not (ART/'cross_type'/stage/name/'predictions.csv').exists()]
    if names:
        subprocess.run([sys.executable,'-u','-m','analysis.experiments.third_wave_cross_type',stage,'--names',*names],check=True)
    bundle=make_data(stage)
    previous=baseline(stage,bundle)
    combined=np.zeros(len(previous))
    components=[]
    for name,weight in selection['weights'].items():
        if name=='round_03_tree_transformer_ensemble':
            values=previous
        else:
            frame=pd.read_csv(ART/'cross_type'/stage/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            np.testing.assert_array_equal(frame.actual,bundle['y_val'])
            values=frame.prediction.to_numpy(float)
        combined+=weight*values
        components.append({'name':name,'weight':weight,**score(bundle['y_val'],values)})
    directory=ART/'cross_ensemble'/stage
    directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':combined}).to_csv(directory/'predictions.csv',index=False)
    result={'stage':stage,'scores':score(bundle['y_val'],combined),
        'previous_scores':score(bundle['y_val'],previous),'components':components}
    write_json(OUT/f'cross_ensemble_{stage}.json',result)
    print('CROSS ENSEMBLE',stage,result['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

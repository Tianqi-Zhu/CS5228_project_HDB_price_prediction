"""Pre-origin convex selection for fourth-wave candidates, preserving all rows."""
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
from analysis.experiments.third_wave_cross_ensemble import baseline as third_baseline
from analysis.experiments.fourth_wave_diagnostics import OUT,ART

MODULES={kind:'analysis.experiments.fourth_wave_'+module for kind,module in [
    ('horizons','horizons'),('spatial','spatial'),('experts','experts'),('flows','flows'),('distribution','distribution')]}
VERSIONS={'conservative':.001,'rmse':0.}


def old_best(stage,bundle):
    if stage!='screen':
        frame=pd.read_csv(ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        return frame.prediction.to_numpy(float)
    records=json.loads((ROOT/'output/experiment_runs/round_03_tree_transformer_ensemble/ablation_models.json').read_text())['records']
    selected=next(r for r in records if r['stage']=='june' and r['kind']=='models')
    pred=selected['previous_weight']*bundle['fallback']
    for item in selected['members'].values():
        frame=pd.read_csv(ROOT/'output/models/round_03_tree_transformer_ensemble'/item['kind']/stage/item['name']/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        pred+=item['weight']*frame.prediction.to_numpy(float)
    return pred


def select():
    destination=OUT/'ensemble_selection.json'
    if destination.exists():
        print('Fourth-wave selection already frozen',flush=True)
        return
    bundle=make_data('screen')
    registered={'previous_best':{'kind':'previous_best'},'third_locked':{'kind':'third_locked'},
        'second_locked':{'kind':'second_locked'}}
    values=[old_best('screen',bundle),third_baseline('screen',bundle),bundle['fallback']]
    diagnostics=[]
    for kind,module in MODULES.items():
        for path in sorted((OUT/kind/'screen').glob('*/results.json')):
            record=json.loads(path.read_text())
            name=record['spec']['name']
            frame=pd.read_csv(ART/kind/'screen'/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            np.testing.assert_array_equal(frame.actual,bundle['y_val'])
            diagnostics.append({'kind':kind,'name':name,**score(frame.actual,frame.prediction)})
            if record['rmse_sgd']>535.:
                continue
            registered[f'{kind}/{name}']={'kind':kind,'name':name,'module':module}
            values.append(frame.prediction.to_numpy(float))
    matrix=np.column_stack(values)/1000.
    actual=bundle['y_val']/1000.
    initial=np.zeros(len(registered));initial[0]=1.
    versions={}
    for version,ridge in VERSIONS.items():
        def objective(w):
            error=matrix@w-actual
            shift=w-initial
            return np.mean(error**2)+ridge*(shift@shift),2*matrix.T@error/len(error)+2*ridge*shift
        fitted=minimize(objective,initial,jac=True,method='SLSQP',
            bounds=[(0,1.)]*3+[(0,.35)]*(len(registered)-3),
            constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
            options={'maxiter':1500,'ftol':1e-12})
        if not fitted.success:
            raise RuntimeError(fitted.message)
        weights=np.where(fitted.x>1e-5,fitted.x,0.);weights/=weights.sum()
        members={name:item|{'weight':float(w)} for (name,item),w in zip(registered.items(),weights) if w>0}
        versions[version]={'ridge':ridge,'members':members,'earlier_scores':score(bundle['y_val'],matrix@weights*1000)}
    result={'versions':versions,'diagnostics':diagnostics,'selection_months':['2023-10','2024-03'],
        'previous_best_screen':score(bundle['y_val'],values[0]),
        'note':'Two predeclared convex objectives: fixed regularization and pure RMSE; no intercept. Later periods have been repeatedly viewed development data, including selection of the previous-best structure.'}
    write_json(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'},indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    for kind,module in MODULES.items():
        names=sorted({item['name'] for version in selection['versions'].values()
            for item in version['members'].values() if item['kind']==kind and
            not (ART/kind/stage/item['name']/'predictions.csv').exists()})
        if names:
            subprocess.run([sys.executable,'-u','-m',module,stage,'--names',*names],check=True)
    bundle=make_data(stage)
    outputs={}
    for name,version in selection['versions'].items():
        pred=np.zeros(len(bundle['y_val']))
        components=[]
        for key,item in version['members'].items():
            if item['kind']=='previous_best':
                values=old_best(stage,bundle)
            elif item['kind']=='third_locked':
                values=third_baseline(stage,bundle)
            elif item['kind']=='second_locked':
                values=bundle['fallback']
            else:
                frame=pd.read_csv(ART/item['kind']/stage/item['name']/'predictions.csv').sort_values('source_row')
                np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
                np.testing.assert_array_equal(frame.actual,bundle['y_val'])
                values=frame.prediction.to_numpy(float)
            pred+=item['weight']*values
            components.append({'name':key,'weight':item['weight'],**score(bundle['y_val'],values)})
        directory=ART/'ensemble'/stage/name
        directory.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':pred}).to_csv(directory/'predictions.csv',index=False)
        outputs[name]={'scores':score(bundle['y_val'],pred),'components':components}
    result={'stage':stage,'rows':len(pred),'versions':outputs,
        'previous_best':score(bundle['y_val'],old_best(stage,bundle)),
        'target_430_achieved':any(v['scores']['rmse_sgd']<=430 for v in outputs.values())}
    write_json(OUT/f'ensemble_{stage}.json',result)
    print('FOURTH WAVE',stage,json.dumps({k:v['scores'] for k,v in outputs.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

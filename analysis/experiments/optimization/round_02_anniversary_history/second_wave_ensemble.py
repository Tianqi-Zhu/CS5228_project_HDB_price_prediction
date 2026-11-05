"""Lock a second-wave blend on earlier forecasts and refit active components."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.second_wave_stacking import OUT,ART,calibration_data
from analysis.experiments.frontier_search import ROOT,score,write_json


def candidates():
    items={}
    for kind in ('normalized_market','anniversary'):
        for path in (OUT/kind/'screen').glob('*/results.json'):
            record=json.loads(path.read_text())
            module='analysis.experiments.second_wave_normalized' if kind=='normalized_market' else 'analysis.experiments.second_wave_anniversary'
            name=record['name']
            items[f'{kind}/{name}']={'kind':kind,'name':name,'module':module,
                'screen_path':str((ART/kind/'screen'/name/'predictions.csv').relative_to(ROOT))}
    # Full earlier confirmation only, not the 1,200-row preliminary screens.
    for path in (OUT/'tabicl/confirm').glob('*/results.json'):
        record=json.loads(path.read_text())
        name=record['spec']['name']
        items[f'tabicl/{name}']={'kind':'tabicl','name':name,
            'module':'analysis.experiments.second_wave_foundation',
            'screen_path':str((ART/'tabicl/confirm'/name/'predictions.csv').relative_to(ROOT))}
    return items


def select():
    destination=OUT/'ensemble_selection.json'
    if destination.exists():
        print('Second-wave selection already locked',flush=True)
        return
    base=calibration_data('screen')
    registered={'baseline':{'kind':'baseline'}}
    matrix=[base['baseline']]
    diagnostics=[]
    for name,item in candidates().items():
        predictions=pd.read_csv(ROOT/item['screen_path']).sort_values('source_row')
        np.testing.assert_array_equal(predictions.source_row,base['row'])
        np.testing.assert_array_equal(predictions.actual,base['actual'])
        rmse=score(predictions.actual,predictions.prediction)['rmse_sgd']
        diagnostics.append({'name':name,'earlier_rmse':rmse})
        if rmse<556.:
            registered[name]=item
            matrix.append(predictions.prediction.to_numpy(float))
    matrix=np.column_stack(matrix)/1000.
    actual=base['actual']/1000.
    def objective(weights):
        residual=matrix@weights-actual
        return np.mean(residual**2),2*matrix.T@residual/len(actual)
    fitted=minimize(objective,np.full(matrix.shape[1],1/matrix.shape[1]),jac=True,
        method='SLSQP',bounds=[(0.,.6)]*matrix.shape[1],
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1000,'ftol':1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    weights=np.where(fitted.x>1e-5,fitted.x,0.)
    weights/=weights.sum()
    members={name:item|{'weight':float(w)} for (name,item),w in zip(registered.items(),weights) if w>0}
    result={'members':members,'earlier_scores':score(actual*1000,matrix@weights*1000),
        'candidate_diagnostics':diagnostics,'selection_months':['2023-10','2024-03'],
        'note':'Convex blend with each weight <=0.6, using only pre-April labels. Existing outer periods are development data.'}
    write_json(destination,result)
    print(json.dumps(result,indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    base=calibration_data('outer' if stage=='outer' else 'june')
    prediction=np.zeros(len(base['actual']))
    components=[]
    for name,item in selection['members'].items():
        if item['kind']=='baseline':
            values=base['baseline']
        else:
            path=ART/item['kind']/stage/item['name']/'predictions.csv'
            if not path.exists():
                subprocess.run([sys.executable,'-u','-m',item['module'],stage,'--names',item['name']],check=True)
            frame=pd.read_csv(path).sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,base['row'])
            np.testing.assert_array_equal(frame.actual,base['actual'])
            values=frame.prediction.to_numpy(float)
        prediction+=item['weight']*values
        components.append({'name':name,'weight':item['weight'],**score(base['actual'],values)})
    destination=ART/'ensemble'/stage
    destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':base['row'],'month':base['month'],'actual':base['actual'],
        'prediction':prediction}).to_csv(destination/'predictions.csv',index=False)
    result={'stage':stage,'rows':len(prediction),'scores':score(base['actual'],prediction),
        'previous_round_baseline':score(base['actual'],base['baseline']),
        'components':components,'target_430_achieved':score(base['actual'],prediction)['rmse_sgd']<=430.}
    write_json(OUT/f'ensemble_{stage}.json',result)
    print('SECOND WAVE ENSEMBLE',stage,result['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

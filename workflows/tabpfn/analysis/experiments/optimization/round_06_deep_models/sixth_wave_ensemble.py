"""Earlier-selected deep-model ensembles and mixing with validated tree blend."""
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
from analysis.experiments.sixth_wave_deep import OUT,ART,SEED_GROUPS,SPECS


def create_seed_means(stage):
    for base,names in SEED_GROUPS.items():
        if not all((ART/stage/name/'predictions.csv').exists() for name in names):
            continue
        combined=None;reference=None
        for name in names:
            frame=pd.read_csv(ART/stage/name/'predictions.csv').sort_values('source_row')
            if reference is None:
                reference=frame;combined=np.zeros(len(frame))
            np.testing.assert_array_equal(frame.source_row,reference.source_row)
            np.testing.assert_array_equal(frame.actual,reference.actual)
            combined+=frame.prediction.to_numpy(float)/len(names)
        name='seedmean__'+base
        directory=ART/stage/name;directory.mkdir(parents=True,exist_ok=True)
        reference=reference.copy();reference['prediction']=combined
        reference.to_csv(directory/'predictions.csv',index=False)
        result=json.loads((OUT/stage/base/'results.json').read_text())
        result['spec']=result['spec']|{'name':name,'seed_members':names}
        result['epochs']={member:json.loads((OUT/stage/member/'results.json').read_text())['epochs'] for member in names}
        seed_records=[json.loads((OUT/stage/member/'results.json').read_text()) for member in names]
        result['seconds']=sum(item['seconds'] for item in seed_records)
        result['peak_vram_gib']=max(item['peak_vram_gib'] for item in seed_records)
        result['training_repetitions']=3
        result.update(score(reference.actual,combined))
        result['note']='Equal mean of predeclared seeds 42/43/44; no seed selected by later validation.'
        write_json(OUT/stage/name/'results.json',result)


def select():
    path=OUT/'ensemble_selection.json'
    if path.exists():
        print('Deep-model blend selection already locked',flush=True);return
    bundle=make_data('screen')
    create_seed_means('screen')
    names=['previous_best'];values=[old_best('screen',bundle)]
    candidates=[]
    for path_model in sorted((OUT/'screen').glob('*/results.json')):
        result=json.loads(path_model.read_text())
        if result['spec'].get('seed_group'):
            continue
        if result['rmse_sgd']>545.:
            continue
        name=result['spec']['name']
        frame=pd.read_csv(ART/'screen'/name/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        names.append(name);values.append(frame.prediction.to_numpy(float))
        candidates.append({'name':name,'family':result['spec']['family'],'rmse_sgd':result['rmse_sgd']})
    matrix=np.column_stack(values)/1000.
    target=bundle['y_val']/1000.
    initial=np.zeros(len(names));initial[0]=1.
    ridge=.001
    def objective(w):
        error=matrix@w-target
        shift=w-initial
        return np.mean(error**2)+ridge*(shift@shift),2*matrix.T@error/len(error)+2*ridge*shift
    result=minimize(objective,initial,jac=True,method='SLSQP',
        bounds=[(0.,1.)]+[(0.,.5)]*(len(names)-1),
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':1000,'ftol':1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    weights=np.where(result.x>1e-5,result.x,0.);weights/=weights.sum()
    # Also select one representative per architecture on the same earlier data,
    # so the report contains a genuine model comparison, not only blend members.
    representatives={}
    for item in sorted(candidates,key=lambda item:item['rmse_sgd']):
        representatives.setdefault(item['family'],item['name'])
    record={'weights':{name:float(weight) for name,weight in zip(names,weights) if weight>0},
        'representatives':representatives,'earlier_scores':score(bundle['y_val'],matrix@weights*1000),
        'previous_scores':score(bundle['y_val'],values[0]),'selection_months':['2023-10','2024-03'],
        'ridge':ridge,'note':'Nonnegative sum-one mixture; each new candidate <=0.5; ridge anchored to previous best. The previous-best structure and all later periods are reused development evidence.'}
    write_json(path,record)
    print(json.dumps(record,indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    names=sorted((set(selection['weights'])|set(selection['representatives'].values()))-{'previous_best'})
    expanded=set()
    for name in names:
        spec=json.loads((OUT/'screen'/name/'results.json').read_text())['spec']
        expanded.update(spec.get('seed_members',[name]))
    missing=sorted(name for name in expanded if not (ART/stage/name/'predictions.csv').exists())
    if missing:
        subprocess.run([sys.executable,'-u','-m','analysis.experiments.sixth_wave_deep',stage,'--names',*missing],check=True)
    create_seed_means(stage)
    bundle=make_data(stage)
    combined=np.zeros(len(bundle['y_val']))
    components=[]
    for name,weight in selection['weights'].items():
        if name=='previous_best':
            pred=old_best(stage,bundle)
        else:
            frame=pd.read_csv(ART/stage/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
            np.testing.assert_array_equal(frame.actual,bundle['y_val'])
            pred=frame.prediction.to_numpy(float)
        combined+=weight*pred
        components.append({'name':name,'weight':weight,**score(bundle['y_val'],pred)})
    directory=ART/'ensemble'/stage;directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':combined}).to_csv(directory/'predictions.csv',index=False)
    record={'stage':stage,'rows':len(combined),'scores':score(bundle['y_val'],combined),
        'previous_scores':score(bundle['y_val'],old_best(stage,bundle)),
        'components':components,'target_430_achieved':score(bundle['y_val'],combined)['rmse_sgd']<=430.}
    write_json(OUT/f'ensemble_{stage}.json',record)
    print('DEEP ENSEMBLE',stage,json.dumps(record['scores']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

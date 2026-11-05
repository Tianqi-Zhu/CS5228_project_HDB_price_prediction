"""Separate competition-information blend with supplied query-month macros."""
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
from analysis.experiments.fourth_wave_diagnostics import OUT,ART
from analysis.experiments.fourth_wave_ensemble import old_best


def select():
    path=OUT/'given_macro_selection.json'
    if path.exists():
        return
    bundle=make_data('screen')
    names=['previous_best']
    predictions=[old_best('screen',bundle)]
    for result in sorted((OUT/'known_macro/screen').glob('*/results.json')):
        record=json.loads(result.read_text())
        name=record['spec']['name']
        frame=pd.read_csv(ART/'known_macro/screen'/name/'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
        np.testing.assert_array_equal(frame.actual,bundle['y_val'])
        names.append(name);predictions.append(frame.prediction.to_numpy(float))
    matrix=np.column_stack(predictions)/1000.
    target=bundle['y_val']/1000.
    initial=np.zeros(len(names));initial[0]=1.
    versions={}
    for version,ridge in [('conservative',.001),('rmse',0.)]:
        def objective(w):
            error=matrix@w-target
            shift=w-initial
            return np.mean(error**2)+ridge*(shift@shift),2*matrix.T@error/len(error)+2*ridge*shift
        result=minimize(objective,initial,jac=True,method='SLSQP',
            bounds=[(0.,1.)]+[(0.,.35)]*(len(names)-1),
            constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
            options={'maxiter':1000,'ftol':1e-12})
        if not result.success:
            raise RuntimeError(result.message)
        weights=np.where(result.x>1e-5,result.x,0.);weights/=weights.sum()
        versions[version]={'weights':{name:float(w) for name,w in zip(names,weights) if w>0},
            'ridge':ridge,'earlier_scores':score(bundle['y_val'],matrix@weights*1000)}
    write_json(path,{'versions':versions,'selection_months':['2023-10','2024-03'],
        'scenario':'Query-month macro covariates are supplied by the competition; rental labels frozen at origin.',
        'notes':'Separate experiment after the strict-information trial; same fixed earlier window selects weights, no later label selects macro model rounds or weights.'})
    print('GIVEN MACRO SELECTION',json.dumps(versions,indent=2),flush=True)


def evaluate(stage):
    selection=json.loads((OUT/'given_macro_selection.json').read_text())
    names=sorted({name for version in selection['versions'].values() for name in version['weights']
        if name!='previous_best' and not (ART/'known_macro'/stage/name/'predictions.csv').exists()})
    if names:
        subprocess.run([sys.executable,'-u','-m','analysis.experiments.fourth_wave_known_macro',stage,'--names',*names],check=True)
    bundle=make_data(stage)
    previous=old_best(stage,bundle)
    versions={}
    for name,configuration in selection['versions'].items():
        combined=np.zeros(len(previous))
        for member,weight in configuration['weights'].items():
            if member=='previous_best':
                values=previous
            else:
                frame=pd.read_csv(ART/'known_macro'/stage/member/'predictions.csv').sort_values('source_row')
                np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
                np.testing.assert_array_equal(frame.actual,bundle['y_val'])
                values=frame.prediction.to_numpy(float)
            combined+=weight*values
        directory=ART/'given_ensemble'/stage/name
        directory.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':combined}).to_csv(directory/'predictions.csv',index=False)
        versions[name]=score(bundle['y_val'],combined)
    result={'stage':stage,'scenario':'given_query_month_macro','versions':versions,
        'previous_best':score(bundle['y_val'],previous)}
    write_json(OUT/f'given_ensemble_{stage}.json',result)
    print('GIVEN ENSEMBLE',stage,json.dumps(versions),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

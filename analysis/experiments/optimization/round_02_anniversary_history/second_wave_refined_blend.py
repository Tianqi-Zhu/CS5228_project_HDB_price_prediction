"""Evaluate the locked blend with the focused cohort refinement."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from analysis.experiments.second_wave_stacking import OUT,ART
from analysis.experiments.frontier_search import ROOT,score,write_json


def select():
    destination=OUT/'cohort_extensions/blend_selection.json'
    if destination.exists():
        return
    members=json.loads((OUT/'ensemble_selection.json').read_text())['members']
    reference,base=None,None
    for item in members.values():
        frame=pd.read_csv(ROOT/item['screen_path']).sort_values('source_row')
        if reference is None:
            reference=frame
            base=np.zeros(len(frame))
        np.testing.assert_array_equal(frame.source_row,reference.source_row)
        np.testing.assert_array_equal(frame.actual,reference.actual)
        base+=item['weight']*frame.prediction.to_numpy()
    candidates=[json.loads(p.read_text()) for p in (OUT/'cohort_extensions/screen').glob('*/results.json')]
    best=min(candidates,key=lambda r:r['rmse_sgd'])
    frame=pd.read_csv(ART/'cohort_extensions/screen'/best['name']/'predictions.csv').sort_values('source_row')
    np.testing.assert_array_equal(frame.source_row,reference.source_row)
    difference=frame.prediction.to_numpy()-base
    weight=float(np.clip(np.dot(reference.actual.to_numpy()-base,difference)/np.dot(difference,difference),0,.6))
    write_json(destination,{'candidate':best['name'],'candidate_weight':weight,
        'earlier_scores':score(reference.actual,base+weight*difference),
        'previous_earlier_scores':score(reference.actual,base),'selection_months':['2023-10','2024-03']})


def evaluate(stage):
    selected=json.loads((OUT/'cohort_extensions/blend_selection.json').read_text())
    candidate=selected['candidate']
    path=ART/'cohort_extensions'/stage/candidate/'predictions.csv'
    if not path.exists():
        subprocess.run([sys.executable,'-u','-m','analysis.experiments.second_wave_cohort_extensions',stage,
            '--names',candidate],check=True)
    base=pd.read_csv(ART/'ensemble'/stage/'predictions.csv').sort_values('source_row')
    part=pd.read_csv(path).sort_values('source_row')
    np.testing.assert_array_equal(base.source_row,part.source_row)
    np.testing.assert_array_equal(base.actual,part.actual)
    weight=selected['candidate_weight']
    predicted=(1-weight)*base.prediction.to_numpy()+weight*part.prediction.to_numpy()
    destination=ART/'refined_ensemble'/stage
    destination.mkdir(parents=True,exist_ok=True)
    frame=base.copy()
    frame.prediction=predicted
    frame.to_csv(destination/'predictions.csv',index=False)
    result={'stage':stage,'selection':selected,'scores':score(base.actual,predicted),
        'base_scores':score(base.actual,base.prediction),'candidate_scores':score(part.actual,part.prediction),
        'rows':len(frame)}
    write_json(OUT/f'refined_ensemble_{stage}.json',result)
    print('REFINED BLEND',stage,result['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','outer','june'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate(args.stage)

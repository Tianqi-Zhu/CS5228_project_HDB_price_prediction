"""Six-month pre-origin selection followed by a frozen 16-month stress forecast.

Model/feature structures come from current development. Iterations and blend
weights are separately selected on June-November 2023, before December origin.
"""
from __future__ import annotations
import gc
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.second_wave_stacking import OUT,ART
from analysis.experiments.second_wave_cohort_extensions import make_data,fit as fit_cohort
from analysis.experiments.second_wave_anniversary import SPECS as COHORT_SPECS
from analysis.experiments.second_wave_normalized import SPECS as NORMAL_SPECS,fit as fit_normal
from analysis.experiments.frontier_search import ROOT,score,write_json

DEST=OUT/'stress_16m'
MODEL_ROOT=ART/'stress_16m'


def main():
    members=json.loads((OUT/'ensemble_selection.json').read_text())['members']
    write_json(DEST/'plan.json',{'members':list(members),
        'selection':['2023-06','2023-11'],'origin':'2023-12',
        'validation':['2023-12','2025-03'],
        'note':'Separate pre-origin iteration and weight selection; structures discovered during current development. Overlaps other windows; never pool with them.'})
    for stage,start,end in [('selection','2023-06','2023-12'),('evaluation','2023-12','2025-04')]:
        bundle=make_data(start,end)
        source=bundle['raw_fit']
        bundle['reference']={method:source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform(method).to_numpy(float)
            for method in ('mean','median')}
        for name,item in members.items():
            destination=DEST/stage/item['name']
            if (destination/'results.json').exists():
                continue
            specs=COHORT_SPECS if item['kind']=='anniversary' else NORMAL_SPECS
            spec=next(s for s in specs if s['name']==item['name'])
            count=None if stage=='selection' else json.loads((DEST/'selection'/item['name']/'results.json').read_text())['iterations']
            print('STRESS',stage,name,flush=True)
            if item['kind']=='anniversary':
                model,prediction,count,_=fit_cohort(spec,bundle,count)
            else:
                model,prediction,count=fit_normal(spec,bundle,count)
            directory=MODEL_ROOT/stage/item['name']
            directory.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
                'actual':bundle['y_val'],'prediction':prediction}).to_csv(directory/'predictions.csv',index=False)
            joblib.dump({'model':model,'spec':spec},directory/'model.joblib',compress=1)
            record={'name':name,'spec':spec,'stage':stage,'iterations':count,'rows':len(prediction),
                **score(bundle['y_val'],prediction),'source_sha256':bundle['hashes']}
            write_json(destination/'results.json',record)
            print('STRESS RESULT',stage,name,record['rmse_sgd'],flush=True)
            del model
            gc.collect()
        if stage=='selection':
            columns=[]
            for name,item in members.items():
                frame=pd.read_csv(MODEL_ROOT/stage/item['name']/'predictions.csv').sort_values('source_row')
                np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
                columns.append(frame.prediction.to_numpy()/1000.)
            matrix=np.column_stack(columns)
            actual=bundle['y_val']/1000.
            def objective(w):
                residual=matrix@w-actual
                return np.mean(residual**2),2*matrix.T@residual/len(actual)
            fitted=minimize(objective,np.full(len(members),1/len(members)),jac=True,
                method='SLSQP',bounds=[(0.,.6)]*len(members),
                constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
                options={'maxiter':1000,'ftol':1e-12})
            if not fitted.success:
                raise RuntimeError(fitted.message)
            weights={name:float(w) for name,w in zip(members,fitted.x)}
            write_json(DEST/'selection.json',{'weights':weights,
                'scores':score(actual*1000,matrix@fitted.x*1000),
                'last_selection_month':'2023-11','evaluation_origin':'2023-12'})
        del bundle
        gc.collect()
    selection=json.loads((DEST/'selection.json').read_text())
    reference,prediction=None,None
    for name,item in members.items():
        frame=pd.read_csv(MODEL_ROOT/'evaluation'/item['name']/'predictions.csv').sort_values('source_row')
        if reference is None:
            reference=frame
            prediction=np.zeros(len(frame))
        np.testing.assert_array_equal(frame.source_row,reference.source_row)
        np.testing.assert_array_equal(frame.actual,reference.actual)
        prediction+=selection['weights'][name]*frame.prediction.to_numpy()
    old=pd.read_csv(ROOT/'output/experiment_runs/comparison_model_search/2023-12/xgboost/predictions.csv').sort_values('source_row')
    np.testing.assert_array_equal(old.source_row,reference.source_row)
    np.testing.assert_array_equal(old.actual,reference.actual)
    reference['prediction']=prediction
    reference.to_csv(MODEL_ROOT/'ensemble_predictions.csv',index=False)
    result={'validation':['2023-12','2025-03'],'rows':len(reference),
        'scores':score(reference.actual,prediction),'previous_xgboost':score(old.actual,old.prediction),
        'weights':selection['weights'],
        'limitations':'Reused development periods and overlapping diagnostic window; structures chosen during current development, no post-origin label selects iterations or weights.'}
    write_json(DEST/'summary.json',result)
    print('STRESS ENSEMBLE',result['scores'],flush=True)


if __name__=='__main__':
    main()

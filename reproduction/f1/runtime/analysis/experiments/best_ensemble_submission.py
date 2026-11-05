"""Refit the validated ensemble on all labels and export unscored submissions.

No upload. Test labels are unavailable. Every history excludes the query month
and freezes after the final supplied training month. Model counts/weights come
from the saved development experiments, never from the unlabelled test target.
"""
from __future__ import annotations
import gc
import json
import sys
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,DistributionHistory,month_number,write_json
from analysis.experiments.second_wave_anniversary import AnniversaryHistory
from analysis.pipeline.data import load_main,sha256
from analysis.pipeline.features import build_feature_frame,model_inputs,FEATURE_CANDIDATES
from analysis.experiments.canonical_final import validate_submission

OUT=ROOT/'output/submissions/best_history_ensemble'
REPORT=ROOT/'reports/experiments/fourth_wave'


def prepare():
    source,hashes=frame_data()
    test_path=ROOT/'data/test.csv'
    query=build_feature_frame(load_main(test_path))
    hashes=dict(hashes)|{'data/test.csv':sha256(test_path)}
    cache=OUT/'features.joblib'
    if cache.exists():
        saved=joblib.load(cache)
        if saved['hashes']==hashes:
            return saved['bundle'],hashes
    x,q=model_inputs(source,query,FEATURE_CANDIDATES['compact_history_27'])
    distribution=DistributionHistory().fit(source)
    fitting_months=month_number(source.RENT_APPROVAL_DATE)
    query_months=month_number(query.RENT_APPROVAL_DATE)
    cutoff=np.minimum(query_months,fitting_months.max()+1)
    print('FULL REFIT: fitting histories',flush=True)
    extra=distribution.transform(source,fitting_months,neighbors=False)
    print('FULL REFIT: frozen query histories',flush=True)
    other=distribution.transform(query,cutoff,neighbors=False)
    cohorts=AnniversaryHistory().fit(source)
    bundle={'x_fit':x,'x_val':q,'extra_fit':extra,'extra_val':other,
        'ann_fit':cohorts.transform(source),'ann_val':cohorts.transform(query),
        'y_fit':source.MONTHLY_RENT.to_numpy(float),
        # Fixed-round fitting paths do not use this placeholder in training,
        # stopping or inference; there are no actual test targets.
        'y_val':np.zeros(len(query)), 'row_val':query.index.to_numpy(),
        'date_val':query.RENT_APPROVAL_DATE.to_numpy(),
        'reference':{method:source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform(method).to_numpy(float)
            for method in ('mean','median')},'raw_fit':source,'raw_val':query,'hashes':hashes}
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'hashes':hashes,'bundle':bundle},cache,compress=1)
    return bundle,hashes


def predict_saved(payload,bundle,kind):
    spec=payload['spec']
    if kind=='normalized_market':
        from analysis.experiments.second_wave_normalized import select_inputs
        _,q,_,_,_,level=select_inputs(bundle,spec)
        prediction=payload['model'].predict(q)
        return prediction*level if spec['target']=='ratio' else prediction+level
    if kind=='anniversary':
        from analysis.experiments.second_wave_cohort_extensions import features
        _,q=features(bundle,spec)
        if spec['family']=='cat':
            for c in q.select_dtypes(include=['category','object','string']).columns:
                q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
            q=q.fillna(-999.)
        return payload['model'].predict(q)
    if kind=='known_macro':
        from analysis.experiments.fourth_wave_known_macro import inputs
        _,q=inputs(bundle,spec)
        if spec['family']=='cat':
            for c in q.select_dtypes(include=['category','object','string']).columns:
                q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
            q=q.fillna(-999.)
        return payload['model'].predict(q)
    from analysis.experiments.third_wave_training import inputs
    _,q,_,offset=inputs(bundle,spec)
    if 'fitted' in payload:
        import torch
        from analysis.pipeline.neural_models import predict_neural
        fitted=payload['fitted']
        fitted['device']='cuda'
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.25)
        fitted['model'].to('cuda')
        prediction=predict_neural(fitted,q[payload['columns']])+offset
        fitted['model'].to('cpu');fitted['device']='cpu'
        torch.cuda.empty_cache()
        return prediction
    return payload['model'].predict(q)+offset


def run():
    sys.path.insert(0,str(ROOT/'.cache/model-deps'))
    OUT.mkdir(parents=True,exist_ok=True)
    bundle,hashes=prepare()
    specification=json.loads((ROOT/'reports/experiments/report_selection/best_ensemble.json').read_text(encoding='utf-8'))
    ensemble=np.zeros(len(bundle['y_val']))
    components=[]
    for item in specification['leaf_members']:
        name=item['name']
        directory=OUT/'models'/name
        directory.mkdir(parents=True,exist_ok=True)
        checkpoint=directory/'model.joblib'
        report=ROOT/'reports/experiments'/item['experiment_root']/item['kind']/'june'/item['artifact_name']/'results.json'
        spec=json.loads(report.read_text())['spec']
        if checkpoint.exists():
            payload=joblib.load(checkpoint)
            assert payload['source_sha256']==hashes and payload['iterations']==item['iterations']
            prediction=predict_saved(payload,bundle,item['kind'])
        else:
            print('FULL REFIT MEMBER',name,'rounds',item['iterations'],flush=True)
            if item['kind']=='normalized_market':
                from analysis.experiments.second_wave_normalized import fit
                model,prediction,count=fit(spec,bundle,item['iterations'])
                payload={'model':model,'spec':spec}
            elif item['kind']=='anniversary':
                from analysis.experiments.second_wave_cohort_extensions import fit
                model,prediction,count,_=fit(spec,bundle,item['iterations'])
                payload={'model':model,'spec':spec}
            else:
                from analysis.experiments.third_wave_models import fit
                payload,prediction,count,_,_=fit(spec,bundle,item['iterations'])
            payload['source_sha256']=hashes
            payload['iterations']=item['iterations']
            joblib.dump(payload,checkpoint,compress=1)
            del payload
            gc.collect()
            payload=joblib.load(checkpoint)
            replay=predict_saved(payload,bundle,item['kind'])
            np.testing.assert_allclose(replay,prediction,atol=1e-6,rtol=0)
        np.save(directory/'predictions.npy',prediction)
        ensemble+=item['weight']*prediction
        components.append({'name':name,'weight':item['weight'],'iterations':item['iterations'],
            'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/')})
        del payload
        gc.collect()
    template=pd.read_csv(ROOT/'data/example-submission.csv')
    submission=validate_submission(template,ensemble)
    path=OUT/'submission.csv'
    submission.to_csv(path,index=False)
    reread=pd.read_csv(path)
    np.testing.assert_allclose(reread.Predicted,ensemble,atol=1e-8,rtol=0)
    np.testing.assert_array_equal(reread.Id,np.arange(len(ensemble)))
    outputs=[{'scenario':'unknown_future_macro','path':str(path.relative_to(ROOT)).replace('\\','/'),
        'sha256':sha256(path),'paired_local_development_rmse':specification['recent']['rmse_sgd']}]
    # Also export the tiny-gain, explicitly different competition-data scenario.
    selection_path=REPORT/'given_macro_selection.json'
    if selection_path.exists():
        from analysis.experiments.fourth_wave_known_macro import macro_table,fit
        table=macro_table()
        for suffix,raw in [('fit',bundle['raw_fit']),('val',bundle['raw_val'])]:
            bundle['macro_'+suffix]=pd.DataFrame(table.reindex(raw.RENT_APPROVAL_DATE).to_numpy(np.float32),
                index=raw.index,columns=table.columns)
        selection=json.loads(selection_path.read_text())['versions']['conservative']['weights']
        combined=np.zeros(len(ensemble))
        for name,weight in selection.items():
            if name=='previous_best':
                combined+=weight*ensemble
                continue
            directory=OUT/'models/known_macro'/name
            directory.mkdir(parents=True,exist_ok=True)
            checkpoint=directory/'model.joblib'
            record=json.loads((REPORT/'known_macro/screen'/name/'results.json').read_text())
            if checkpoint.exists():
                payload=joblib.load(checkpoint)
                assert payload['source_sha256']==hashes
                pred=predict_saved(payload,bundle,'known_macro')
            else:
                payload,pred,_,_=fit(record['spec'],bundle,record['iterations'])
                payload['source_sha256']=hashes
                joblib.dump(payload,checkpoint,compress=1)
                restored=joblib.load(checkpoint)
                np.testing.assert_allclose(pred,predict_saved(restored,bundle,'known_macro'),atol=1e-6,rtol=0)
            combined+=weight*pred
            del payload
            gc.collect()
        macro_submission=validate_submission(template,combined)
        path=OUT/'submission_given_macro.csv'
        macro_submission.to_csv(path,index=False)
        reread=pd.read_csv(path)
        np.testing.assert_allclose(reread.Predicted,combined,atol=1e-8,rtol=0)
        np.testing.assert_array_equal(reread.Id,np.arange(len(ensemble)))
        outputs.append({'scenario':'given_query_month_macro','path':str(path.relative_to(ROOT)).replace('\\','/'),
            'sha256':sha256(path),'paired_local_development_rmse':442.88921226230013})
    for path,digest in hashes.items():
        if sha256(ROOT/path)!=digest:
            raise AssertionError('Raw source changed')
    summary={'training_rows':len(bundle['y_fit']),'test_rows':len(ensemble),
        'training_months':[bundle['raw_fit'].RENT_APPROVAL_DATE.min(),bundle['raw_fit'].RENT_APPROVAL_DATE.max()],
        'test_months':[bundle['raw_val'].RENT_APPROVAL_DATE.min(),bundle['raw_val'].RENT_APPROVAL_DATE.max()],
        'outputs':outputs,'members':components,'source_sha256':hashes,
        'actual_kaggle_score':None,'uploaded':False,
        'note':'Local scores describe chronological development refits, not these unlabelled test rows. No constant local-to-Kaggle offset is assumed.'}
    write_json(OUT/'summary.json',summary)
    write_json(REPORT/'submission_export.json',summary)
    print('SUBMISSION EXPORT',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run()

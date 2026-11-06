"""Recent author Transformer foundation models, with frozen rental-history origins."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
import random
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data,inputs
from analysis.experiments.sixth_wave_deep import NativeEncoder
from analysis.pipeline.data import sha256

OUT=ROOT/'output/experiment_runs/round_07_pretrained_transformers'
ART=ROOT/'output/models/round_07_pretrained_transformers'
SPECS=[]
for target in ('raw','residual'):
    SPECS.append({'name':f'tabdpt13_{target}_global2048','family':'tabdpt13','target':target,
        'context':2048,'members':2,'reduction':'subsample'})
    SPECS.append({'name':f'nori15_{target}_2048','family':'nori15','target':target,
        'context':2048,'members':4})
SPECS += [{'name':'tabdpt13_residual_retrieval128_bf16','family':'tabdpt13','target':'residual',
    'context':128,'members':2,'reduction':'retrieval','amp':True},
    {'name':'nori6m_residual_2048','family':'nori6m','target':'residual','context':2048,'members':4},
    {'name':'tabicl2_residual_24000','family':'tabicl2','target':'residual','context':24000,'members':2},
    {'name':'tabicl2_raw_24000','family':'tabicl2','target':'raw','context':24000,'members':2}]
SPECS += [{'name':'tabdpt13_ratio_global2048','family':'tabdpt13','target':'ratio','context':2048,
    'members':2,'reduction':'subsample'},
    {'name':'nori15_ratio_2048','family':'nori15','target':'ratio','context':2048,'members':4},
    {'name':'tabicl2_ratio_24000','family':'tabicl2','target':'ratio','context':24000,'members':2}]
SPECS += [{'name':'tabdpt13_raw_mature2048','family':'tabdpt13','target':'raw','context':2048,
    'members':2,'reduction':'subsample','mature':True},
    {'name':'nori15_raw_mature2048','family':'nori15','target':'raw','context':2048,'members':4,'mature':True},
    {'name':'tabicl2_raw_mature24000','family':'tabicl2','target':'raw','context':24000,'members':2,'mature':True}]
SPECS += [{'name':'tabdpt13_adapt_residual','family':'tabdpt13_adapt','target':'residual',
    'context':2048,'members':2,'reduction':'subsample','amp':True,'adapt':True}]


def enable():
    # No requests, client mode, or remote table uploads: author local estimators only.
    for path in reversed([ROOT/'.cache/transformer-deps',ROOT/'.cache/foundation-deps',ROOT/'.cache/model-deps']):
        if str(path) not in sys.path:
            sys.path.insert(0,str(path))
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TABPFN_NO_BROWSER']='1'
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/transformer-numba')
    os.environ['SYNTHEFY_MAX_ELEMENTS_BUDGET']='500000'
    os.environ['SYNTHEFY_DISABLE_PIPELINE_BATCHING']='1'
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.70)


def context_indices(bundle,spec,smoke=False):
    n=len(bundle['y_fit'])
    eligible=np.arange(n)
    if spec.get('mature'):
        counts=bundle['ann_fit'].ANN_BLOCK_24_R1_N.to_numpy(float)
        eligible=np.flatnonzero((counts>0)&(bundle['fit_month']>=np.min(bundle['fit_month'])+24))
        if len(eligible)<100:
            raise ValueError('Insufficient source records with mature 24-month history')
    if spec['family'].startswith('tabdpt13'):
        return eligible # Author predicts with retrieval or seeded subsampling within this source pool.
    count=min(len(eligible),512 if smoke else spec['context'])
    # Neither source rents nor query rents determine context membership.
    return np.sort(np.random.default_rng(42).choice(eligible,count,replace=False))


def frame_inputs(bundle,spec,indices):
    mode='residual' if spec['target']=='ratio' else spec['target']
    x,q,base,offset=inputs(bundle,{'adjusted':True,'target':mode})
    columns=[c for c in x if x[c].iloc[indices].notna().any() and x[c].iloc[indices].nunique(dropna=False)>1]
    x,q=x[columns].copy(),q[columns].copy()
    for column in x.select_dtypes(include=['category','string','object']).columns:
        x[column]=x[column].astype('string').fillna('__MISSING__').astype(str)
        q[column]=q[column].astype('string').fillna('__MISSING__').astype(str)
    if spec['target']=='ratio':
        if not (np.isfinite(base).all() and np.isfinite(offset).all() and (base>0).all() and (offset>0).all()):
            raise ValueError('Invalid source-derived baseline for ratio target')
        target=bundle['y_fit']/base
    else:
        target=bundle['y_fit']-base
    return x,q,target,offset


def numeric_arrays(x,q):
    encoder=NativeEncoder().fit(x)
    cats,nums=encoder.transform(x);qc,qn=encoder.transform(q)
    return encoder,np.concatenate([nums,cats.astype(np.float32)],axis=1),np.concatenate([qn,qc.astype(np.float32)],axis=1)


def nori_config(family,members):
    import synthefy_nori
    source=ROOT/'.cache/transformer-deps/synthefy_nori/configs'/('model_v3_inference.json' if family=='nori15' else 'default_inference.json')
    configurations=json.loads(source.read_text())[:members]
    for pipeline in configurations:
        if 'HighDimFeatureSelector' in pipeline:
            pipeline['HighDimFeatureSelector']['fit_on_test']=False
    path=OUT/f'{family}_inference_{members}.json'
    write_json(path,configurations)
    return path


def fit_predict(bundle,spec,smoke=False):
    indices=context_indices(bundle,spec,smoke)
    x,q,target,offset=frame_inputs(bundle,spec,indices)
    if smoke:
        # Evenly spread query IDs; this pilot diagnoses API/memory/runtime, not model selection.
        rows=np.linspace(0,len(q)-1,128,dtype=int)
    else:
        rows=np.arange(len(q))
    q=q.iloc[rows];offset=offset[rows]
    manifest=json.loads((OUT/'checkpoint_manifest.json').read_text())
    family=spec['family'];encoding=None
    members=1 if smoke else spec['members']
    context=256 if smoke else spec['context']
    checkpoint=ROOT/manifest['tabdpt13' if family.startswith('tabdpt13') else family]['path'] if family!='tabicl2' else ROOT/'.cache/foundation-models/tabicl-regressor-v2-20260212.ckpt'
    adaptation={}
    if family.startswith('tabdpt13'):
        from tabdpt import TabDPTRegressor
        encoding,numbers,query=numeric_arrays(x.iloc[indices],q)
        model=TabDPTRegressor(model_weight_path=str(checkpoint),device='cuda',compile=False,use_flash=False,
            context_reduction=spec['reduction'],verbose=False)
        model.fit(numbers,target[indices].astype(np.float32))
        if spec.get('adapt'):
            from analysis.experiments.seventh_wave_adaptation import adapt
            epochs=None if bundle['start']=='2023-10' else json.loads((OUT/'screen'/spec['name']/'results.json').read_text())['training_epochs']
            adaptation=adapt(model,target[indices],query,bundle,offset,spec,OUT,epochs)
        started=time.perf_counter()
        query_batch=32 if spec['reduction']=='retrieval' else 2048
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=spec.get('amp',False)):
            prediction=model.predict(query,n_ensembles=members,context_size=context,batch_size=query_batch,seed=42)
        extras={'prediction_seconds':time.perf_counter()-started,'context_available_rows':len(indices),
            'context_rows_per_view':context,'query_batch_size':query_batch,
            'mixed_precision':spec.get('amp',False),
            'attention_backend':'Author use_flash=False, supported PyTorch math backend on Windows.',
            'transductive_preprocessing':'Author per-forward normalization; fitting imputer/scaler is source-only.'}
    elif family.startswith('nori'):
        from synthefy_nori import NoriRegressor
        model=NoriRegressor(model_path=str(checkpoint),model='nori-1-5' if family=='nori15' else 'nori-6m',
            device='cuda:0',inference_config=str(nori_config(family,members)),
            memory_policy={'allow_quantization':False,'gpu_budget_frac':.50,
                'allow_subsample':False,'elements_budget':int((len(indices)+768)*len(x.columns))},
            augmentations=(),large_context_policy=None)
        model.fit(x.iloc[indices],target[indices])
        started=time.perf_counter()
        prediction=model.predict(q,output_type='mean')
        extras={'prediction_seconds':time.perf_counter()-started,'context_available_rows':len(indices),
            'context_rows_per_view':len(indices),'memory_report':model.memory_report_,
            'query_batch_size':'Native chunks from (context+768)*source-column element budget; pipeline batching disabled.',
            'transductive_preprocessing':'Native inductive preprocessing; HighDimFeatureSelector.fit_on_test=False. Augmentations disabled; continuous distribution mean.'}
    else:
        from tabicl import TabICLRegressor
        model=TabICLRegressor(n_estimators=members,batch_size=1,kv_cache=True,
            model_path=str(checkpoint),allow_auto_download=False,device='cuda',
            norm_methods=['none'] if members==1 else ['none','power'],use_amp=True,use_fa3=False,
            offload_mode='auto',disk_offload_dir=str(ROOT/'.cache/seventh_wave_offload'),n_jobs=4,
            random_state=42,verbose=False)
        model.fit(x.iloc[indices],target[indices])
        started=time.perf_counter();parts=[]
        for begin in range(0,len(q),512):
            parts.append(np.asarray(model.predict(q.iloc[begin:begin+512],output_type='mean'),float))
            if begin%4096==0:
                print(f'  TabICLv2 {min(begin+512,len(q))}/{len(q)}',flush=True)
        prediction=np.concatenate(parts)
        extras={'prediction_seconds':time.perf_counter()-started,'context_available_rows':len(indices),
            'context_rows_per_view':len(indices),'query_batch_size':512,
            'transductive_preprocessing':'Author TabICL native numeric preprocessing.'}
    prediction=np.asarray(prediction,dtype=float).reshape(-1)
    prediction=prediction*offset if spec['target']=='ratio' else prediction+offset
    if len(prediction)!=len(q) or not np.isfinite(prediction).all():
        raise ValueError('Invalid Transformer predictions')
    payload={'model':model,'encoding':encoding,'columns':list(x),'context_indices':indices,
        'spec':spec,'checkpoint_sha256':sha256(checkpoint),'smoke':smoke}
    return payload,prediction,rows,extras|adaptation


def run(stage,names=None,smoke=False):
    enable();bundle=make_data(stage)
    write_json(OUT/'plan.json',{'specs':SPECS,'screen':['2023-10','2024-03'],
        'later':{'outer':['2024-04','2025-03'],'june':['2024-06','2025-03']},
        'notes':'Local inference from public author weights. Frozen source labels. Native preprocessing and finite context budgets disclosed per model. Smoke is 128 queries and never used to rank accuracy.'})
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination=OUT/('smoke' if smoke else stage)/spec['name']
        if (destination/'results.json').exists():
            continue
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();print('TRANSFORMER START',stage,spec['name'],'smoke',smoke,flush=True)
        payload,pred,rows,extras=fit_predict(bundle,spec,smoke)
        artifact=ART/('smoke' if smoke else stage)/spec['name'];artifact.mkdir(parents=True,exist_ok=True)
        if spec.get('adapt'):
            state={name:value.detach().cpu() for name,value in payload['model'].model.state_dict().items()}
            torch.save(state,artifact/'adapted_state.pt')
            extras['adapted_state_sha256']=sha256(artifact/'adapted_state.pt')
        pd.DataFrame({'source_row':bundle['row_val'][rows],'month':bundle['date_val'][rows],
            'actual':bundle['y_val'][rows],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
        # Reconstruct foundation estimators from source rows and checkpoint rather than pickle CUDA/KV caches.
        recipe={k:v for k,v in payload.items() if k!='model'}
        joblib.dump(recipe,artifact/'recipe.joblib',compress=1)
        result={'spec':spec,'stage':stage,'smoke':smoke,'query_rows':len(rows),
            'source_rows':len(bundle['y_fit']),'features':len(payload['columns']),
            **score(bundle['y_val'][rows],pred),**extras,'seconds':time.perf_counter()-started,
            'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'source_sha256':bundle['hashes'],
            'checkpoint_sha256':payload['checkpoint_sha256']}
        write_json(destination/'results.json',result)
        print('TRANSFORMER RESULT',stage,spec['name'],result['rmse_sgd'],'seconds',result['seconds'],flush=True)
        del payload;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.stage,args.names,args.smoke)

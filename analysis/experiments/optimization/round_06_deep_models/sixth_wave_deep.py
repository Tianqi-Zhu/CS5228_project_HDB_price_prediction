"""Author deep-tabular models on historical HDB features and time validation."""
from __future__ import annotations
import argparse
import contextlib
import gc
import importlib.util
import json
import sys
import time
import types
import warnings
from dataclasses import asdict
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data,inputs
from analysis.pipeline.neural_models import NeuralFrameEncoder

OUT=ROOT/'output/experiment_runs/round_06_deep_models'
ART=ROOT/'output/models/round_06_deep_models'


def dependencies():
    for path in (ROOT/'.cache/deep-deps',ROOT/'.cache/model-deps'):
        if str(path) not in sys.path:
            sys.path.insert(0,str(path))
    warnings.filterwarnings('ignore',message=r'The .* feature has just two bin edges.*')


def load_realmlp():
    name='cs5228_author_realmlp'
    if name not in sys.modules:
        path=ROOT/'.cache/deep-deps/pytabkit/models/realmlp'
        spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_nca():
    base=ROOT/'.cache/deep-sources/TALENT'
    for name,path in [('TALENT',base),('TALENT.model',base/'model'),
        ('TALENT.model.models',base/'model/models'),('TALENT.model.lib',base/'model/lib'),
        ('TALENT.model.lib.tabr',base/'model/lib/tabr')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    for name,path in [('TALENT.model.lib.tabr.utils',base/'model/lib/tabr/utils.py'),
        ('TALENT.model.models.modernNCA',base/'model/models/modernNCA.py')]:
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module
            spec.loader.exec_module(module)
    return sys.modules['TALENT.model.models.modernNCA'].ModernNCA


def seed(seed_value=42):
    dependencies()
    np.random.seed(seed_value);torch.manual_seed(seed_value);torch.cuda.manual_seed_all(seed_value)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.65)


class NativeEncoder:
    """Learn categories on fitting rows; leave NaNs for native RealMLP."""
    def fit(self,x):
        self.categorical=list(x.select_dtypes(include=['category','object','string']).columns)
        self.numeric=[c for c in x if c not in self.categorical]
        self.levels={c:{v:i+1 for i,v in enumerate(sorted(x[c].astype('string').dropna().unique()))}
            for c in self.categorical}
        self.cardinalities=[len(self.levels[c])+1 for c in self.categorical]
        return self
    def transform(self,x):
        cats=np.column_stack([x[c].astype('string').map(self.levels[c]).fillna(0).to_numpy(np.int64)
            for c in self.categorical])
        nums=x[self.numeric].to_numpy(np.float32)
        if np.isinf(nums).any():
            raise ValueError('Infinite native numerical input')
        return cats,nums


class MLPPLR(nn.Module):
    def __init__(self,n_numeric,cardinalities):
        super().__init__()
        import rtdl_num_embeddings as num
        from rtdl_revisiting_models import MLP
        self.numerical=num.PeriodicEmbeddings(n_numeric,d_embedding=16,n_frequencies=32,
            frequency_init_scale=.05,activation=True,lite=True)
        self.embeddings=nn.ModuleList(nn.Embedding(size,8) for size in cardinalities)
        self.network=MLP(d_in=n_numeric*16+len(cardinalities)*8,d_out=1,n_blocks=3,d_block=256,dropout=.1)
    def forward(self,nums,cats):
        values=[self.numerical(nums).flatten(1)]+[layer(cats[:,i]) for i,layer in enumerate(self.embeddings)]
        return self.network(torch.cat(values,dim=1))


SPECS=[]
for embedding,target,lr in [('plain','raw',.002),('plain','residual',.002),('plain','residual',.0005),
    ('ple','raw',.002),('ple','residual',.002),('ple','residual',.0005)]:
    SPECS.append({'name':f'tabm_{embedding}_{target}_lr{lr}','family':'tabm','embedding':embedding,
        'target':target,'lr':lr,'epochs':60,'patience':10,'seed':42})
for target in ('raw','residual'):
    SPECS.append({'name':f'mlp_plr_{target}','family':'mlp_plr','target':target,
        'lr':.0003,'epochs':60,'patience':10,'seed':42})
for target,ens,lr in [('raw',1,.2),('residual',1,.2),('residual',4,.04)]:
    SPECS.append({'name':f'realmlp_{target}_ens{ens}','family':'realmlp','target':target,
        'members':ens,'lr':lr,'epochs':128,'patience':12,'seed':42})
for target,embedding in [('raw',False),('residual',False),('residual',True)]:
    SPECS.append({'name':f'modernnca_{target}_{"plr" if embedding else "plain"}',
        'family':'modernnca','target':target,'embedding':embedding,'context_budget':4096,
        'lr':.001,'epochs':50,'patience':8,'seed':42})
SPECS.append({'name':'modernnca_author_default_residual','family':'modernnca','target':'residual',
    'embedding':True,'context_budget':8192,'lr':.01,'epochs':50,'patience':8,'seed':42,'author_defaults':True})
SEED_GROUPS={name:[name]+[name+f'_seed{seed}' for seed in (43,44)]
    for name in ('tabm_ple_residual_lr0.0005','realmlp_residual_ens4')}
for base_name,members in SEED_GROUPS.items():
    original=next(spec for spec in SPECS if spec['name']==base_name)
    for name,seed_value in zip(members[1:],(43,44)):
        SPECS.append(original|{'name':name,'seed':seed_value,'seed_group':base_name})


def prepare(bundle,spec,native=False):
    x,q,base,offset=inputs(bundle,{'adjusted':True,'target':spec['target'],
        'drop_ann36':spec.get('drop_ann36',False)})
    # Fit-only schema decisions: an all-missing/constant field has no trained effect.
    columns=[c for c in x if x[c].notna().any() and x[c].nunique(dropna=False)>1]
    x,q=x[columns],q[columns]
    categories=list(x.select_dtypes(include=['category','object','string']).columns)
    encoder=NativeEncoder().fit(x) if native else NeuralFrameEncoder(categories,True).fit(x)
    cats,nums=encoder.transform(x);qc,qn=encoder.transform(q)
    return encoder,columns,cats,nums,qc,qn,bundle['y_fit']-base,offset


def tensor_arrays(cats,nums,device='cuda'):
    return torch.as_tensor(cats,device=device),torch.as_tensor(nums,device=device)


def nca_array(nums,cats,cardinalities):
    return torch.cat([nums]+[torch.nn.functional.one_hot(cats[:,i],size).float()
        for i,size in enumerate(cardinalities)],dim=1)


def sample_nca_context(n_rows,query_indices,budget):
    candidate=torch.randperm(n_rows,device=query_indices.device)[:budget+len(query_indices)]
    return candidate[~torch.isin(candidate,query_indices)][:budget]


def nca_encode(model,x):
    if model.num_embeddings is not None and model.d_num>0:
        x=torch.cat([model.num_embeddings(x[:,:model.d_num]).flatten(1),x[:,model.d_num:]],dim=1)
    z=model.encoder(x)
    return model.post_encoder(z) if model.n_blocks>0 else z


@torch.inference_mode()
def predict_network(model,nums,cats,family,*,source=None,source_y=None,cardinalities=None):
    model.eval()
    predictions=[]
    if family=='modernnca':
        encoded=torch.cat([nca_encode(model,source[i:i+2048]) for i in range(0,len(source),2048)])
        query=nca_array(nums,cats,cardinalities)
        for i in range(0,len(query),128):
            z=nca_encode(model,query[i:i+128])
            weights=torch.softmax(-torch.cdist(z,encoded)/model.T,dim=1)
            predictions.append((weights@source_y).float().cpu().numpy())
    else:
        for i in range(0,len(nums),512):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=model(nums[i:i+512],cats[i:i+512])
            value=value.squeeze(-1).float()
            if family=='tabm':
                value=value.mean(dim=1)
            predictions.append(value.cpu().numpy())
    return np.concatenate(predictions).astype(float)


def fit_real(spec,bundle,epochs=None):
    api=load_realmlp()
    encoder,columns,cats,nums,qc,qn,target,offset=prepare(bundle,spec,native=True)
    cfg=api.RealMLPConfig.td_reg(n_ens=spec['members'],lr=spec['lr'],n_epochs=spec['epochs'],
        batch_size=256 if spec['members']==1 else 512,predict_batch_size=1024,
        use_early_stopping=True,early_stopping_additive_patience=spec['patience'],
        early_stopping_multiplicative_patience=1,verbosity=2)
    if epochs is None:
        all_nums=np.concatenate([nums,qn]);all_cats=np.concatenate([cats,qc])
        y=np.r_[target,bundle['y_val']-offset]
        val=torch.arange(len(nums),len(all_nums))[None]
    else:
        all_nums,all_cats,y=nums,cats,target
        val=None
    log=OUT/'logs'/f"{bundle['start']}_{spec['name']}.log"
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream):
        model,info=api.fit(cfg,torch.as_tensor(all_nums),torch.as_tensor(all_cats),encoder.cardinalities,
            torch.as_tensor(y,dtype=torch.float32)[:,None],train_idxs=torch.arange(len(nums))[None],
            val_idxs=val,stop_epochs=[epochs] if epochs else None,seed=spec['seed'],device='cuda')
    selected=epochs or int(np.asarray(info['best_mean_epochs']).flatten()[0])
    prediction=api.predict(model,torch.as_tensor(qn),torch.as_tensor(qc))[0].reshape(-1).detach().cpu().numpy().astype(float)+offset
    payload={'model':model.to('cpu'),'encoder':encoder,'columns':columns,'spec':spec,
        'realmlp_config':asdict(cfg),'training_info':info}
    return payload,prediction,int(selected)


def fit_network(spec,bundle,epochs=None):
    encoder,columns,cats,nums,qc,qn,target,offset=prepare(bundle,spec)
    cats,nums=tensor_arrays(cats,nums);qc,qn=tensor_arrays(qc,qn)
    mean,std=float(target.mean()),max(1.,float(target.std()))
    y=torch.as_tensor((target-mean)/std,dtype=torch.float32,device='cuda')
    source=None
    family=spec['family']
    if family=='tabm':
        import tabm,rtdl_num_embeddings as numerical
        embeddings=None
        if spec['embedding']=='ple':
            bins=numerical.compute_bins(nums.detach().cpu(),n_bins=48)
            embeddings=numerical.PiecewiseLinearEmbeddings(bins,16,activation=False,version='B')
        model=tabm.TabM.make(n_num_features=nums.shape[1],cat_cardinalities=encoder.cardinalities,
            num_embeddings=embeddings,d_out=1,k=32,n_blocks=3,d_block=256,dropout=.1,arch_type='tabm')
    elif family=='mlp_plr':
        model=MLPPLR(nums.shape[1],encoder.cardinalities)
    else:
        cls=load_nca()
        source=nca_array(nums,cats,encoder.cardinalities)
        embedding=({'type':'PLREmbeddings','n_frequencies':77 if spec.get('author_defaults') else 32,
            'frequency_scale':.04431360576139521 if spec.get('author_defaults') else .05,
            'd_embedding':34 if spec.get('author_defaults') else 16,'lite':True} if spec['embedding'] else None)
        model=cls(d_in=source.shape[1],d_num=nums.shape[1],d_out=1,dim=128,
            dropout=.1,d_block=512 if spec.get('author_defaults') else 256,
            n_blocks=0 if spec.get('author_defaults') else 1,num_embeddings=embedding,
            temperature=1.,sample_rate=.5 if spec.get('author_defaults') else .8)
    model=model.to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=.0002 if spec.get('author_defaults') else .0003)
    best=np.inf;state=None;best_epoch=1;stale=0;history=[]
    batch_size=256 if family in ('tabm','modernnca') else 512
    for epoch in range(1,(epochs or spec['epochs'])+1):
        model.train()
        order=torch.randperm(len(y),device='cuda')
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            if len(indices)<2:
                continue
            optimizer.zero_grad(set_to_none=True)
            if family=='modernnca':
                context=sample_nca_context(len(y),indices,spec['context_budget'])
                output=model(source[indices],y[indices],source[context],y[context],True)
                loss=(output-y[indices]).square().mean()
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(nums[indices],cats[indices]).squeeze(-1).float()
                    loss=(output-y[indices,None]).square().mean() if family=='tabm' else (output-y[indices]).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epochs is None:
            prediction=predict_network(model,qn,qc,family,source=source,source_y=y,
                cardinalities=encoder.cardinalities)*std+mean+offset
            rmse=score(bundle['y_val'],prediction)['rmse_sgd']
            history.append({'epoch':epoch,'rmse_sgd':rmse})
            if rmse<best-.02:
                best,stale,best_epoch=rmse,0,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale+=1
            if epoch%5==0:
                print(f"  {spec['name']} epoch {epoch}: RMSE {rmse:.3f}, best {best:.3f}",flush=True)
            if stale>=spec['patience']:
                break
        elif epoch%5==0:
            print(f"  fixed {spec['name']} epoch {epoch}/{epochs}",flush=True)
    if state is not None:
        model.load_state_dict(state)
    prediction=predict_network(model,qn,qc,family,source=source,source_y=y,
        cardinalities=encoder.cardinalities)*std+mean+offset
    payload={'model':model.to('cpu'),'encoder':encoder,'columns':columns,'spec':spec,'mean':mean,'std':std,
        'history':history,'context_nums':nums.detach().cpu() if family=='modernnca' else None,
        'context_cats':cats.detach().cpu() if family=='modernnca' else None,
        'context_y':y.detach().cpu() if family=='modernnca' else None}
    return payload,prediction,int(epochs or best_epoch)


def predict_saved(payload,bundle):
    spec=payload['spec'];family=spec['family']
    _,q,_,offset=inputs(bundle,{'adjusted':True,'target':spec['target']})
    cats,nums=payload['encoder'].transform(q[payload['columns']])
    model=payload['model'].to('cuda')
    if family=='realmlp':
        prediction=load_realmlp().predict(model,torch.as_tensor(nums),torch.as_tensor(cats))[0].reshape(-1).detach().cpu().numpy().astype(float)+offset
    else:
        cats,nums=tensor_arrays(cats,nums)
        source=None;source_y=None
        if family=='modernnca':
            source=nca_array(payload['context_nums'].to('cuda'),payload['context_cats'].to('cuda'),payload['encoder'].cardinalities)
            source_y=payload['context_y'].to('cuda')
        prediction=predict_network(model,nums,cats,family,source=source,source_y=source_y,
            cardinalities=payload['encoder'].cardinalities)*payload['std']+payload['mean']+offset
    model.to('cpu');torch.cuda.empty_cache()
    return prediction


def load_payload(path):
    dependencies();load_realmlp();load_nca()
    # Compatibility with the first three pilot artifacts executed under -m.
    main=sys.modules.get('__main__')
    for cls in (MLPPLR,NativeEncoder):
        if main is not None and not hasattr(main,cls.__name__):
            setattr(main,cls.__name__,cls)
    return joblib.load(path)


def run(stage,names=None):
    dependencies()
    write_json(OUT/'plan.json',{'specs':SPECS,'selection_months':['2023-10','2024-03'],
        'later_windows':{'outer':['2024-04','2025-03'],'june':['2024-06','2025-03']},
        'data':'train + supplied static auxiliary, historical labels frozen at origin',
        'implementations':{'TabM':'author tabm 0.0.3','MLP-PLR':'author RTDL embeddings and MLP composed by project adapter',
            'RealMLP':'author experimental standalone implementation, pytabkit 1.8.0',
            'ModernNCA':'unmodified author source, own capped context training; full fitting-context evaluation'},
        'notes':'All query records retained. Fit-only vocabulary/numeric transforms; later epoch counts fixed from earlier screen. ModernNCA context never includes the current training batch, nor validation labels.'})
    bundle=make_data(stage)
    for spec in SPECS:
        name=spec['name']
        if names and name not in names:
            continue
        directory=OUT/stage/name
        if (directory/'results.json').exists():
            continue
        seed(spec['seed']);torch.cuda.reset_peak_memory_stats()
        epochs=None if stage=='screen' else json.loads((OUT/'screen'/name/'results.json').read_text())['epochs']
        print('STRONG DEEP MODEL',stage,name,flush=True)
        started=time.perf_counter()
        payload,pred,selected=(fit_real if spec['family']=='realmlp' else fit_network)(spec,bundle,epochs)
        artifact=ART/stage/name;artifact.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,artifact/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
            'actual':bundle['y_val'],'prediction':np.asarray(pred,dtype=float)}).to_csv(artifact/'predictions.csv',index=False)
        record={'spec':spec,'stage':stage,'epochs':selected,'features':len(payload['columns']),
            'fit_rows':len(bundle['y_fit']),'query_rows':len(pred),**score(bundle['y_val'],pred),
            'seconds':time.perf_counter()-started,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,
            'source_sha256':bundle['hashes']}
        write_json(directory/'results.json',record)
        print('STRONG DEEP RESULT',stage,name,record['rmse_sgd'],'epochs',selected,flush=True)
        del payload;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    # Stable class module names in serialized preprocessing/model artifacts.
    from analysis.experiments.sixth_wave_deep import run as run_registered
    run_registered(args.stage,args.names)

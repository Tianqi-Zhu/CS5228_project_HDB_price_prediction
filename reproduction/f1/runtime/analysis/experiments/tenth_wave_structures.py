"""Controlled numerical-token, explicit-cross and expert-network exploration."""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.sixth_wave_deep import seed,prepare,tensor_arrays,dependencies

OUT=ROOT/'reports/experiments/tenth_wave'
ART=ROOT/'output/models/tenth_wave'
SPECS=[]
for embedding,width,blocks in [('linear',96,2),('ple',96,2),('ple',128,3),('plr',96,2),('plr',128,3)]:
    SPECS.append({'name':f'ft_{embedding}_{width}_{blocks}','family':'ft_'+embedding,
        'embedding':embedding,'width':width,'blocks':blocks,'lr':.0003 if width==96 else .0001})
SPECS += [{'name':'dcn_plain_2','family':'dcn','embedding':'linear','cross_layers':2,'lr':.0005},
    {'name':'dcn_plr_3','family':'dcn','embedding':'plr','cross_layers':3,'lr':.0003},
    {'name':'moe_plr_4','family':'moe','embedding':'plr','experts':4,'lr':.0005},
    {'name':'moe_plr_8','family':'moe','embedding':'plr','experts':8,'lr':.0005}]
SPECS=[spec|{'target':'residual','drop_ann36':True,'seed':42,'max_epochs':16,'patience':4} for spec in SPECS]


def numerical_module(kind,nums,dimension):
    import rtdl_num_embeddings as numerical
    if kind=='linear':return numerical.LinearEmbeddings(nums.shape[1],dimension)
    if kind=='ple':
        bins=numerical.compute_bins(torch.as_tensor(nums),n_bins=min(48,len(nums)-1))
        return numerical.PiecewiseLinearEmbeddings(bins,dimension,activation=False,version='B')
    return numerical.PeriodicEmbeddings(nums.shape[1],dimension,n_frequencies=16,
        frequency_init_scale=.05,activation=True,lite=True)


class CrossLayer(nn.Module):
    """DCN-V2 full-matrix cross: x + x0 * (W x + b)."""
    def __init__(self,dimension):
        super().__init__();self.linear=nn.Linear(dimension,dimension)
        nn.init.xavier_uniform_(self.linear.weight,gain=.1);nn.init.zeros_(self.linear.bias)
    def forward(self,x0,x):return x+x0*self.linear(x)


class FlatEmbedding(nn.Module):
    def __init__(self,nums,cardinalities,kind):
        super().__init__();self.numeric=numerical_module(kind,nums,8)
        self.categorical=nn.ModuleList(nn.Embedding(size,8) for size in cardinalities)
        self.dimension=(nums.shape[1]+len(cardinalities))*8
    def forward(self,nums,cats):
        return torch.cat([self.numeric(nums).flatten(1)]+
            [layer(cats[:,i]) for i,layer in enumerate(self.categorical)],dim=1)


class DeepCross(nn.Module):
    def __init__(self,nums,cardinalities,spec):
        super().__init__();self.embedding=FlatEmbedding(nums,cardinalities,spec['embedding'])
        dimension=self.embedding.dimension
        self.cross=nn.ModuleList(CrossLayer(dimension) for _ in range(spec['cross_layers']))
        self.deep=nn.Sequential(nn.Linear(dimension,192),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(192,128),nn.ReLU(),nn.Dropout(.1))
        self.output=nn.Linear(dimension+128,1)
    def forward(self,nums,cats):
        x0=self.embedding(nums,cats);x=x0
        for cross in self.cross:x=cross(x0,x)
        return self.output(torch.cat([x,self.deep(x0)],dim=1)).squeeze(-1)


class ExpertNetwork(nn.Module):
    """Shared numerical/categorical encoder with learned soft expert gates."""
    def __init__(self,nums,cardinalities,spec):
        super().__init__();self.embedding=FlatEmbedding(nums,cardinalities,spec['embedding'])
        self.shared=nn.Sequential(nn.Linear(self.embedding.dimension,128),nn.LayerNorm(128),nn.ReLU())
        self.experts=nn.ModuleList(nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(128,1)) for _ in range(spec['experts']))
        self.gate=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,spec['experts']))
    def components(self,nums,cats):
        shared=self.shared(self.embedding(nums,cats))
        probabilities=torch.softmax(self.gate(shared).float(),dim=1)
        outputs=torch.cat([expert(shared) for expert in self.experts],dim=1).float()
        return probabilities,outputs
    def forward(self,nums,cats):
        probabilities,outputs=self.components(nums,cats)
        return (probabilities*outputs).sum(dim=1)


def build(spec,nums,cardinalities):
    if spec['family'].startswith('ft_'):
        from rtdl_revisiting_models import FTTransformer
        model=FTTransformer(n_cont_features=nums.shape[1],cat_cardinalities=cardinalities,
            d_out=1,n_blocks=spec['blocks'],d_block=spec['width'],attention_n_heads=8,
            attention_dropout=.2,ffn_d_hidden=None,ffn_d_hidden_multiplier=4/3,
            ffn_dropout=.1,residual_dropout=0.)
        if spec['embedding']!='linear':
            model.cont_embeddings=numerical_module(spec['embedding'],nums,spec['width'])
        return model
    if spec['family']=='dcn':return DeepCross(nums,cardinalities,spec)
    return ExpertNetwork(nums,cardinalities,spec)


@torch.inference_mode()
def inference(model,cats,nums):
    model.eval();values=[]
    for start in range(0,len(nums),512):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(nums[start:start+512],cats[start:start+512])
        values.append(output.reshape(-1).float().cpu().numpy())
    return np.concatenate(values).astype(float)


def fit(spec,bundle,epochs=None):
    seed(spec['seed'])
    encoder,columns,cats,nums,qc,qn,target,offset=prepare(bundle,spec)
    model=build(spec,nums,encoder.cardinalities).to('cuda')
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups=model.make_parameter_groups() if spec['family'].startswith('ft_') else model.parameters()
    optimizer=torch.optim.AdamW(groups,lr=spec['lr'],weight_decay=.001)
    cats,nums=tensor_arrays(cats,nums);qc,qn=tensor_arrays(qc,qn)
    mean,std=float(target.mean()),max(1.,float(target.std()))
    y=torch.as_tensor((target-mean)/std,device='cuda',dtype=torch.float32)
    best=np.inf;state=None;best_epoch=1;stale=0;history=[]
    for epoch in range(1,(epochs or spec['max_epochs'])+1):
        model.train();order=torch.randperm(len(y),device='cuda')
        for start in range(0,len(order),256 if spec['family'].startswith('ft_') else 512):
            size=256 if spec['family'].startswith('ft_') else 512
            indices=order[start:start+size]
            if len(indices)<2:continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if spec['family']=='moe':
                    gates,outputs=model.components(nums[indices],cats[indices])
                    output=(gates*outputs).sum(dim=1)
                    balance=spec['experts']*torch.sum((gates.mean(dim=0)-1/spec['experts'])**2)
                else:output=model(nums[indices],cats[indices]).reshape(-1).float();balance=0.
                loss=(output-y[indices]).square().mean()+.001*balance
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epochs is None:
            prediction=inference(model,qc,qn)*std+mean+offset
            rmse=score(bundle['y_val'],prediction)['rmse_sgd']
            history.append({'epoch':epoch,'rmse_sgd':rmse})
            if rmse<best-.02:
                best,best_epoch,stale=rmse,epoch,0
                state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:stale+=1
            print('STRUCTURE EPOCH',spec['name'],epoch,round(rmse,4),'best',round(best,4),flush=True)
            if stale>=spec['patience']:break
        elif epoch%5==0:print('FIXED STRUCTURE',spec['name'],epoch,epochs,flush=True)
    if state is not None:model.load_state_dict(state)
    prediction=inference(model,qc,qn)*std+mean+offset
    payload={'model':model.to('cpu'),'encoder':encoder,'columns':columns,'mean':mean,'std':std,
        'spec':spec,'history':history,'parameters':parameters}
    return payload,prediction,int(epochs or best_epoch)


def predict_saved(payload,bundle):
    from analysis.experiments.third_wave_training import inputs
    _,q,_,offset=inputs(bundle,{'adjusted':True,'target':'residual','drop_ann36':True})
    cats,nums=payload['encoder'].transform(q[payload['columns']])
    cats,nums=tensor_arrays(cats,nums);model=payload['model'].to('cuda')
    prediction=inference(model,cats,nums)*payload['std']+payload['mean']+offset
    model.to('cpu');torch.cuda.empty_cache()
    return prediction


def run(stage):
    dependencies()
    write_json(OUT/'plan.json',{'specs':SPECS,'data':'train plus supplied auxiliary; no external rental labels',
        'inputs':'same adjusted history features, five ANN36 fields removed in every new configuration',
        'selection':'Earlier full window selects epochs and one representative per structural route; later fixed epochs.',
        'implementations':{'FT':'Author RTDL FT backbone; author rtdl-num-embeddings PLE/PLR numerical tokens substituted.',
            'DCN':'Project implementation of DCN-V2 full-matrix cross equation with parallel deep branch.',
            'MoE':'Project shared encoder, soft learned gate and four/eight regression experts; load balance penalty .001.'},
        'sources':{'numerical_embeddings':'https://arxiv.org/abs/2203.05556',
            'DCNv2':'https://arxiv.org/abs/2008.13535',
            'FT':'https://arxiv.org/abs/2106.11959',
            'author_embedding_docs':'https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md'},
        'notes':'All windows reused development data. Network changes are tested empirically, not presumed superior.'})
    bundle=make_data(stage)
    specs=SPECS
    if stage!='screen':
        chosen=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['representatives']
        specs=[spec for spec in SPECS if spec['name'] in chosen.values()]
    for spec in specs:
        path=OUT/stage/spec['name']/'results.json'
        if path.exists():continue
        epochs=None if stage=='screen' else json.loads((OUT/'screen'/spec['name']/'results.json').read_text(encoding='utf-8'))['epochs']
        print('STRUCTURE',stage,spec['name'],flush=True);started=time.perf_counter()
        payload,prediction,count=fit(spec,bundle,epochs)
        folder=ART/stage/spec['name'];folder.mkdir(parents=True,exist_ok=True)
        joblib.dump(payload,folder/'model.joblib',compress=1)
        pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],'actual':bundle['y_val'],
            'prediction':prediction}).to_csv(folder/'predictions.csv',index=False)
        record={'stage':stage,'name':spec['name'],'spec':spec,'epochs':count,
            'parameters':payload['parameters'],'features':len(payload['columns']),
            'fit_rows':len(bundle['y_fit']),'rows':len(prediction),**score(bundle['y_val'],prediction),
            'seconds':time.perf_counter()-started,'source_sha256':bundle['hashes']}
        write_json(path,record);print('STRUCTURE RESULT',stage,spec['name'],record['rmse_sgd'],flush=True)
        del payload;gc.collect();torch.cuda.empty_cache()
    if stage=='screen':
        records=[json.loads((OUT/stage/spec['name']/'results.json').read_text(encoding='utf-8')) for spec in SPECS]
        representatives={}
        for record in sorted(records,key=lambda r:r['rmse_sgd']):
            representatives.setdefault(record['spec']['family'],record['name'])
        path=OUT/'selection.json'
        if not path.exists():write_json(path,{'representatives':representatives,'records':records,
            'decision_before_later_evaluation':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['screen','outer','june'])
    from analysis.experiments.tenth_wave_structures import run as run_registered
    run_registered(parser.parse_args().stage)

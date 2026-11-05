"""Official TabM with optional numerical embeddings and learned address embeddings."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib,tabm,rtdl_num_embeddings as ne
from torch import nn
from analysis.experiments.embedding_rent_search import prepare
from analysis.pipeline.config import DATA
from analysis.pipeline.data import check_output_path,sha256
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.evaluation import metrics


class Model(nn.Module):
    def __init__(self,cards,n,bins,spec):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(card,min(12,max(3,int(card**.25*1.6)))) for card in cards])
        # B initializes the piecewise weights at zero. ReLU here would make
        # that entire branch's gradient zero; keep its correction linear.
        dim=spec.get('num_embedding_dim',8)
        if spec.get('encoding')=='periodic':
            self.num=ne.PeriodicEmbeddings(n,dim,n_frequencies=16,
                frequency_init_scale=spec.get('frequency_scale',.1),activation=spec.get('lite',False),lite=spec.get('lite',False))
        else:
            self.num = ne.PiecewiseLinearEmbeddings(bins,dim,activation=False,version='B') if bins is not None else nn.Identity()
        nnum = n*dim if bins is not None or spec.get('encoding')=='periodic' else n
        self.mixture=spec.get('mixture',False)
        self.backbone=tabm.TabM.make(n_num_features=nnum+sum(e.embedding_dim for e in self.embeddings),
                 cat_cardinalities=[],d_out=6 if self.mixture else 1,k=spec['k'],d_block=spec['width'],n_blocks=spec['blocks'],
                 dropout=spec.get('dropout',.1),start_scaling_init='normal' if bins is not None or spec.get('encoding')=='periodic' else 'random-signs')
    def forward(self,c,n):
        nums=self.num(n).flatten(1)
        output=self.backbone(torch.cat([nums,*[e(c[:,i]) for i,e in enumerate(self.embeddings)]],dim=1))
        if self.mixture:
            self.gate_logits=output[...,3:]
            self.expert_predictions=output[...,:3]
            return (self.gate_logits.softmax(-1)*self.expert_predictions).sum(-1)
        return output.squeeze(-1)


def predict(model,c,n,log_mean=None,log_std=None):
    model.eval();result=[]
    with torch.no_grad():
        for i in range(0,len(c),1024):
            values=model(c[i:i+1024],n[i:i+1024])
            if log_mean is not None:values=(values*log_std+log_mean).exp()
            result.append(values.mean(1).numpy())
    return np.concatenate(result)


def fit(f,v,spec,epochs=None):
    torch.manual_seed(spec.get('seed',42))
    if spec.get('property_features'):
        from analysis.experiments.property_features import augment
        f=augment(f);v=augment(v)
    if spec.get('distribution_dir'):
        origin='inner' if v.RENT_APPROVAL_DATE.min()<'2024-04' else 'outer'
        extra=pd.read_csv(Path(spec['distribution_dir'])/(origin+'_features.csv'),index_col='source_row')
        f=f.join(extra);v=v.join(extra)
    if spec.get('adjusted_history'):
        from analysis.experiments.adjusted_rent_history import adjusted_history
        af,av=adjusted_history(f,v);f=f.join(af);v=v.join(av)
    (c,n,cv,nv),meta=prepare(f,v,spec.get('history',True),spec.get('quantile',True),spec.get('raw_time',False))
    factor_fit=np.ones(len(f));factor_val=np.ones(len(v))
    if spec.get('target_unit')=='per_sqm':
        factor_fit=f.FLOOR_AREA_SQM.to_numpy(float);factor_val=v.FLOOR_AREA_SQM.to_numpy(float)
    if spec.get('target_unit')=='market_ratio':
        from analysis.experiments.adjusted_rent_history import adjusted_history
        af,av=adjusted_history(f,v)
        factor_fit=af.ADJ_PRIOR_MARKET.to_numpy(float);factor_val=av.ADJ_PRIOR_MARKET.to_numpy(float)
    transformed=f.MONTHLY_RENT.to_numpy(float)/factor_fit
    if spec.get('target_unit')=='log':transformed=np.log(f.MONTHLY_RENT.to_numpy(float))
    mu=float(transformed.mean());sd=float(transformed.std(ddof=0))
    y=torch.tensor((transformed-mu)/sd,dtype=torch.float32)
    bins=None
    if spec['encoding'] not in ('none','periodic'):
        if spec['encoding']=='supervised':
            bins=ne.compute_bins(n,n_bins=spec.get('n_bins',16),tree_kwargs={'min_samples_leaf':100,'random_state':42},y=y,regression=True)
        else:bins=ne.compute_bins(n,n_bins=spec.get('n_bins',16))
    model=Model([len(meta['vocab'][c])+1 for c in meta['categories']],n.shape[1],bins,spec)
    if spec.get('mixture'):
        meta['mixture_cutoffs']=np.quantile(f.MONTHLY_RENT,[.1,.9])
        bands=torch.tensor(np.digitize(f.MONTHLY_RENT,meta['mixture_cutoffs']),dtype=torch.long)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec.get('weight_decay',.001))
    best=float('inf');selected=1;stale=0;trace=[]
    for epoch in range(1,(epochs or spec.get('max_epochs',60))+1):
        model.train()
        for ix in torch.randperm(len(y)).split(512):
            optimizer.zero_grad();pred=model(c[ix],n[ix])
            # Preserve rent-SGD squared-error objective after target conversion.
            if spec.get('target_unit')=='log' and not spec.get('log_mse',False):
                actual=torch.tensor(f.MONTHLY_RENT.to_numpy('float32')[ix.numpy()])
                loss=(((pred*sd+mu).exp()-actual[:,None])/f.MONTHLY_RENT.std(ddof=0)).square().mean()
            else:
                loss=((pred-y[ix,None]).square()*torch.tensor((factor_fit[ix.numpy()]/factor_fit.mean())**2,dtype=torch.float32)[:,None]).mean()
            if spec.get('mixture'):
                labels=bands[ix,None].expand(-1,spec['k'])
                classification=nn.functional.cross_entropy(model.gate_logits.reshape(-1,3),labels.reshape(-1))
                selected_expert=model.expert_predictions.gather(2,labels[...,None]).squeeze(-1)
                expert_loss=(selected_expert-y[ix,None]).square().mean()
                loss=loss+spec.get('aux_weight',.1)*classification+.1*expert_loss
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epochs is None:
            pred=predict(model,cv,nv,mu,sd) if spec.get('target_unit')=='log' else (predict(model,cv,nv)*sd+mu)*factor_val
            score=metrics(v.MONTHLY_RENT.to_numpy(),pred)['rmse_sgd']
            trace.append({'epoch':epoch,'rmse_sgd':score})
            if epoch==1 or epoch%5==0:
                print('Epoch',epoch,'inner RMSE',round(score,3),flush=True)
            if score<best-.1:best=score;selected=epoch;stale=0
            else:stale+=1
            if stale>=8:break
    meta.update(target_mean=mu,target_std=sd,spec=spec,bins=bins)
    predictions=predict(model,cv,nv,mu,sd) if spec.get('target_unit')=='log' else (predict(model,cv,nv)*sd+mu)*factor_val
    return model,meta,selected,best,trace,predictions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--repeat-selected-from',type=Path)
    p.add_argument('--round',type=int,choices=[1,2,3,4,5,6,7,8,9],default=1)
    p.add_argument('--distribution-dir',type=Path)
    p.add_argument('--inner-start',default='2023-10')
    a=p.parse_args();out=check_output_path(a.output_dir)
    if not '2021-04' <= a.inner_start < '2024-04':p.error('Inner start must precede outer origin')
    if out.exists() and any(out.iterdir()) and not a.resume:p.error('Empty directory required, or use --resume')
    if a.resume and not (out/'protocol.json').exists():p.error('Resume needs an existing protocol')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    before=sha256(DATA/'train.csv');frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for c in ['TOWN','STREET','BLOCK']:frame[c]=frame[c].astype(str).str.casefold().str.strip()
    frame['STREET_ID']=frame.TOWN+'|'+frame.STREET;frame['BUILDING_ID']=frame.STREET_ID+'|'+frame.BLOCK
    specs=[{'name':'tabm_plain','encoding':'none','k':8,'width':128,'blocks':2,'lr':.001},
           {'name':'tabm_piecewise','encoding':'quantile','k':8,'width':128,'blocks':2,'lr':.001},
           {'name':'tabm_supervised_bins','encoding':'supervised','k':8,'width':128,'blocks':2,'lr':.001},
           {'name':'tabm_piecewise_16','encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001}]
    if a.round==2:
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001}
        specs=[dict(common,name='piecewise32_wide',n_bins=32,width=192,blocks=3),
               dict(common,name='piecewise32_members32',n_bins=32,width=192,k=32),
               dict(common,name='adjusted_only',adjusted_history=True,history=False),
               dict(common,name='adjusted_plus_raw',adjusted_history=True),
               dict(common,name='standard_piecewise',quantile=False)]
    if a.round==3:
        if a.distribution_dir is None:p.error('Round 3 requires --distribution-dir')
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001,'distribution_dir':str(a.distribution_dir)}
        specs=[dict(common,name='distribution_raw'),
               dict(common,name='distribution_adjusted',adjusted_history=True,history=False),
               dict(common,name='distribution_slow',lr=.0005),
               dict(common,name='distribution_wide',width=192,blocks=3,n_bins=32)]
    if a.round==4:
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001}
        specs=[dict(common,name='per_sqm',target_unit='per_sqm'),
               dict(common,name='market_ratio',target_unit='market_ratio'),
               dict(common,name='per_sqm_slow',target_unit='per_sqm',lr=.0005),
               dict(common,name='market_ratio_adjusted',target_unit='market_ratio',adjusted_history=True,history=False)]
    if a.round==5:
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001,'mixture':True}
        specs=[dict(common,name='three_experts_01',aux_weight=.1),
               dict(common,name='three_experts_05',aux_weight=.5),
               dict(common,name='three_experts_slow',aux_weight=.1,lr=.0005)]
    if a.round==6:
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001,'raw_time':True}
        specs=[dict(common,name='raw_time_piecewise'),
               dict(common,name='raw_time_adjusted',target_unit='market_ratio',adjusted_history=True,history=False),
               dict(common,name='raw_time_per_sqm',target_unit='per_sqm',lr=.0005)]
    if a.round==7:
        common={'encoding':'periodic','k':16,'width':128,'blocks':2,'lr':.001}
        specs=[dict(common,name='periodic_01',frequency_scale=.1),
               dict(common,name='periodic_1',frequency_scale=1.),
               dict(common,name='periodic_lite',frequency_scale=.1,lite=True),
               dict(common,name='periodic_per_sqm',frequency_scale=.1,target_unit='per_sqm',lr=.0005)]
    if a.round==8:
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001,'target_unit':'log'}
        specs=[dict(common,name='log_rent_sgd_loss'),
               dict(common,name='log_rent_slow',lr=.0005),
               dict(common,name='log_rent_mse',log_mse=True),
               dict(common,name='log_rent_adjusted',adjusted_history=True,history=False)]
    if a.round==9:
        common={'encoding':'quantile','k':16,'width':128,'blocks':2,'lr':.001,'property_features':True}
        specs=[dict(common,name='property_piecewise'),
               dict(common,name='property_per_sqm',target_unit='per_sqm',lr=.0005),
               dict(common,name='property_adjusted',adjusted_history=True,history=False),
               dict(common,name='property_slow',lr=.0005)]
    if a.repeat_selected_from:
        previous=json.loads((a.repeat_selected_from/'protocol.json').read_text())
        chosen=json.loads((a.repeat_selected_from/'selected.json').read_text())
        reference=next(s for s in previous['specifications'] if s['name']==chosen['model'])
        specs=[dict(reference,name=f'seed_{seed}',seed=seed) for seed in [42,43,44]]
    if a.resume:
        saved=json.loads((out/'protocol.json').read_text())
        if saved['specifications']!=specs or saved.get('inner_tune_start','2023-10')!=a.inner_start:
            p.error('Resume specifications/time origin differ from the saved run')
    (out/'protocol.json').write_text(json.dumps({'specifications':specs,'seed':42,'fit':'2021-01..2024-03',
          'inner_fit_end_exclusive':a.inner_start,'inner_tune_start':a.inner_start,'inner_tune_end_exclusive':'2024-04','outer':'2024-04..2025-03',
          'scenario':'Fixed-origin historical features; no validation target history updates',
          'implementation':'Official TabM backbone plus learned category embeddings; optional external numerical embeddings.',
          'numerical_embedding':{'version':'B','activation':False},
          'versions':{'tabm':__import__('importlib.metadata',fromlist=['version']).version('tabm'),
                      'rtdl_num_embeddings':__import__('importlib.metadata',fromlist=['version']).version('rtdl_num_embeddings')},
          'caveat':'Previously inspected outer; exploration. No guarantee of RMSE 450.'},indent=2))
    inner=frame[frame.RENT_APPROVAL_DATE<a.inner_start];tune=frame[(frame.RENT_APPROVAL_DATE>=a.inner_start)&(frame.RENT_APPROVAL_DATE<'2024-04')]
    selections=pd.read_csv(out/'selection_metrics.csv').to_dict('records') if a.resume and (out/'selection_metrics.csv').exists() else []
    for spec in specs:
        if any(row['model']==spec['name'] for row in selections):continue
        print('Inner fit',spec['name'],flush=True)
        _,_,epoch,score,trace,_=fit(inner,tune,spec)
        selections.append({'model':spec['name'],'epochs':epoch,'inner_rmse':score})
        pd.DataFrame(selections).to_csv(out/'selection_metrics.csv',index=False)
        pd.DataFrame(trace).to_csv(out/(spec['name']+'_trace.csv'),index=False)
        print(selections[-1],flush=True)
    winner=min(selections,key=lambda r:r['inner_rmse']);spec=next(s for s in specs if s['name']==winner['model'])
    (out/'selected.json').write_text(json.dumps(winner,indent=2))
    f=frame[frame.RENT_APPROVAL_DATE<'2024-04'];v=frame[frame.RENT_APPROVAL_DATE>='2024-04']
    if a.repeat_selected_from:
        predictions=pd.DataFrame({'source_row':v.index,'month':v.RENT_APPROVAL_DATE,'actual':v.MONTHLY_RENT})
        parts=[]
        for config,selection in zip(specs,selections):
            print('Refit seed',selection,flush=True)
            model,meta,_,_,_,pred=fit(f,v,config,epochs=selection['epochs'])
            predictions[config['name']]=pred;parts.append(pred)
            torch.save(model.state_dict(),out/(config['name']+'.pt'))
            joblib.dump(meta,out/(config['name']+'_preprocessor.joblib'))
        predictions['prediction']=np.mean(parts,axis=0)
        predictions.to_csv(out/'predictions.csv',index=False)
        row={'model':'equal_three_seed_tabm','n_fit':len(f),'n_validation':len(v),
             **metrics(v.MONTHLY_RENT.to_numpy(),predictions.prediction.to_numpy())}
        pd.DataFrame([row]).to_csv(out/'outer_metrics.csv',index=False)
        assert sha256(DATA/'train.csv')==before
        print(row,flush=True)
        return
    print('Refit',winner,flush=True)
    model,meta,_,_,_,pred=fit(f,v,spec,epochs=winner['epochs'])
    row={**winner,'n_fit':len(f),'n_validation':len(v),**metrics(v.MONTHLY_RENT.to_numpy(),pred)}
    pd.DataFrame([row]).to_csv(out/'outer_metrics.csv',index=False)
    pd.DataFrame({'source_row':v.index,'month':v.RENT_APPROVAL_DATE,'actual':v.MONTHLY_RENT,'prediction':pred}).to_csv(out/'predictions.csv',index=False)
    torch.save(model.state_dict(),out/'model.pt');joblib.dump(meta,out/'preprocessor.joblib')
    assert sha256(DATA/'train.csv')==before
    print(row,flush=True)


if __name__=='__main__':main()

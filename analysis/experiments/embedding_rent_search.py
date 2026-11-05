"""Inner-selected embedding MLP experiments toward RMSE 450; no outer tuning."""
import argparse,json,copy
from pathlib import Path
import numpy as np,pandas as pd,joblib,torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from analysis.pipeline.config import DATA
from analysis.pipeline.data import check_output_path,sha256
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.evaluation import metrics
from analysis.experiments.auxiliary_ablation import BASE,BLOCK,AMENITIES,CATEGORICAL
from analysis.experiments.spatiotemporal_hybrid import add_month,historical_features,HISTORY_FEATURES
from analysis.experiments.deep_learning import build_network,ExperimentConfig


def prepare(fit,val,history,quantile,raw_time=False):
    f,v=fit.copy(),val.copy()
    if history:
        hf,hv=historical_features(add_month(f),add_month(v));f=f.join(hf);v=v.join(hv)
    features=BASE+BLOCK+AMENITIES+(HISTORY_FEATURES if history else [])
    features += [col for col in f.columns if col.startswith(('COMP_','ADJ_','DIST_'))]
    features += [col for col in f.columns if col.startswith(('XNUM_','XCAT_'))]
    cats=[c for c in features if c in CATEGORICAL or c.startswith('XCAT_')]+['STREET_ID','BUILDING_ID']
    nums=[c for c in features if c not in cats]
    fc=[];vc=[];vocab={}
    for c in cats:
        counts=f[c].value_counts();vocab[c]=list(counts[counts>=4].index)
        idx=pd.Index(vocab[c]);fc.append(idx.get_indexer(f[c])+1);vc.append(idx.get_indexer(v[c])+1)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)
    xn=imputer.fit_transform(f[nums]);xv=imputer.transform(v[nums])
    scaler=QuantileTransformer(n_quantiles=256,output_distribution='normal',random_state=42) if quantile else StandardScaler()
    time_scaler=None
    raw_train=raw_val=None
    if raw_time:
        column=nums.index('APPROVAL_MONTH_INDEX')
        time_scaler=StandardScaler().fit(xn[:,[column]])
        raw_train=time_scaler.transform(xn[:,[column]])
        raw_val=time_scaler.transform(xv[:,[column]])
    xn=scaler.fit_transform(xn).astype('float32');xv=scaler.transform(xv).astype('float32')
    if raw_time:
        xn=np.column_stack([xn,raw_train]).astype('float32');xv=np.column_stack([xv,raw_val]).astype('float32')
    arrays=[np.stack(fc,axis=1),xn,np.stack(vc,axis=1),xv]
    meta={'categories':cats,'numeric':nums,'vocab':vocab,'imputer':imputer,'scaler':scaler,'raw_time_scaler':time_scaler}
    return [torch.tensor(x) for x in arrays],meta


def predict(net,c,n):
    net.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(c),4096):
            output=net(c[i:i+4096],n[i:i+4096])
            parts.append((output[:,0] if output.ndim==2 else output).numpy())
    return np.concatenate(parts)


def fit_model(f,v,spec,epochs=None):
    torch.manual_seed(spec.get('seed',42))
    if spec.get('adjusted_history'):
        from analysis.experiments.adjusted_rent_history import adjusted_history
        af,av=adjusted_history(f,v);f=f.join(af);v=v.join(av)
    if spec.get('comparables_dir'):
        origin = 'inner' if v.RENT_APPROVAL_DATE.min() < '2024-04' else 'outer'
        additional = pd.read_csv(Path(spec['comparables_dir'])/(origin+'_features.csv'),index_col='source_row')
        f=f.join(additional);v=v.join(additional)
    (c,n,cv,nv),meta=prepare(f,v,spec['history'],spec['quantile'])
    reference_fit=np.zeros(len(f));reference_val=np.zeros(len(v))
    if spec.get('residual_reference'):
        from analysis.experiments.adjusted_rent_history import adjusted_reference
        reference_fit=adjusted_reference(f);reference_val=adjusted_reference(v)
    target=f.MONTHLY_RENT.to_numpy('float32')-reference_fit
    mu=float(target.mean());sd=float(target.std())
    y=torch.tensor((target-mu)/sd,dtype=torch.float32)
    config=ExperimentConfig(architecture=spec['architecture'],width=spec.get('width',128),blocks=spec['blocks'],dropout=spec.get('dropout',.1))
    net=build_network([len(meta['vocab'][x])+1 for x in meta['categories']],n.shape[1],config)
    if spec.get('aux_weight'):
        # One regression output and three price-band logits, sharing the same MLP.
        net.output[-1]=nn.Linear(net.output[-1].in_features,4)
        boundaries=np.quantile(f.MONTHLY_RENT,[.1,.9])
        bands=torch.tensor(np.digitize(f.MONTHLY_RENT,boundaries),dtype=torch.long)
        meta['band_boundaries']=boundaries
    optimizer=torch.optim.AdamW(net.parameters(),lr=spec['lr'],weight_decay=spec.get('weight_decay',.01))
    weights = torch.ones(len(y))
    if spec.get('half_life'):
        months = pd.PeriodIndex(f.RENT_APPROVAL_DATE, freq='M').asi8
        weights = torch.tensor(2.**(-(months.max()-months)/spec['half_life']),dtype=torch.float32)
        weights /= weights.mean()
    best=float('inf');selected=1;stale=0;trace=[]
    for epoch in range(1,(epochs or spec.get('max_epochs',60))+1):
        net.train();order=torch.randperm(len(y))
        for ix in order.split(1024):
            optimizer.zero_grad();p=net(c[ix],n[ix])
            regression=p[:,0] if p.ndim==2 else p
            loss=((regression-y[ix]).square()*weights[ix]).mean()
            if spec.get('aux_weight'):
                loss=loss+spec['aux_weight']*nn.functional.cross_entropy(p[:,1:],bands[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if epochs is None:
            pred=predict(net,cv,nv)*sd+mu+reference_val;score=metrics(v.MONTHLY_RENT.to_numpy(),pred)['rmse_sgd']
            trace.append({'epoch':epoch,'rmse_sgd':score})
            if score<best-.1:best=score;selected=epoch;stale=0
            else:stale+=1
            if stale>=8:break
    meta.update(target_mean=mu,target_std=sd,config=spec)
    return net,meta,selected,best,trace,predict(net,cv,nv)*sd+mu+reference_val, predict(net,c,n)*sd+mu+reference_fit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--round',type=int,choices=[1,2,3,4,5,6,7],default=1)
    p.add_argument('--comparables-dir',type=Path)
    p.add_argument('--repeat-selected-from',type=Path)
    a=p.parse_args();out=check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()):p.error('Empty output directory required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    original=sha256(DATA/'train.csv');frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for c in ['TOWN','STREET','BLOCK']:frame[c]=frame[c].astype(str).str.casefold().str.strip()
    frame['STREET_ID']=frame.TOWN+'|'+frame.STREET
    frame['BUILDING_ID']=frame.STREET_ID+'|'+frame.BLOCK
    # Keep same outer rows; compare more data only as an explicitly recorded factor.
    specs=[{'name':'embedding_mlp','history':False,'quantile':False,'architecture':'mlp','blocks':2,'lr':.001,'start':'2022-01'},
           {'name':'embedding_history','history':True,'quantile':False,'architecture':'mlp','blocks':2,'lr':.001,'start':'2022-01'},
           {'name':'quantile_history','history':True,'quantile':True,'architecture':'mlp','blocks':2,'lr':.001,'start':'2022-01'},
           {'name':'resnet_history','history':True,'quantile':True,'architecture':'resnet','blocks':2,'lr':.0005,'start':'2022-01'},
           {'name':'full_history_mlp','history':True,'quantile':True,'architecture':'mlp','blocks':2,'lr':.0005,'start':'2021-01'}]
    if a.round==2:
        common={'history':True,'architecture':'mlp','blocks':3,'start':'2021-01','width':256,'dropout':.15,'weight_decay':.001}
        specs=[dict(common,name='wide_history_standard',quantile=False,lr=.0005),
               dict(common,name='wide_history_quantile',quantile=True,lr=.0005),
               dict(common,name='recency12_quantile',quantile=True,lr=.0005,half_life=12),
               dict(common,name='recency6_standard',quantile=False,lr=.0003,half_life=6),
               dict(common,name='recent2023',quantile=False,lr=.0005,start='2023-01'),
               dict(common,name='recent2022',quantile=False,lr=.0005,start='2022-01')]
    if a.round==3:
        common={'history':True,'quantile':True,'architecture':'mlp','blocks':2,'start':'2021-01','width':128,'lr':.0005}
        specs=[dict(common,name='aux_band_002',aux_weight=.02),
               dict(common,name='aux_band_01',aux_weight=.1),
               dict(common,name='aux_band_05',aux_weight=.5),
               dict(common,name='aux_resnet_01',aux_weight=.1,architecture='resnet')]
    if a.round==4:
        common={'history':True,'quantile':True,'architecture':'resnet','blocks':2,
                'start':'2021-01','width':128,'lr':.0005,'aux_weight':.1}
        specs=[dict(common,name='aux_resnet_regularized',dropout=.2,weight_decay=.02),
               dict(common,name='aux_resnet_slow',lr=.0002,max_epochs=100),
               dict(common,name='aux_resnet_wide',width=192,blocks=3,weight_decay=.005),
               dict(common,name='aux_resnet_recency',half_life=18),
               dict(common,name='aux_resnet_strong_task',aux_weight=.5)]
    if a.round==5:
        if a.comparables_dir is None:p.error('--round 5 requires --comparables-dir')
        common={'history':True,'quantile':True,'architecture':'resnet','blocks':2,'start':'2021-01',
                'width':128,'lr':.0005,'aux_weight':.1,'comparables_dir':str(a.comparables_dir)}
        specs=[dict(common,name='comparable_resnet'),
               dict(common,name='comparable_mlp',architecture='mlp'),
               dict(common,name='comparable_wide',width=192,blocks=3,weight_decay=.005),
               dict(common,name='comparable_recent',half_life=18)]
    if a.round==6:
        common={'history':False,'adjusted_history':True,'quantile':True,'architecture':'resnet','blocks':2,
                'start':'2021-01','width':128,'lr':.0005,'aux_weight':.1}
        specs=[dict(common,name='adjusted_resnet'),
               dict(common,name='adjusted_with_raw',history=True),
               dict(common,name='adjusted_mlp',architecture='mlp'),
               dict(common,name='adjusted_wide',width=192,blocks=3,weight_decay=.005)]
    if a.round==7:
        common={'history':False,'adjusted_history':True,'residual_reference':True,'quantile':True,
                'architecture':'resnet','blocks':2,'start':'2021-01','width':128,'lr':.0005,'aux_weight':.1}
        specs=[dict(common,name='adjusted_residual'),
               dict(common,name='adjusted_residual_mlp',architecture='mlp'),
               dict(common,name='adjusted_residual_wide',width=192,blocks=3,weight_decay=.005)]
    if a.repeat_selected_from:
        previous=json.loads((a.repeat_selected_from/'protocol.json').read_text())
        chosen=json.loads((a.repeat_selected_from/'selected.json').read_text())
        reference=next(s for s in previous['specifications'] if s['name']==chosen['model'])
        specs=[dict(reference,name=f"seed_{seed}",seed=seed) for seed in [42,43,44]]
    (out/'protocol.json').write_text(json.dumps({'specifications':specs,'inner_fit_end':'2023-09','inner_validation':'2023-10..2024-03',
        'outer_validation':'2024-04..2025-03','seed':42,'target':450,'selection':'inner RMSE only',
        'caveat':'Previously inspected outer; single-seed exploratory study. Histories frozen at origin; actual outer rents not features.'},indent=2))
    tuning=[]
    for spec in specs:
        ff=frame[(frame.RENT_APPROVAL_DATE>=spec['start'])&(frame.RENT_APPROVAL_DATE<'2023-10')]
        vv=frame[(frame.RENT_APPROVAL_DATE>='2023-10')&(frame.RENT_APPROVAL_DATE<'2024-04')]
        print('Inner fitting',spec['name'],flush=True)
        _,_,selected,score,trace,_,_=fit_model(ff,vv,spec)
        tuning.append({'model':spec['name'],'epochs':selected,'inner_rmse':score})
        pd.DataFrame(trace).to_csv(out/(spec['name']+'_epochs.csv'),index=False)
        pd.DataFrame(tuning).to_csv(out/'selection_metrics.csv',index=False)
        print(tuning[-1],flush=True)
    # The inner winner is fixed before accessing outer predictions.
    winner=min(tuning,key=lambda x:x['inner_rmse'])
    if a.repeat_selected_from:
        # Fixed equal weights; no outer-based seed or weight selection.
        v=frame[frame.RENT_APPROVAL_DATE>='2024-04']
        exported=pd.DataFrame({'source_row':v.index,'month':v.RENT_APPROVAL_DATE,'actual':v.MONTHLY_RENT})
        parts=[]
        for spec,selection in zip(specs,tuning):
            f=frame[(frame.RENT_APPROVAL_DATE>=spec['start'])&(frame.RENT_APPROVAL_DATE<'2024-04')]
            net,meta,_,_,_,pred,_=fit_model(f,v,spec,epochs=selection['epochs'])
            exported[spec['name']]=pred;parts.append(pred)
            torch.save(net.state_dict(),out/(spec['name']+'.pt'))
            joblib.dump(meta,out/(spec['name']+'_preprocessor.joblib'))
        exported['prediction']=np.mean(parts,axis=0)
        exported.to_csv(out/'predictions.csv',index=False)
        score={'model':'equal_weight_three_seed_ensemble','n_fit':len(f),'n_validation':len(v),
               **metrics(v.MONTHLY_RENT.to_numpy(),exported.prediction.to_numpy())}
        pd.DataFrame([score]).to_csv(out/'outer_metrics.csv',index=False)
        assert sha256(DATA/'train.csv')==original
        print(score,flush=True)
        return
    (out/'selected.json').write_text(json.dumps(winner,indent=2))
    selected_spec=next(s for s in specs if s['name']==winner['model'])
    f=frame[(frame.RENT_APPROVAL_DATE>=selected_spec['start'])&(frame.RENT_APPROVAL_DATE<'2024-04')]
    v=frame[frame.RENT_APPROVAL_DATE>='2024-04']
    print('Refit inner winner',winner,flush=True)
    net,meta,_,_,_,pred,tp=fit_model(f,v,selected_spec,epochs=winner['epochs'])
    row={**winner,'n_fit':len(f),'n_validation':len(v),'train_rmse':metrics(f.MONTHLY_RENT.to_numpy(),tp)['rmse_sgd'],**metrics(v.MONTHLY_RENT.to_numpy(),pred)}
    pd.DataFrame([row]).to_csv(out/'outer_metrics.csv',index=False)
    pd.DataFrame({'source_row':v.index,'month':v.RENT_APPROVAL_DATE,'actual':v.MONTHLY_RENT,'prediction':pred}).to_csv(out/'predictions.csv',index=False)
    torch.save(net.state_dict(),out/'model.pt');joblib.dump(meta,out/'preprocessor.joblib')
    assert sha256(DATA/'train.csv')==original
    print(row,flush=True)


if __name__=='__main__':main()

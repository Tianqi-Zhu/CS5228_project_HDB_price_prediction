"""MLP historical-rent and residual-target ablation with nested temporal selection."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analysis.pipeline.config import DATA
from analysis.pipeline.data import check_output_path,sha256
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.evaluation import metrics
from analysis.experiments.market_latent_mlp import setup
from analysis.experiments.spatiotemporal_hybrid import add_month,historical_features,HISTORY_FEATURES
from analysis.experiments.auxiliary_ablation import BASE,BLOCK,AMENITIES,CATEGORICAL


def history_frames(fit,val):
    a,b=historical_features(add_month(fit),add_month(val))
    f=fit.join(a);v=val.join(b)
    for frame in [f,v]:
        for c in HISTORY_FEATURES:
            if c.endswith('_COUNT') or c.endswith('_RECENCY_MONTHS'):
                frame[c]=np.log1p(frame[c].clip(lower=0))
    return f,v


def reference(frame,fallback):
    # Only strictly-prior training histories or frozen-origin validation histories.
    return frame.HIST_SIGNATURE_LAST_MEAN.fillna(frame.HIST_BLOCK_FLAT_LAST_MEAN).fillna(frame.HIST_AREA_FLAT_LAST_MEAN).fillna(fallback).to_numpy(float)


def initialize(fit,val,layers,residual,lr,addresses=False):
    fit,val=history_frames(fit,val)
    fallback=float(fit.MONTHLY_RENT.mean())
    bf=reference(fit,fallback) if residual else np.zeros(len(fit))
    bv=reference(val,fallback) if residual else np.zeros(len(val))
    features=BASE+BLOCK+AMENITIES+HISTORY_FEATURES
    if addresses:
        features = features + ['STREET_ID', 'BUILDING_ID']
    learn=fit.copy();learn.MONTHLY_RENT=fit.MONTHLY_RENT-bf
    prep,net,x,y,mu,sd=setup(learn,features,layers,42)
    net.set_params(learning_rate_init=lr)
    xx=prep.transform(val[features]).astype(np.float32)
    return fit,val,prep,net,x,y,xx,mu,sd,bf,bv,features


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=__import__('pathlib').Path,required=True)
    p.add_argument('--addresses',action='store_true')
    args=p.parse_args();out=check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()):p.error('Empty output directory required')
    out.mkdir(parents=True,exist_ok=True)
    before=sha256(DATA/'train.csv')
    frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    if args.addresses:
        frame['STREET_ID']=frame.STREET.astype(str).str.casefold().str.strip()
        frame['BUILDING_ID']=frame.TOWN.astype(str).str.casefold().str.strip()+'|'+frame.BLOCK.astype(str).str.casefold().str.strip()+'|'+frame.STREET_ID
        CATEGORICAL.update(['STREET_ID','BUILDING_ID'])
    frame=frame[frame.RENT_APPROVAL_DATE>='2022-01']
    inner=frame[frame.RENT_APPROVAL_DATE<'2023-10'];tune=frame[(frame.RENT_APPROVAL_DATE>='2023-10')&(frame.RENT_APPROVAL_DATE<'2024-04')]
    fit=frame[frame.RENT_APPROVAL_DATE<'2024-04'];val=frame[frame.RENT_APPROVAL_DATE>='2024-04']
    variants=[('history_2layers',(128,64),False,.001),
              ('history_8layers',(128,128,128,128,64,64,32,16),False,.001),
              ('residual_2layers',(128,64),True,.001),
              ('residual_4layers_slow',(128,128,64,32),True,.0003)]
    if args.addresses:
        variants=[('address_history_4slow',(128,128,64,32),False,.0003),
                  ('address_history_8slow',(128,128,128,128,64,64,32,16),False,.0003),
                  ('address_history_8slower',(128,128,128,128,64,64,32,16),False,.0001)]
    rows=[];predictions=val[['RENT_APPROVAL_DATE','MONTHLY_RENT']].copy();predictions['source_row']=val.index
    for name,layers,residual,lr in variants:
        print('Fitting',name,flush=True)
        f,v,prep,net,x,y,xx,mu,sd,bf,bv,features=initialize(inner,tune,layers,residual,lr,args.addresses)
        best=float('inf');selected=1;stale=0;history=[]
        for epoch in range(1,81):
            net.partial_fit(x,y)
            pred=net.predict(xx)*sd+mu+bv
            score=metrics(v.MONTHLY_RENT.to_numpy(),pred)
            history.append({'epoch':epoch,**score})
            if score['rmse_sgd']<best-.1:best=score['rmse_sgd'];selected=epoch;stale=0
            else:stale+=1
            if stale>=12:break
        pd.DataFrame(history).to_csv(out/f'{name}_epochs.csv',index=False)
        f,v,prep,net,x,y,xx,mu,sd,bf,bv,features=initialize(fit,val,layers,residual,lr,args.addresses)
        for _ in range(selected):net.partial_fit(x,y)
        pred=net.predict(xx)*sd+mu+bv
        train_pred=net.predict(x)*sd+mu+bf
        row={'model':name,'layers':str(layers),'residual':residual,'learning_rate':lr,'selected_epochs':selected,
             'inner_rmse':best,'train_rmse':metrics(f.MONTHLY_RENT.to_numpy(),train_pred)['rmse_sgd'],
             'n_fit':len(f),'n_validation':len(v),**metrics(v.MONTHLY_RENT.to_numpy(),pred)}
        print(row,flush=True);rows.append(row);predictions[name]=pred
        if residual:predictions[name+'_history_reference']=bv
        joblib.dump({'model':net,'preprocessor':prep,'features':features,'mean':mu,'std':sd,'residual':residual,
                     'history_fallback':float(f.MONTHLY_RENT.mean())},out/f'{name}.joblib')
    scores=pd.DataFrame(rows);scores.to_csv(out/'metrics.csv',index=False)
    predictions.to_csv(out/'predictions.csv',index=False)
    winner=scores.loc[scores.inner_rmse.idxmin()]
    assert sha256(DATA/'train.csv')==before
    (out/'protocol.json').write_text(json.dumps({'scenario':'Fixed-origin predictions; history frozen at each origin',
         'fit_range':['2022-01','2024-03'],'inner_tune':['2023-10','2024-03'],'validation':['2024-04','2025-03'],
         'seed':42,'addresses':args.addresses,'selection_metric':'inner RMSE only','selected_model':winner.model,'raw_sha256':before,
         'caveats':['Reused validation, exploratory.', 'No outlier removal; same rows as previous MLP.',
                    'Residual fallback mean learned from fitting targets only; first-month fitting residual reference is in-sample global mean, not validation labels.']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()

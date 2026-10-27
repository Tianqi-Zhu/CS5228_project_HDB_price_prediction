"""Depth/latent ablation with identical rental rows and inner temporal epoch selection."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from analysis.pipeline.config import DATA
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.data import check_output_path,sha256
from analysis.pipeline.evaluation import metrics
from analysis.experiments.auxiliary_ablation import BASE,BLOCK,AMENITIES,CATEGORICAL


def setup(fit,features,layers,seed):
    cats=[c for c in features if c in CATEGORICAL];nums=[c for c in features if c not in cats]
    prep=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(keep_empty_features=True)),('scale',StandardScaler())]),nums),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cats)],sparse_threshold=1.)
    x=prep.fit_transform(fit[features]).astype(np.float32)
    y=fit.MONTHLY_RENT.to_numpy(np.float32);mu=float(y.mean());sd=float(y.std())
    net=MLPRegressor(hidden_layer_sizes=layers,activation='relu',solver='adam',learning_rate_init=.001,
                     alpha=.01,batch_size=512,random_state=seed,max_iter=1,early_stopping=False)
    return prep,net,x,(y-mu)/sd,mu,sd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--latent-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-epochs',type=int,default=60)
    a=p.parse_args();out=check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()):p.error('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    before=sha256(DATA/'train.csv')
    frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    frame=frame[frame.RENT_APPROVAL_DATE>='2022-01'].copy()
    extras={'none':[]}
    for kind in ['rnn','lstm']:
        d=pd.read_csv(a.latent_dir/f'{kind}_42_latent.csv').set_index('prediction_month').drop(columns='input_end_month')
        frame=frame.join(d,on='RENT_APPROVAL_DATE');extras[kind]=list(d.columns)
    inner=frame[frame.RENT_APPROVAL_DATE<'2023-10']
    tune=frame[(frame.RENT_APPROVAL_DATE>='2023-10')&(frame.RENT_APPROVAL_DATE<'2024-04')]
    fit=frame[frame.RENT_APPROVAL_DATE<'2024-04'];val=frame[frame.RENT_APPROVAL_DATE>='2024-04']
    predictions=val[['RENT_APPROVAL_DATE','MONTHLY_RENT']].copy();predictions['source_row']=val.index
    rows=[]
    for depth,layers in [('shallow',(128,64)),('deep',(128,128,64,32))]:
        for kind,columns in extras.items():
            name=f'{depth}_{kind}';features=BASE+BLOCK+AMENITIES+columns
            print('Fitting',name,flush=True)
            prep,net,x,y,mu,sd=setup(inner,features,layers,42)
            xt=prep.transform(tune[features]).astype(np.float32)
            history=[];best=float('inf');best_epoch=1;stale=0
            for epoch in range(1,a.max_epochs+1):
                net.partial_fit(x,y)
                prediction=net.predict(xt)*sd+mu
                score=metrics(tune.MONTHLY_RENT.to_numpy(),prediction)
                history.append({'epoch':epoch,**score})
                if score['rmse_sgd']<best-.1:best=score['rmse_sgd'];best_epoch=epoch;stale=0
                else:stale+=1
                if stale>=8:break
            pd.DataFrame(history).to_csv(out/f'{name}_epochs.csv',index=False)
            prep,net,x,y,mu,sd=setup(fit,features,layers,42)
            for _ in range(best_epoch):net.partial_fit(x,y)
            prediction=net.predict(prep.transform(val[features]).astype(np.float32))*sd+mu
            training=net.predict(x)*sd+mu
            row={'model':name,'hidden_layers':str(layers),'selected_epochs':best_epoch,
                 'inner_rmse':best,'n_fit':len(fit),'n_validation':len(val),
                 'train_rmse':metrics(fit.MONTHLY_RENT.to_numpy(),training)['rmse_sgd'],
                 **metrics(val.MONTHLY_RENT.to_numpy(),prediction)}
            rows.append(row);print(row,flush=True)
            predictions[name]=prediction
            joblib.dump({'preprocessor':prep,'model':net,'target_mean':mu,'target_std':sd,'features':features},out/f'{name}.joblib')
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    predictions.to_csv(out/'predictions.csv',index=False)
    assert before==sha256(DATA/'train.csv')
    (out/'protocol.json').write_text(json.dumps({'seed':42,'features':'same 29 spatial features, optionally 3 latent coordinates',
        'training_filter':'no IQR filter, identical rows to previous tree comparison','inner_fit':'2022-01..2023-09',
        'epoch_selection':'2023-10..2024-03','outer_fit':'2022-01..2024-03','outer_validation':'2024-04..2025-03',
        'max_epochs':a.max_epochs,'patience':8,'min_improvement_sgd':.1,'raw_train_sha256':before,
        'caveat':'Previously reused outer period; exploratory one-seed depth ablation. Monthly-updated latent inputs.'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()

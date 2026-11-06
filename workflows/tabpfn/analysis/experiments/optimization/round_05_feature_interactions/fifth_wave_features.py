"""Fixed-model comparison of historical rent shape and building context."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from lightgbm import LGBMRegressor
from analysis.experiments.frontier_search import ROOT,frame_data,month_number,keys,score,write_json
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.third_wave_training import make_data

OUT=ROOT/'output/experiment_runs/round_05_feature_interactions'
ART=ROOT/'output/models/round_05_feature_interactions'
VERSION='exact-history-shape-building-v1'
STATS=('N','Q10','Q25','MEDIAN','Q75','Q90','IQR','LOW_SHARE','LOW_MEAN','HIGH_MEAN',
    'MODE_SHARE','ENTROPY','LOW_RECENCY','LAST_MONTH_MEDIAN','RECENT_MINUS_OLD_MEDIAN')


class RentShapeHistory:
    """Exact finite-sample statistics queried strictly before each cutoff."""
    def fit(self,source):
        self.tables={}
        level=source.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform('mean').to_numpy(float)
        ratio=source.MONTHLY_RENT.to_numpy(float)/level
        months=month_number(source.RENT_APPROVAL_DATE)
        for grain,fields in [('BLOCK',('ADDRESS_KEY','FLAT_TYPE_CLEAN')),
            ('STREET',('STREET_CLEAN','FLAT_TYPE_CLEAN'))]:
            frame=pd.DataFrame({'key':keys(source,fields),'month':months,'ratio':ratio})
            groups={}
            for key,part in frame.groupby('key',sort=False):
                part=part.sort_values('month',kind='stable')
                groups[key]=(part.month.to_numpy(),part.ratio.to_numpy())
            self.tables[grain]=groups
        return self

    def transform(self,query,cutoff):
        cutoff=np.asarray(cutoff)
        target_months=month_number(query.RENT_APPROVAL_DATE)
        if np.any(cutoff>target_months):
            raise ValueError('Cutoff follows target month')
        columns={}
        for grain,fields in [('BLOCK',('ADDRESS_KEY','FLAT_TYPE_CLEAN')),
            ('STREET',('STREET_CLEAN','FLAT_TYPE_CLEAN'))]:
            output={window:np.full((len(query),len(STATS)),np.nan) for window in (0,12)}
            identifiers=keys(query,fields)
            requests=pd.DataFrame({'key':identifiers,'cutoff':cutoff,'position':np.arange(len(query))})
            for key,positions in requests.groupby('key',sort=False):
                if key not in self.tables[grain]:
                    continue
                months,ratios=self.tables[grain][key]
                for cut,part in positions.groupby('cutoff',sort=False):
                    stop=np.searchsorted(months,cut,side='left')
                    if stop==0:
                        continue
                    pos=part.position.to_numpy()
                    for window in output:
                        start=0 if window==0 else np.searchsorted(months,cut-window,side='left')
                        values=ratios[start:stop]
                        dates=months[start:stop]
                        if not len(values):
                            continue
                        q10,q25,median,q75,q90=np.quantile(values,[.1,.25,.5,.75,.9])
                        low=values<.65
                        low_mean=values[low].mean() if low.any() else np.nan
                        high_mean=values[~low].mean() if (~low).any() else np.nan
                        _,counts=np.unique(np.rint(values/.1).astype(int),return_counts=True)
                        probabilities=counts/counts.sum()
                        entropy=-(probabilities*np.log(probabilities)).sum()
                        last_median=np.median(values[dates==dates.max()])
                        recent=values[dates>=cut-6];old=values[dates<cut-6]
                        shift=np.median(recent)-np.median(old) if len(recent) and len(old) else np.nan
                        output[window][pos]=[len(values),q10,q25,median,q75,q90,q75-q25,
                            low.mean(),low_mean,high_mean,probabilities.max(),entropy,
                            0.,last_median,shift]
                        output[window][pos,STATS.index('LOW_RECENCY')]=(
                            target_months[pos]-dates[low].max() if low.any() else np.nan)
            for window,values in output.items():
                for index,name in enumerate(STATS):
                    if name=='N':
                        values[:,index]=np.nan_to_num(values[:,index])
                    columns[f'SHAPE_{grain}_{window}_{name}']=values[:,index].astype(np.float32)
        return pd.DataFrame(columns,index=query.index)


class BuildingContext:
    """Unlabelled building-stock context from the supplied auxiliary table."""
    def fit(self,buildings):
        self.buildings=buildings.drop_duplicates(['TOWN','BLOCK','STREET']).copy()
        coords=self.buildings[['LATITUDE','LONGITUDE']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        selected=np.isfinite(coords).all(axis=1)
        self.buildings=self.buildings.loc[selected].reset_index(drop=True)
        self.tree=BallTree(np.deg2rad(coords[selected]),metric='haversine')
        self.height=pd.to_numeric(self.buildings.MAX_FLOOR,errors='coerce').to_numpy(float)
        self.completed=pd.to_numeric(self.buildings.YEAR_COMPLETED,errors='coerce').to_numpy(float)
        return self

    def transform(self,query):
        coords=query[['LATITUDE','LONGITUDE']].to_numpy(float)
        valid=np.isfinite(coords).all(axis=1)
        if not valid.any():
            names=[f'BUILDING_COUNT_{radius}KM' for radius in (.3,.8,1.5)]+[
                'NEIGHBOR_BUILDING_DISTANCE','NEIGHBOR_HEIGHT_MEAN','NEIGHBOR_HEIGHT_SD',
                'NEIGHBOR_TALL_SHARE','NEIGHBOR_COMPLETION_MEDIAN','NEIGHBOR_COMPLETION_SD',
                'NEIGHBOR_OLD_SHARE','HEIGHT_VS_NEIGHBORS','YEAR_VS_NEIGHBORS']
            return pd.DataFrame(np.nan,index=query.index,columns=names,dtype=np.float32)
        points=np.deg2rad(coords[valid])
        k=min(17,len(self.buildings))
        distance,neighbors=self.tree.query(points,k=k)
        # Each model record is tied to a building in this auxiliary stock;
        # remove the nearest entry (normally itself) from neighbour summaries.
        distance,neighbors=distance[:,1:]*6371.,neighbors[:,1:]
        columns={}
        for radius in (.3,.8,1.5):
            count=self.tree.query_radius(points,r=radius/6371.,count_only=True)
            values=np.full(len(query),np.nan)
            values[valid]=np.maximum(0,count-1)
            columns[f'BUILDING_COUNT_{radius}KM']=values
        if neighbors.shape[1]:
            height=self.height[neighbors]
            completed=self.completed[neighbors]
            values={'NEIGHBOR_BUILDING_DISTANCE':distance.mean(axis=1),
                'NEIGHBOR_HEIGHT_MEAN':np.nanmean(height,axis=1),
                'NEIGHBOR_HEIGHT_SD':np.nanstd(height,axis=1),
                'NEIGHBOR_TALL_SHARE':np.mean(height>=20,axis=1),
                'NEIGHBOR_COMPLETION_MEDIAN':np.nanmedian(completed,axis=1),
                'NEIGHBOR_COMPLETION_SD':np.nanstd(completed,axis=1),
                'NEIGHBOR_OLD_SHARE':np.mean(completed<1990,axis=1),
                'HEIGHT_VS_NEIGHBORS':query.MAX_FLOOR.to_numpy(float)[valid]-np.nanmean(height,axis=1),
                'YEAR_VS_NEIGHBORS':query.YEAR_COMPLETED.to_numpy(float)[valid]-np.nanmedian(completed,axis=1)}
            for name,array in values.items():
                destination=np.full(len(query),np.nan);destination[valid]=array
                columns[name]=destination
        return pd.DataFrame(columns,index=query.index,dtype=np.float32)


def feature_data(stage):
    bundle=make_data(stage)
    cache=ART/'features'/f'{stage}.joblib'
    if cache.exists():
        saved=joblib.load(cache)
        if saved['version']==VERSION and saved['hashes']==bundle['hashes']:
            bundle.update(saved['features']);return bundle
    raw,_=frame_data()
    source=raw.loc[raw.RENT_APPROVAL_DATE<bundle['start']]
    query=raw.loc[(raw.RENT_APPROVAL_DATE>=bundle['start'])&(raw.RENT_APPROVAL_DATE<bundle['end'])]
    history=RentShapeHistory().fit(source)
    print('EXACT RENT SHAPE',stage,'training',flush=True)
    fit_shape=history.transform(source,month_number(source.RENT_APPROVAL_DATE))
    print('EXACT RENT SHAPE',stage,'query',flush=True)
    query_shape=history.transform(query,np.full(len(query),bundle['origin']))
    buildings=pd.read_csv(ROOT/'data/auxiliary/sg-hdb-block.csv')
    context=BuildingContext().fit(buildings)
    features={'shape_fit':fit_shape,'shape_val':query_shape,
        'building_fit':context.transform(source),'building_val':context.transform(query)}
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'version':VERSION,'hashes':bundle['hashes'],'features':features},cache,compress=1)
    bundle.update(features)
    return bundle


def inputs(bundle,kind):
    x,q=select_inputs(bundle,'window')
    if kind in ('no36','only24'):
        remove=[c for c in x if c.startswith('ANN_BLOCK_36_') or
            (kind=='only24' and c.startswith('ANN_BLOCK_12_'))]
        x,q=x.drop(columns=remove),q.drop(columns=remove)
    if kind in ('shape','combined'):
        x=pd.concat([x,bundle['shape_fit']],axis=1)
        q=pd.concat([q,bundle['shape_val']],axis=1)
    if kind in ('building','combined'):
        x=pd.concat([x,bundle['building_fit']],axis=1)
        q=pd.concat([q,bundle['building_val']],axis=1)
    return x,q


FAMILIES=('lgb','cat','xgb')
KINDS=('base','shape','building','combined','no36','only24')


def fit(bundle,family,kind):
    x,q=inputs(bundle,kind)
    if family=='lgb':
        model=LGBMRegressor(n_estimators=353,learning_rate=.03,num_leaves=15,
            min_child_samples=40,reg_lambda=10.,random_state=42,n_jobs=4,verbosity=-1)
        model.fit(x,bundle['y_fit'])
    elif family=='cat':
        from catboost import CatBoostRegressor
        categories=list(x.select_dtypes(include=['category','object','string']).columns)
        for c in categories:
            x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        x,q=x.fillna(-999.),q.fillna(-999.)
        model=CatBoostRegressor(iterations=535,learning_rate=.04,depth=6,
            l2_leaf_reg=10.,random_strength=.4,random_seed=42,task_type='GPU',devices='0',
            gpu_ram_part=.55,border_count=128,allow_writing_files=False,verbose=False)
        model.fit(x,bundle['y_fit'],cat_features=categories)
    else:
        sys.path.insert(0,str(ROOT/'.cache/model-deps'))
        from xgboost import XGBRegressor
        model=XGBRegressor(n_estimators=1046,max_depth=3,learning_rate=.03,
            min_child_weight=20,reg_lambda=20.,reg_alpha=.5,subsample=.85,
            colsample_bytree=.9,random_state=42,n_jobs=4,tree_method='hist',
            device='cuda',enable_categorical=True)
        model.fit(x,bundle['y_fit'])
    return model,model.predict(q),x.shape[1]


def run(stage,families=None,kinds=None):
    write_json(OUT/'plan.json',{'models':{'lgb':'15 leaves, 353 rounds, lr=.03, L2=10',
        'cat':'GPU depth6, 535 rounds, lr=.04, L2=10',
        'xgb':'GPU depth3, 1046 rounds, lr=.03, L2=20'},'feature_groups':list(KINDS),
        'selection_months':['2023-10','2024-03'],
        'notes':'Parameters and round counts fixed from earlier experiments. No validation early stopping; identical rows. Exact quantiles and past low-rent persistence; unlabelled auxiliary building-stock context. No public rental targets or query rental labels.'})
    bundle=feature_data(stage)
    for family in families or FAMILIES:
        for kind in kinds or KINDS:
            name=f'{family}_{kind}'
            path=OUT/'matrix'/stage/name
            if (path/'results.json').exists():
                continue
            started=time.perf_counter()
            print('FEATURE MATRIX',stage,name,flush=True)
            model,pred,n_features=fit(bundle,family,kind)
            pred=np.asarray(pred,dtype=float)
            artifact=ART/'matrix'/stage/name
            artifact.mkdir(parents=True,exist_ok=True)
            joblib.dump({'model':model,'family':family,'kind':kind},artifact/'model.joblib',compress=1)
            pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
                'actual':bundle['y_val'],'prediction':pred}).to_csv(artifact/'predictions.csv',index=False)
            record={'name':name,'family':family,'kind':kind,'stage':stage,'features':n_features,
                'query_rows':len(pred),**score(bundle['y_val'],pred),'seconds':time.perf_counter()-started,
                'source_sha256':bundle['hashes']}
            write_json(path/'results.json',record)
            print('FEATURE MATRIX RESULT',stage,name,record['rmse_sgd'],flush=True)
            del model
            gc.collect()


def repair_xgb_exports(stage):
    """Replay existing models to upgrade float32 CSV text to full precision."""
    sys.path.insert(0,str(ROOT/'.cache/model-deps'))
    bundle=feature_data(stage)
    for kind in KINDS:
        name='xgb_'+kind
        directory=ART/'matrix'/stage/name
        record=json.loads((OUT/'matrix'/stage/name/'results.json').read_text())
        assert record['source_sha256']==bundle['hashes']
        saved=joblib.load(directory/'model.joblib')
        _,q=inputs(bundle,kind)
        prediction=np.asarray(saved['model'].predict(q),dtype=float)
        original=pd.read_csv(directory/'predictions.csv')
        np.testing.assert_array_equal(original.source_row,bundle['row_val'])
        np.testing.assert_allclose(original.prediction,prediction,atol=.001,rtol=0)
        for key,value in score(bundle['y_val'],prediction).items():
            np.testing.assert_allclose(value,record[key],atol=1e-8,rtol=0)
        original['prediction']=prediction
        original.to_csv(directory/'predictions.csv',index=False)
        print('FULL-PRECISION EXPORT',stage,name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','outer','june'])
    parser.add_argument('--families',nargs='+',choices=FAMILIES)
    parser.add_argument('--kinds',nargs='+',choices=KINDS)
    args=parser.parse_args()
    run(args.stage,args.families,args.kinds)

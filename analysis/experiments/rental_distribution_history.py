"""Prior-only hierarchical rent distributions, normalized by historical market month."""
from pathlib import Path
import numpy as np,pandas as pd

GRAINS={
 'STREET_TYPE':['TOWN','STREET','FLAT_TYPE_CLEAN'],
 'BUILDING':['TOWN','BLOCK','STREET'],
 'TOWN_MODEL':['TOWN','FLAT_MODEL_CLEAN'],
 'AREA_TYPE_SIZE':['PLANNING_AREA','FLAT_TYPE_CLEAN','_AREA_BIN'],
 'TYPE_MODEL_SIZE':['FLAT_TYPE_CLEAN','FLAT_MODEL_CLEAN','_AREA_BIN'],
}
STATS=['mean','std','q10','median','q90','low_share','high_share','count']


def distribution_history(fitting,prediction):
    f=fitting.copy();v=prediction.copy()
    fm=pd.PeriodIndex(f.RENT_APPROVAL_DATE,freq='M');vm=pd.PeriodIndex(v.RENT_APPROVAL_DATE,freq='M')
    if fm.max()>=vm.min():raise ValueError('Fitting months must precede prediction')
    f['_MONTH']=fm;v['_MONTH']=vm
    for frame in [f,v]:frame['_AREA_BIN']=(frame.FLOOR_AREA_SQM/10).round().astype('Int64')
    market=f.groupby('_MONTH').MONTHLY_RENT.median().sort_index()
    f['_RATIO']=f.MONTHLY_RENT.to_numpy(float)/market.reindex(fm).to_numpy()
    result_fit=pd.DataFrame(index=f.index);result_val=pd.DataFrame(index=v.index)
    for name,keys in GRAINS.items():
        columns=[f'DIST_{name}_{stat.upper()}' for stat in STATS]
        result_fit[columns]=np.nan;result_val[columns]=np.nan
        for month in sorted(f._MONTH.unique()):
            previous=f[f._MONTH<month];current=f[f._MONTH==month]
            if previous.empty:continue
            groups=summary(previous,keys)
            result_fit.loc[current.index,columns]=lookup(current,groups,keys)
        groups=summary(f,keys)
        result_val[columns]=lookup(v,groups,keys)
    prior=market.shift(1)
    result_fit['DIST_PRIOR_MARKET']=prior.reindex(fm).fillna(2000).to_numpy()
    result_val['DIST_PRIOR_MARKET']=float(market.iloc[-1])
    return result_fit,result_val


def summary(frame,keys):
    return frame.groupby(keys,dropna=False)._RATIO.agg(
        mean='mean',std='std',q10=lambda s:s.quantile(.1),median='median',q90=lambda s:s.quantile(.9),
        low_share=lambda s:(s<.65).mean(),high_share=lambda s:(s>1.35).mean(),count='size')


def lookup(frame,groups,keys):
    keys_index=pd.MultiIndex.from_frame(frame[keys])
    a=groups.reindex(keys_index).copy()
    a['count']=np.log1p(a['count'])
    return a[STATS].to_numpy()


def main():
    import argparse
    from analysis.pipeline.config import DATA
    from analysis.pipeline.auxiliary import build_auxiliary_features
    from analysis.pipeline.data import check_output_path,sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()):parser.error('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    before=sha256(DATA/'train.csv')
    frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for origin,filename in [('2023-10','inner_features.csv'),('2024-04','outer_features.csv')]:
        print('Distribution origin',origin,flush=True)
        fit=frame[frame.RENT_APPROVAL_DATE<origin]
        val=frame[frame.RENT_APPROVAL_DATE>=origin]
        if origin=='2023-10':val=val[val.RENT_APPROVAL_DATE<'2024-04']
        a,b=distribution_history(fit,val)
        pd.concat([a,b]).to_csv(out/filename,index_label='source_row')
    assert sha256(DATA/'train.csv')==before


if __name__=='__main__':main()

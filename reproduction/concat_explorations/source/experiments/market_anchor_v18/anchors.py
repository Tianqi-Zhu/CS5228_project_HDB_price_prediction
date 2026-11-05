"""Forecast historical group rents at each query origin with bounded trends.

12-month trailing observations; fixed level pseudo-count 10 and slope pseudo-
count 50. Every regression is rebuilt from observations no later than origin.
"""
import numpy as np
import pandas as pd

LEVEL_PRIOR=10.; TREND_PRIOR=50.; WINDOW=12; MAX_CHANGE_FRACTION=.20


def _aggregate(frame,keys,origin):
    f=frame.copy(); f['_x']=f.MONTH_INDEX.to_numpy(dtype=float)-origin
    f['_y']=pd.to_numeric(f.MONTHLY_RENT,errors='coerce'); f=f[np.isfinite(f['_y'])]
    f['_xx']=f['_x']**2; f['_xy']=f['_x']*f['_y']; f['_yy']=f['_y']**2
    f['_all']='all'; keys=keys or ['_all']
    g=f.groupby(keys,dropna=False,sort=False).agg(n=('_y','size'),sy=('_y','sum'),sx=('_x','sum'),sxx=('_xx','sum'),sxy=('_xy','sum'),syy=('_yy','sum'),last=('MONTH_INDEX','max'))
    n=g.n.to_numpy(dtype=float); mean=g.sy.to_numpy()/n; den=g.sxx.to_numpy()-g.sx.to_numpy()**2/n
    slope=np.divide(g.sxy.to_numpy()-g.sx.to_numpy()*mean,den,out=np.zeros_like(n),where=den>1e-8)
    # Weighted fitted level at origin, not the mid-window mean.
    intercept=mean-slope*g.sx.to_numpy()/n
    sd=np.sqrt(np.maximum(g.syy.to_numpy()/n-mean**2,0))
    g['mean']=mean; g['raw_slope']=slope; g['intercept']=intercept; g['sd']=sd
    return g


def market_anchors(source,queries):
    required=['MONTH_INDEX','MONTHLY_RENT','TOWN_CLEAN','FLAT_TYPE_CLEAN']; f=source[required].copy()
    for c in ['TOWN_CLEAN','FLAT_TYPE_CLEAN']: f[c]=f[c].fillna('__missing__').astype(str)
    q=queries.reset_index(drop=True).copy(); n=len(q)
    if np.any(q.horizon.to_numpy()<=0): raise ValueError('horizon must be positive')
    for c in ['town','flat_type']: q[c]=q[c].fillna('__missing__').astype(str)
    result={name:{'anchor_sgd':np.zeros(n,dtype=np.float32),'count':np.zeros(n,dtype=np.float32),'recency':np.full(n,13,dtype=np.float32),'level':np.full(n,3,dtype=np.int8),'missing':np.ones(n,dtype=bool),'uncertainty_sgd':np.zeros(n,dtype=np.float32)} for name in ('market_level','market_trend')}
    for origin in np.unique(q.origin.to_numpy(dtype=int)):
        ix=np.flatnonzero(q.origin.to_numpy()==origin); hist=f[(f.MONTH_INDEX<=origin)&(f.MONTH_INDEX>=origin-WINDOW+1)]
        if hist.empty: continue
        glob=_aggregate(hist,[],origin).iloc[0]; flat=_aggregate(hist,['FLAT_TYPE_CLEAN'],origin); town=_aggregate(hist,['TOWN_CLEAN','FLAT_TYPE_CLEAN'],origin)
        target=q.iloc[ix]; fg=flat.reindex(pd.Index(target.flat_type.to_numpy(),name='FLAT_TYPE_CLEAN')); tg=town.reindex(pd.MultiIndex.from_arrays([target.town,target.flat_type],names=['TOWN_CLEAN','FLAT_TYPE_CLEAN']))
        fn=fg.n.fillna(0).to_numpy(); tn=tg.n.fillna(0).to_numpy(); fs=fg.sy.fillna(0).to_numpy(); ts=tg.sy.fillna(0).to_numpy()
        global_level=float(glob['mean']); fl=(fs+LEVEL_PRIOR*global_level)/(fn+LEVEL_PRIOR); tl=(ts+LEVEL_PRIOR*fl)/(tn+LEVEL_PRIOR)
        # Town slope is shrunk toward flat-type slope, flat type toward global.
        fsl=fg.raw_slope.fillna(0).to_numpy(); tsl=tg.raw_slope.fillna(0).to_numpy()
        parent_slope=(fn*fsl+TREND_PRIOR*float(glob['raw_slope']))/(fn+TREND_PRIOR)
        slope=(tn*tsl+TREND_PRIOR*parent_slope)/(tn+TREND_PRIOR)
        # Recenter each fitted regression from mean month to origin before shrink.
        g_origin=float(glob['intercept']); f_origin=np.where(fn>0,fg['intercept'].fillna(0).to_numpy(),0.)
        t_origin=np.where(tn>0,tg['intercept'].fillna(0).to_numpy(),0.)
        # Weight recentered trend-adjusted levels with parent, bounded by past mean.
        fi=(fn*f_origin+LEVEL_PRIOR*g_origin)/(fn+LEVEL_PRIOR); ti=(tn*t_origin+LEVEL_PRIOR*fi)/(tn+LEVEL_PRIOR)
        h=target.horizon.to_numpy(dtype=float); forecast=tl+np.clip((ti-tl)+slope*h,-MAX_CHANGE_FRACTION*tl,MAX_CHANGE_FRACTION*tl)
        lvl=np.where(tn>0,0,np.where(fn>0,1,2)).astype(np.int8)
        last=np.where(tn>0,tg['last'].fillna(-1).to_numpy(),np.where(fn>0,fg['last'].fillna(-1).to_numpy(),float(glob['last'])))
        sd=np.where(tn>1,tg['sd'].fillna(0).to_numpy(),np.where(fn>1,fg['sd'].fillna(0).to_numpy(),float(glob['sd'])))
        for name,value in [('market_level',tl),('market_trend',forecast)]:
            r=result[name]; r['anchor_sgd'][ix]=value; r['count'][ix]=tn; r['recency'][ix]=origin-last; r['level'][ix]=lvl; r['missing'][ix]=False; r['uncertainty_sgd'][ix]=sd
    return result

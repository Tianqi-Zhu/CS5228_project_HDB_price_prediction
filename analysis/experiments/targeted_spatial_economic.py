"""Small causal hierarchy and historical residual smoother for frozen TabPFN."""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = [f'HD_{level}_{name}' for level in ('TOWN','STREET')
                    for name in ('RECENT_SHRUNK','RELATIVE_GROWTH','COUNT_6M','STD_6M')]
HISTORY_FEATURES += ['HD_BUILDING_RECENT_SHRUNK','HD_BUILDING_RELATIVE_GROWTH','HD_HISTORY_MISSING']


def row_cutoffs(matrix):
    months = pd.PeriodIndex.from_ordinals(pd.Period('2021-01','M').ordinal + matrix.APPROVAL_MONTH_INDEX.to_numpy(int),freq='M')
    horizon = matrix.TEMP_FORECAST_HORIZON.to_numpy(float)
    origins = pd.Series(pd.PeriodIndex.from_ordinals(months.asi8-np.nan_to_num(horizon).astype(int),freq='M'),index=matrix.index)
    origins = origins.where(np.isfinite(horizon))
    return months, origins-1  # Extra full month for approval/publication delay.


def group_stats(past, query, keys):
    """Return aligned sufficient statistics, leaving empty groups at zero."""
    if not keys:
        n=len(past); s=past.MONTHLY_RENT.sum(); ss=(past.MONTHLY_RENT**2).sum()
        return np.tile([n,s,ss],(len(query),1)).astype(float)
    p=past.assign(_squared=past.MONTHLY_RENT**2)
    a=p.groupby(keys,observed=True).agg(n=('MONTHLY_RENT','size'),s=('MONTHLY_RENT','sum'),ss=('_squared','sum'))
    index=pd.MultiIndex.from_frame(query[keys]) if len(keys)>1 else pd.Index(query[keys[0]])
    return a.reindex(index).fillna(0).to_numpy(float)


def moments(stats, parent=None, shrink=20.):
    n,s,ss=stats.T
    if parent is None:
        mean=np.divide(s,n,out=np.full(len(n),np.nan),where=n>0)
        second=np.divide(ss,n,out=np.full(len(n),np.nan),where=n>0)
    else:
        mean=(s+shrink*parent[0])/(n+shrink)
        second=(ss+shrink*parent[1])/(n+shrink)
    return mean,second


def hierarchy_features(source, query, cutoffs, shrink=20.):
    """No current/query target is read; complete month windows are fixed at origin."""
    cutoffs=pd.Series(pd.PeriodIndex(cutoffs,freq='M'),index=query.index)
    result=pd.DataFrame(np.nan,index=query.index,columns=HISTORY_FEATURES)
    audit=[]
    for cutoff in sorted(cutoffs.dropna().unique()):
        selected=cutoffs.eq(cutoff)
        q=query.loc[selected]
        if (q.month<=cutoff).any(): raise ValueError('Prediction must follow source cutoff')
        permitted=source[(source.month<=cutoff)&(source.month>cutoff-6)]
        recent=permitted[permitted.month>cutoff-3]
        older=permitted[permitted.month<=cutoff-3]
        grains=[['FLAT_TYPE_CLEAN'],['TOWN_CLEAN','FLAT_TYPE_CLEAN'],
                ['TOWN_CLEAN','STREET','FLAT_TYPE_CLEAN'],['TOWN_CLEAN','STREET','BLOCK','FLAT_TYPE_CLEAN']]
        # Unknown flat types fall back to the national pool; no-history stays NaN.
        last_r=moments(group_stats(recent,q,[])); last_o=moments(group_stats(older,q,[]))
        last_s=moments(group_stats(permitted,q,[]))
        national_growth=None
        for level,keys in zip(('NATIONAL','TOWN','STREET','BUILDING'),grains):
            a,b,c=[group_stats(f,q,keys) for f in (recent,older,permitted)]
            pr,po,ps=[moments(z,parent=parent,shrink=shrink) for z,parent in ((a,last_r),(b,last_o),(c,last_s))]
            growth=100*np.log(pr[0]/po[0])
            if level=='NATIONAL': national_growth=growth
            else:
                result.loc[q.index,f'HD_{level}_RECENT_SHRUNK']=pr[0]
                result.loc[q.index,f'HD_{level}_RELATIVE_GROWTH']=growth-national_growth
                if level!='BUILDING':
                    result.loc[q.index,f'HD_{level}_COUNT_6M']=c[:,0]
                    result.loc[q.index,f'HD_{level}_STD_6M']=np.sqrt(np.maximum(ps[1]-ps[0]**2,0))
            last_r,last_o,last_s=pr,po,ps
        result.loc[q.index,'HD_HISTORY_MISSING']=float(permitted.empty)
        audit.append(dict(cutoff=str(cutoff),latest_source_month=str(permitted.month.max()) if len(permitted) else None,
                          source_rentals=len(permitted),query_rows=len(q),target_min_month=str(q.month.min())))
    result['HD_HISTORY_MISSING']=result.HD_HISTORY_MISSING.fillna(1.)
    result[['HD_TOWN_COUNT_6M','HD_STREET_COUNT_6M']]=result[['HD_TOWN_COUNT_6M','HD_STREET_COUNT_6M']].fillna(0.)
    return result,pd.DataFrame(audit)


def residual_smoother(history, query, cutoff, shrink=20., half_life=6.):
    """Use only matured, genuinely out-of-time residuals at a fixed batch cutoff.

    Nested estimates borrow the global residual mean -> flat type -> town/type
    -> street/type -> building/type. The final estimate is shrunken toward zero.
    All sample weights are preserved for the historical sampled query windows.
    """
    past=history[history.month<=cutoff].copy()
    if not len(past): return np.zeros(len(query)),dict(residual_rows=0,latest_residual_month=None)
    if (past.context_last_month>=past.month).any(): raise ValueError('Invalid in-sample residual provenance')
    ages=cutoff.ordinal-pd.PeriodIndex(past.month,freq='M').asi8
    w=past.weight.to_numpy(float)*2.**(-ages/half_life)
    # Rescale population weights to effective observed support rather than
    # treating sampled cases as thousands of independent observations.
    w=w/max(float(past.weight.mean()),1e-12)
    past['_w']=w;past['_wr']=w*past.residual.to_numpy(float)
    national=float(past._wr.sum()/(past._w.sum()+shrink))
    estimate=np.full(len(query),national)
    for keys in [['FLAT_TYPE_CLEAN'],['TOWN_CLEAN','FLAT_TYPE_CLEAN'],
                 ['TOWN_CLEAN','STREET','FLAT_TYPE_CLEAN'],['TOWN_CLEAN','STREET','BLOCK','FLAT_TYPE_CLEAN']]:
        stats=past.groupby(keys,observed=True)[['_w','_wr']].sum()
        index=pd.MultiIndex.from_frame(query[keys]) if len(keys)>1 else pd.Index(query[keys[0]])
        a=stats.reindex(index).fillna(0).to_numpy(float)
        estimate=(a[:,1]+shrink*estimate)/(a[:,0]+shrink)
    return estimate,dict(residual_rows=len(past),latest_residual_month=str(past.month.max()),source_prediction_origins=sorted(past.origin.unique()))

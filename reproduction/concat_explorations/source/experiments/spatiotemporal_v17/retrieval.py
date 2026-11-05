"""Causal per-query candidate selection; this does not invoke TabFM."""
import numpy as np

def retrieve(fit, query, k, seed=42, distance_scale=0.01, age_scale=12.):
    if k<=0 or distance_scale<=0 or age_scale<=0: raise ValueError('positive k and scales required')
    legal=np.flatnonzero((fit['month']<=query['origin']) & (fit['row_id']!=query['row_id']))
    if len(legal)==0: return np.array([],dtype=int)
    lat=np.asarray(fit['lat'])[legal]; lon=np.asarray(fit['lon'])[legal]
    dist=np.hypot(lat-query['lat'],lon-query['lon'])/distance_scale
    age=(query['origin']-np.asarray(fit['month'])[legal])/age_scale
    mismatch=(np.asarray(fit['flat_type'])[legal]!=query['flat_type']).astype(float)
    score=np.nan_to_num(dist,nan=1e6)+age+2*mismatch
    ranked=legal[np.argsort(score,kind='stable')]
    count=min(k,len(legal)); local_count=min(count,int(np.ceil(.75*count)))
    local=ranked[:local_count]; remainder=ranked[local_count:]
    rng=np.random.default_rng(seed); diverse=rng.permutation(remainder)[:count-local_count]
    return np.concatenate((local,diverse))

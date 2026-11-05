"""Past-only comparable-property features; no same-month or future labels."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.neighbors import NearestNeighbors

FEATURES=['COMP_MEAN','COMP_MEDIAN','COMP_STD','COMP_P10','COMP_P90','COMP_LOW_RATE',
          'COMP_RECENT_MEAN','COMP_RENT_PER_SQM','COMP_DISTANCE','COMP_COUNT']

def comparable_features(frame,cutoff='2024-04',k=32):
    dates=pd.PeriodIndex(frame.RENT_APPROVAL_DATE,freq='M').asi8
    origin=pd.Period(cutoff,freq='M').ordinal
    # Distance scales express approximate comparability, fixed before validation.
    numeric=frame[['LATITUDE','LONGITUDE','FLOOR_AREA_SQM','FLAT_AGE_YEARS']].copy().fillna(0).to_numpy(float)
    numeric=numeric/np.array([.015,.015,20.,20.])
    numeric=np.column_stack([numeric,dates/18.])
    values=frame.MONTHLY_RENT.to_numpy(float);area=frame.FLOOR_AREA_SQM.to_numpy(float)
    types=frame.FLAT_TYPE_CLEAN.to_numpy(str)
    result=pd.DataFrame(np.nan,index=frame.index,columns=FEATURES)
    for t in sorted(set(dates)):
        for typ in sorted(set(types[dates==t])):
            query=np.flatnonzero((dates==t)&(types==typ))
            candidates=np.flatnonzero((dates<min(t,origin))&(types==typ))
            if not len(candidates):continue
            search=NearestNeighbors(n_neighbors=min(k,len(candidates)),algorithm='ball_tree').fit(numeric[candidates])
            distance,indices=search.kneighbors(numeric[query]);matched=candidates[indices]
            rent=values[matched];age=t-dates[matched]
            weights=np.exp(-distance)/(1+age/6.)
            weighted=(weights*rent).sum(1)/weights.sum(1)
            per_area=(weights*rent/area[matched]).sum(1)/weights.sum(1)
            result.loc[frame.index[query],FEATURES]=np.column_stack([
                rent.mean(1),np.median(rent,axis=1),rent.std(1),np.quantile(rent,.1,axis=1),np.quantile(rent,.9,axis=1),
                (rent<=1900).mean(1),weighted,per_area,distance.mean(1),np.full(len(query),len(candidates) if len(candidates)<k else k)])
    return result

"""Historical rents adjusted to the last known market level, with prior-only inputs."""
import numpy as np,pandas as pd
from analysis.experiments.spatiotemporal_hybrid import add_month,historical_features,HISTORY_FEATURES


def adjusted_history(fitting,prediction):
    f=add_month(fitting);v=add_month(prediction)
    if f.empty or v.empty:raise ValueError('Nonempty fitting and prediction required')
    if f._MONTH.max()>=v._MONTH.min():raise ValueError('Prediction origin must follow all fitting months')
    monthly=f.groupby('_MONTH').MONTHLY_RENT.median().sort_index()
    # Normalizing labels in month s is safe for histories at t>s; those labels
    # never enter their own month's history. Inputs for t use market level t-1.
    ratio=f.copy()
    ratio.MONTHLY_RENT=f.MONTHLY_RENT.to_numpy(float)/monthly.reindex(f._MONTH).to_numpy()
    hf,hv=historical_features(ratio,v)
    all_months=pd.period_range(monthly.index.min(),monthly.index.max(),freq='M')
    prior=monthly.reindex(all_months).ffill().shift(1)
    factor=prior.reindex(f._MONTH).fillna(2000.).to_numpy()
    last=float(monthly.iloc[-1])
    for c in HISTORY_FEATURES:
        if c.endswith('_MEAN'):
            hf[c]=hf[c].to_numpy()*factor;hv[c]=hv[c]*last
    hf=hf.add_prefix('ADJ_');hv=hv.add_prefix('ADJ_')
    hf['ADJ_PRIOR_MARKET']=factor;hv['ADJ_PRIOR_MARKET']=last
    return hf,hv


def adjusted_reference(frame):
    """Shrink sparse property estimates toward block/area and prior market level."""
    prior=frame.ADJ_PRIOR_MARKET.to_numpy(float)
    area=frame.ADJ_HIST_AREA_FLAT_MEAN.fillna(frame.ADJ_PRIOR_MARKET).to_numpy(float)
    result=area
    for grain,strength in [('BLOCK_FLAT',10.),('SIGNATURE',3.)]:
        count=frame[f'ADJ_HIST_{grain}_COUNT'].fillna(0).to_numpy(float)
        value=frame[f'ADJ_HIST_{grain}_MEAN'].to_numpy(float)
        weight=count/(count+strength)
        weight=np.where(np.isfinite(value),weight,0.)
        result=weight*np.nan_to_num(value,nan=0.)+(1-weight)*result
    return np.where(np.isfinite(result),result,prior)

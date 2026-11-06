"""Training-only exponential month weights and a sample-weighted MLP loss."""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def recency_weights(months, half_life_months=None):
    """Mean-one weights, anchored at the most recent supplied fitting month.

    A record h months older has half the raw weight. Normalization preserves
    the average loss scale. Dates and labels outside fitting are not inputs.
    None is the exact uniform-weight baseline.
    """
    dates=pd.PeriodIndex(months,freq='M')
    if not len(dates) or dates.isna().any():
        raise ValueError('Nonempty valid fitting months required')
    if half_life_months is None:
        return np.ones(len(dates),dtype=np.float32)
    h=float(half_life_months)
    if not np.isfinite(h) or h<=0:
        raise ValueError('Half-life must be finite and positive')
    age=dates.asi8.max()-dates.asi8
    raw=np.exp2(-age/h)
    return (raw/raw.mean()).astype(np.float32)


def weight_summary(months, half_life_months=None):
    dates=pd.PeriodIndex(months,freq='M')
    weights=recency_weights(dates,half_life_months).astype(float)
    return {'half_life_months':half_life_months,'anchor_month':str(dates.max()),
            'oldest_month':str(dates.min()),'rows':len(dates),
            'weight_min':float(weights.min()),'weight_max':float(weights.max()),
            'weight_mean':float(weights.mean()),
            'effective_sample_size':float(weights.sum()**2/(weights**2).sum())}


def training_loss(output,target,weights,*,aux_weight=0.,band_targets=None,weight_auxiliary=False):
    regression=output[:,0] if output.ndim==2 else output
    loss=((regression-target).square()*weights).mean()
    if aux_weight:
        if weight_auxiliary:
            auxiliary=(F.cross_entropy(output[:,1:],band_targets,reduction='none')*weights).mean()
        else:
            # Preserve the original uniform baseline's exact reduction path.
            auxiliary=F.cross_entropy(output[:,1:],band_targets)
        loss=loss+aux_weight*auxiliary
    return loss

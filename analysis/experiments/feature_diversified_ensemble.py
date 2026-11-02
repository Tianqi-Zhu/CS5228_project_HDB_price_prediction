"""Small convex prediction ensembles with exact active-face optimization."""
from __future__ import annotations
from itertools import combinations
import numpy as np


def validate_predictions(predictions):
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=2 or not p.shape[0] or not p.shape[1] or not np.isfinite(p).all():
        raise ValueError('Finite, nonempty row-by-member predictions required')
    return p


def combine_predictions(predictions,weights):
    p=validate_predictions(predictions); w=np.asarray(weights,dtype=float)
    if w.shape!=(p.shape[1],) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-10,rtol=0):
        raise ValueError('Nonnegative member weights must sum to one')
    # Preserve vertex choices exactly, including the frozen baseline fallback.
    if np.count_nonzero(w)==1: return p[:,np.flatnonzero(w)[0]].copy()
    return p@w


def optimize_weights(actual,predictions,sample_weight=None):
    """Global convex quadratic optimum for <=3 members, no intercept.

    Enumerate all faces of the simplex. Solve the equality-constrained KKT
    system on each face and keep nonnegative stationary points/vertices.
    Singular identical predictors are supported. Exact ties prefer the first
    (baseline) member, then fewer active members. Never uses evaluation labels.
    """
    p=validate_predictions(predictions); y=np.asarray(actual,dtype=float)
    v=np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight,dtype=float)
    if p.shape[1]>3 or y.shape!=(len(p),) or v.shape!=(len(p),) or not np.isfinite(y).all() or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('Aligned finite targets, positive sample weights and at most3 members required')
    e=p-y[:,None]; v=v/v.sum(); scale=max(float(np.sqrt(np.sum(v[:,None]*e**2)/p.shape[1])),1.)
    covariance=(e/scale).T@(v[:,None]*(e/scale)); k=p.shape[1]; candidates=[]
    for size in range(1,k+1):
        for subset in combinations(range(k),size):
            q=covariance[np.ix_(subset,subset)]
            system=np.block([[q,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            solution=np.linalg.lstsq(system,np.r_[np.zeros(size),1.],rcond=None)[0][:size]
            if (solution < -1e-9).any() or not np.isclose(solution.sum(),1,atol=1e-8): continue
            w=np.zeros(k); w[list(subset)]=np.maximum(solution,0); w/=w.sum()
            # Evaluate actual residuals to avoid tiny negative quadratic-form
            # roundoff when members cancel perfectly.
            mse=float(np.sum(v*(e@w)**2))
            candidates.append((mse,w))
    best=min(s for s,_ in candidates)
    tied=[(s,w) for s,w in candidates if s<=best+max(1e-9,best*1e-13)]
    mse,w=min(tied,key=lambda t:(-t[1][0],np.count_nonzero(t[1]),tuple(-t[1])))
    assert np.isfinite(w).all() and (w>=0).all() and abs(w.sum()-1)<1e-10
    return w,dict(weighted_mse_sgd2=mse,weighted_rmse_sgd=float(np.sqrt(mse)),
        active_members=int(np.count_nonzero(w)),solver='Exact active-face convex quadratic optimization',intercept=False)

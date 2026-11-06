"""Paired saved-prediction statistics; no fitting, selection or network access."""
import numpy as np

def aligned_arrays(actual,candidate,reference):
    values=[np.asarray(x,dtype=np.float64) for x in [actual,candidate,reference]]
    if any(x.ndim!=1 for x in values) or len({x.shape for x in values})!=1 or not len(values[0]):
        raise ValueError('Expected nonempty aligned one-dimensional arrays')
    if not all(np.isfinite(x).all() for x in values):raise ValueError('Nonfinite predictions/targets')
    return values

def score(actual,prediction):
    y,p,_=aligned_arrays(actual,prediction,prediction);e=p-y
    return {'n':len(y),'rmse_sgd':float(np.sqrt(np.mean(e*e))),
            'mae_sgd':float(np.mean(abs(e))),'bias_sgd':float(e.mean()),'sse':float(e@e)}

def validate_identity(ids,actual,months,raw):
    ids=np.asarray(ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids):raise ValueError('Invalid/duplicate IDs')
    if ids.min()<0 or ids.max()>=len(raw):raise ValueError('ID outside raw data')
    if not np.array_equal(raw.iloc[ids].MONTHLY_RENT.to_numpy(float),actual):raise ValueError('Targets misaligned')
    if not np.array_equal(raw.iloc[ids].RENT_APPROVAL_DATE.to_numpy(str),months):raise ValueError('Months misaligned')

def aggregate_errors(actual,candidate,reference,labels):
    y,c,r=aligned_arrays(actual,candidate,reference)
    labels=np.asarray(labels)
    if labels.shape!=y.shape:raise ValueError('Cluster labels misaligned')
    unique,index=np.unique(labels,return_inverse=True)
    stats=np.column_stack([np.bincount(index,weights=x,minlength=len(unique)) for x in
                           [np.ones(len(y)),(c-y)**2,(r-y)**2,abs(c-y),abs(r-y)]])
    return unique,stats

def differences(stats):
    stats=np.asarray(stats,float)
    count=stats[...,0]
    if np.any(count<=0):raise ValueError('Empty resample')
    return np.stack([np.sqrt(stats[...,1]/count)-np.sqrt(stats[...,2]/count),
                     (stats[...,3]-stats[...,4])/count],axis=-1)

def moving_month_indices(months,repeats=3000,block=3,seed=42):
    if months<block or block<1 or repeats<1:raise ValueError('Invalid moving-block specification')
    starts=np.random.default_rng(seed).integers(0,months-block+1,size=(repeats,(months+block-1)//block))
    return (starts[...,None]+np.arange(block)).reshape(repeats,-1)[:,:months]

def paired_intervals(actual,candidate,reference,labels,method,repeats=3000,seed=42,block=3):
    units,stats=aggregate_errors(actual,candidate,reference,labels)
    rng=np.random.default_rng(seed)
    if method=='moving_month_block':
        draws=moving_month_indices(len(units),repeats,block,seed)
        distribution=differences(stats[draws].sum(axis=1))
    elif method=='building_cluster':
        distribution=np.empty((repeats,2))
        for start in range(0,repeats,32):
            count=min(32,repeats-start)
            draws=rng.integers(0,len(units),size=(count,len(units)))
            distribution[start:start+count]=differences(stats[draws].sum(axis=1))
    else:raise ValueError('Unknown resampling method')
    point=differences(stats.sum(axis=0))
    intervals=np.quantile(distribution,[.025,.975],axis=0)
    return [{'method':method,'metric':metric,'candidate_minus_reference':float(point[j]),
             'p025':float(intervals[0,j]),'p975':float(intervals[1,j]),'resamples':repeats,
             'seed':seed,'clusters':len(units),'rows':len(actual),
             'scope':'Descriptive fixed-prediction sensitivity; no training/selection uncertainty'}
            for j,metric in enumerate(['rmse_sgd','mae_sgd'])]

def blend_moments(actual,trees,tabpfn):
    y,t,p=aligned_arrays(actual,trees,tabpfn)
    e=t-y;d=p-t
    base=float(np.mean(e*e));cross=float(np.mean(e*d));dsq=float(np.mean(d*d))
    observed=float(np.mean(((t+p)/2-y)**2))
    reconstructed=base+cross+.25*dsq
    if not np.isclose(observed,reconstructed,rtol=0,atol=1e-8):raise ValueError('MSE identity failed')
    correlation=float(np.corrcoef(t-y,p-y)[0,1]) if np.std(t-y)>0 and np.std(p-y)>0 else None
    return {'tree_mse':base,'mean_e_times_d':cross,'mean_d_squared':dsq,'fixed_tabpfn_weight':.5,
            'observed_blend_mse':observed,'reconstructed_blend_mse':reconstructed,
            'error_correlation':correlation,
            'formula':'MSE(tree + w*(tabpfn-tree)) = E[e^2] + 2*w*E[e*d] + w^2*E[d^2]',
            'weight_search_performed':False}

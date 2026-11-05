"""Prediction-level metrics; negative paired MSE delta favors candidate."""
import numpy as np

def evaluate(pop,anchors,pred,baseline=None):
    y=np.asarray(pop['y_sgd'],dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
    if len(pred)!=len(y) or not np.isfinite(pred).all(): raise ValueError('invalid predictions')
    err=pred-y
    def score(mask):
        e=err[mask]; return {'n':len(e),'rmse_sgd':float(np.sqrt(np.mean(e**2))),'mae_sgd':float(np.mean(abs(e))),'bias_sgd':float(np.mean(e))}
    out=score(np.ones(len(y),bool))
    for col in ('month','horizon','town','flat_type'):
        vals=np.asarray(pop[col]);out['by_'+col]={str(v):score(vals==v) for v in np.unique(vals)}
    count=np.asarray(anchors['count']);out['by_anchor_support']={label:score(mask) for label,mask in [('none',count==0),('low',(count>0)&(count<10)),('high',count>=10)] if mask.any()}
    if 'block_known' in pop:
        known=np.asarray(pop['block_known'],dtype=bool);out['by_block_known']={label:score(mask) for label,mask in [('known',known),('unseen',~known)] if mask.any()}
    pred_residual=pred-anchors['anchor_sgd'];actual_residual=y-anchors['anchor_sgd']
    out.update(residual_prediction_mean_sgd=float(pred_residual.mean()),residual_prediction_std_sgd=float(pred_residual.std()),residual_correlation=float(np.corrcoef(pred_residual,actual_residual)[0,1]) if pred_residual.std()>1e-10 and actual_residual.std()>1e-10 else None)
    if baseline is not None:
        baseline=np.asarray(baseline,dtype=np.float64)
        if baseline.shape!=y.shape: raise ValueError('baseline shape mismatch')
        out['paired_mean_squared_error_delta']=float(np.mean(err**2-(baseline-y)**2))
    return out

def paired(candidate,baseline):
    if not np.array_equal(candidate['row_id'],baseline['row_id']) or not np.allclose(candidate['y'],baseline['y'],rtol=0,atol=.01): raise ValueError('paired rows or targets differ')
    y=np.asarray(candidate['y'],dtype=float);return float(np.mean((np.asarray(candidate['pred'])-y)**2-(np.asarray(baseline['pred'])-y)**2))

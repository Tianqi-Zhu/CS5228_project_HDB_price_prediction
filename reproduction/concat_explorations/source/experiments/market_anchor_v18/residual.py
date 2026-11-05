"""Fit-only design matrix and target units for matched anchor experiments."""
import hashlib,json
import numpy as np

def reconstruct(pred,anchor_sgd,mode):
    if mode=='residual': return 1000*np.asarray(pred)+np.asarray(anchor_sgd)
    if mode=='direct': return 1000*np.asarray(pred)
    raise ValueError('unknown target mode')

def build_inputs(bank,anchors):
    for split in ('train','val'):
        p=bank[split]; a=anchors[split]; n=len(p['row_id'])
        if not np.array_equal(p['row_id'],a['row_id']): raise ValueError('anchor row IDs differ')
        for key in ('anchor_sgd','count','recency','level','missing','uncertainty_sgd'):
            if len(a[key])!=n: raise ValueError('anchor field length mismatch')
        if not np.isfinite(a['anchor_sgd']).all(): raise ValueError('nonfinite anchor')
    def numeric(split):
        p=bank[split];a=anchors[split]
        return np.column_stack((p['x_num'],p['horizon'],a['anchor_sgd']/1000,a['count'],a['recency'],a['missing'],a['uncertainty_sgd']/1000)).astype(np.float64)
    tr=numeric('train'); va=numeric('val'); tr[~np.isfinite(tr)]=np.nan;va[~np.isfinite(va)]=np.nan
    med=np.nanmedian(tr,axis=0);med=np.nan_to_num(med,nan=0.0)
    tr=np.where(np.isnan(tr),med,tr); va=np.where(np.isnan(va),med,va)
    mean=tr.mean(axis=0); scale=tr.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
    x={}; encodings=[]
    for i in range(bank['train']['x_cat'].shape[1]): encodings.append((('x_cat',i),np.unique(bank['train']['x_cat'][:,i]).astype(str)))
    encodings.append((('level',None),np.unique(anchors['train']['level'].astype(str))))
    for split,nums in [('train',tr),('val',va)]:
        cols=[(nums-mean)/scale]
        for (kind,i),vocab in encodings:
            vals=bank[split]['x_cat'][:,i].astype(str) if kind=='x_cat' else anchors[split]['level'].astype(str)
            cols.append((vals[:,None]==vocab[None,:]).astype(float))
            cols.append((~np.isin(vals,vocab))[:,None].astype(float))
        x[split]=np.column_stack(cols).astype(np.float32)
    state={'median':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'category_vocabularies':[v.tolist() for _,v in encodings]}
    state['fingerprint']=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
    y={split:{'direct':np.asarray(bank[split]['y_sgd'],dtype=np.float64)/1000,'residual':(np.asarray(bank[split]['y_sgd'],dtype=np.float64)-anchors[split]['anchor_sgd'])/1000} for split in ('train','val')}
    return x,y,state

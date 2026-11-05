"""Train-fitted encoders and causally bounded rent aggregates for v17."""
import numpy as np
import pandas as pd


def fit_categories(frame, fit_indices, columns):
    encoded=[]; vocabs={}
    for col in columns:
        s=frame[col].fillna('__missing__').astype(str)
        vocab={v:i+1 for i,v in enumerate(sorted(s.iloc[fit_indices].unique()))}
        encoded.append(s.map(vocab).fillna(0).to_numpy(dtype=np.int64)); vocabs[col]=vocab
    return np.column_stack(encoded),vocabs


def temporal_features(frame, origins, cutoff, source_frame=None):
    """Group monthly history ends at each query origin; source labels stop at cutoff."""
    origin=np.minimum(np.asarray(origins,dtype=int),cutoff); n=len(frame)
    f=(frame if source_frame is None else source_frame).copy(); f['_rent']=pd.to_numeric(f.MONTHLY_RENT,errors='coerce')
    area=pd.to_numeric(f.FLOOR_AREA_SQM,errors='coerce'); f['_psm']=f['_rent']/area.where(area>0)
    f=f.loc[f.MONTH_INDEX<=cutoff]
    outputs=[]; names=[]; seqs=[]; masks=[]
    for label,keys in [('town_ft',['TOWN_CLEAN','FLAT_TYPE_CLEAN']),('block_ft',['BLOCK_ID','FLAT_TYPE_CLEAN'])]:
        # Missing group keys are explicit; no merge against future labels.
        def key(df): return df[keys].fillna('__missing__').astype(str).agg('\x1f'.join,axis=1)
        source=f.assign(_key=key(f)); query_key=key(frame).to_numpy()
        monthly=source.groupby(['_key','MONTH_INDEX'],sort=False).agg(rent_sum=('_rent','sum'), rent_n=('_rent','count'), psm_sum=('_psm','sum'), psm_n=('_psm','count'))
        monthly.index=monthly.index.set_names(['key','month'])
        rent=np.zeros((n,12),dtype=np.float64); counts=np.zeros_like(rent); psm=np.zeros_like(rent); pcounts=np.zeros_like(rent)
        for t in range(12):
            idx=pd.MultiIndex.from_arrays([query_key,origin-11+t],names=['key','month'])
            values=monthly.reindex(idx).fillna(0).to_numpy()
            rent[:,t],counts[:,t],psm[:,t],pcounts[:,t]=values.T
        seq=np.stack([np.divide(rent,counts,out=np.zeros_like(rent),where=counts>0)/1000, np.log1p(counts)],axis=2)
        seqs.append(seq.astype(np.float32)); masks.append(np.repeat((counts>0)[:,:,None],2,axis=2))
        averages=[]
        for window in (3,6,12):
            c=counts[:,-window:].sum(1); pc=pcounts[:,-window:].sum(1)
            avg=np.divide(rent[:,-window:].sum(1),c,out=np.zeros(n),where=c>0)
            avp=np.divide(psm[:,-window:].sum(1),pc,out=np.zeros(n),where=pc>0)
            # Support makes shrinkage explicit: sparse means shrink toward zero.
            avg=avg*c/(c+5); avp=avp*pc/(pc+5); averages.append(avg)
            outputs.extend([avg,avp,np.log1p(c),(c==0).astype(float),c/(c+5)])
            names.extend([f'{label}_{window}_{k}' for k in ('rent_shrunk','psm_shrunk','log_count','missing','support')])
        recency=np.full(n,13.0)
        for t in range(12): recency=np.where(counts[:,t]>0,11-t,recency)
        outputs.extend([recency,averages[0]-averages[2]]); names.extend([label+'_recency',label+'_trend_3_12'])
    return np.column_stack(outputs).astype(np.float32),np.concatenate(seqs,axis=2),np.concatenate(masks,axis=2),names

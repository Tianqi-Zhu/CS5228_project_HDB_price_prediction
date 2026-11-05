"""Create isolated support-aware screen banks from frozen causal arrays."""
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from .contract import OUT, load_bank


def support_trends(sequence, mask):
    """Recent-minus-year mean rent in kSGD, with explicit missingness.

    A missing recent window falls back to its own long-window mean, so its
    difference is zero; a completely missing block falls back to town trend.
    Missing town history yields zero, represented by its separate flag.
    """
    s=np.asarray(sequence); m=np.asarray(mask,dtype=bool)
    if s.ndim!=3 or s.shape[1:]!=(12,4) or m.shape!=s.shape: raise ValueError('expected N x 12 x 4 sequence and mask')
    trends=[]; flags=[]
    for channel in (0,2):
        counts=np.where(m[:,:,channel],np.expm1(s[:,:,channel+1].astype(np.float64)),0.)
        weighted=s[:,:,channel].astype(np.float64)*counts
        all_count=counts.sum(axis=1); short_count=counts[:,-3:].sum(axis=1)
        long=np.divide(weighted.sum(axis=1),all_count,out=np.zeros(len(s)),where=all_count>0)
        recent=np.divide(weighted[:,-3:].sum(axis=1),short_count,out=long.copy(),where=short_count>0)
        trend=recent-long
        if channel==2: trend=np.where(all_count>0,trend,trends[0])
        trends.append(trend); flags.extend([(short_count==0).astype(np.float32),(all_count==0).astype(np.float32)])
    return np.column_stack(trends).astype(np.float32),np.column_stack(flags).astype(np.float32)


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    base=OUT/'orig'/'screen'/'causal_no_mrt'
    definitions=[('F4_support','F4'),('F3_no_block','F3')]
    for new,_ in definitions:
        if (base/new).exists(): raise FileExistsError(f'refusing to overwrite {base/new}')
    generated={}
    for new,parent in definitions:
        b=load_bank(parent,variant='causal_no_mrt'); names=list(b['numeric_names']); cnames=list(b['categorical_names']); cards=list(b['cardinalities']); extra={}
        if new=='F4_support':
            old=['town_ft_trend_3_12','block_ft_trend_3_12']; keep=[i for i,n in enumerate(names) if n not in old]
            assert len(keep)==len(names)-2
            tr,tf=support_trends(b['train']['rent_sequence'],b['train']['sequence_mask']); va,vf=support_trends(b['val']['rent_sequence'],b['val']['sequence_mask'])
            mu=tr.mean(0,dtype=np.float64); std=tr.std(0,dtype=np.float64); std[std<1e-8]=1
            for part,trend,flags in [('train',tr,tf),('val',va,vf)]:
                b[part]['x_num']=np.concatenate((b[part]['x_num'][:,keep],np.clip((trend-mu)/std,-8,8).astype(np.float32),flags),axis=1)
            names=[names[i] for i in keep]+['town_ft_support_trend_3_12','block_ft_support_trend_3_12','town_ft_recent_missing','town_ft_all_missing','block_ft_recent_missing','block_ft_all_missing']
            extra={'trend_scaler_mean':mu.tolist(),'trend_scaler_std':std.tolist(),'fallback':'missing recent uses own 12-month mean; fully missing block uses town trend; fully missing town uses zero; flags retained','sequence_units':'ksgd and log1p count','fit_rows':len(tr)}
        else:
            index=cnames.index('BLOCK_ID'); cnames.pop(index); cards.pop(index)
            for part in ('train','val'): b[part]['x_cat']=np.delete(b[part]['x_cat'],index,axis=1)
            extra={'removed_category':'BLOCK_ID'}
        temp=Path(tempfile.mkdtemp(prefix='.'+new+'-',dir=base)); arr={}
        for part in ('train','val'):
            for k,value in b[part].items(): arr[part+'_'+k]=value
        np.savez_compressed(temp/'arrays.npz',**arr)
        manifest={**b['manifest'],'bank_id':new,'parent_bank':parent,'parent_array_sha256':b['manifest']['array_sha256'],'array_sha256':_sha(temp/'arrays.npz'),'numeric_names':names,'categorical_names':cnames,'cardinalities':cards,'variant_details':extra,'status':'ready','screen_only':True}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)); os.rename(temp,base/new)
        generated[new]=manifest['array_sha256']
    return generated

if __name__=='__main__': print(json.dumps(build(),indent=2))

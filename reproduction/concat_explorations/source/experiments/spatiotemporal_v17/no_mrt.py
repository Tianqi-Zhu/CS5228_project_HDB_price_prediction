"""Create a separately versioned, uniform no-MRT variant of audited snapshot banks."""
import argparse, hashlib, json, os
from .contract import OUT, load_bank

REMOVED = ('NEAREST_OPEN_MRT_KM','OPEN_MRT_COUNT_500M','OPEN_MRT_COUNT_1000M','OPEN_MRT_COUNT_2000M')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def build(split='orig',stage='screen'):
    import numpy as np
    base=OUT/split/stage; dest=base/'causal_no_mrt'; dest.mkdir(parents=True,exist_ok=True)
    for bank_id in ('F0','F1','F2','F3','F4','F5'):
        bank=load_bank(bank_id,split,stage); old=bank['manifest']; names=bank['numeric_names']
        assert set(REMOVED).issubset(names)
        keep=[i for i,name in enumerate(names) if name not in REMOVED]
        path=dest/bank_id; path.mkdir(exist_ok=True)
        out={}
        for part in ('train','val'):
            for key,value in bank[part].items(): out[part+'_'+key]=value[:,keep] if key=='x_num' else value
        temporary=path/'arrays.tmp.npz'; np.savez_compressed(temporary,**out); os.replace(temporary,path/'arrays.npz')
        m={**old,'variant':'causal_no_mrt','numeric_names':[names[i] for i in keep],
           'array_sha256':sha(path/'arrays.npz'),'parent_array_sha256':old['array_sha256'],
           'removed_features':list(REMOVED),'status':'auditing',
           'provenance_caveat':'MRT snapshot fields excluded. Mall and school features use current snapshots with historical availability unverified; other spatial source availability is not fully proven.',
           'comparison':'Within-variant screen only; F0/F1 snapshot reference semantics are preserved separately.'}
        tmp=path/'manifest.tmp'; tmp.write_text(json.dumps(m,indent=2)); os.replace(tmp,path/'manifest.json')
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--split',default='orig'); p.add_argument('--stage',default='screen'); a=p.parse_args(); print(build(a.split,a.stage))

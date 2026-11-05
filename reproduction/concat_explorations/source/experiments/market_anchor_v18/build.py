"""Build v18 screen anchors in origin batches; no model training or refit."""
import hashlib,json,os,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.fusion_contract_v2 import export,contract as source_contract
from experiments.spatiotemporal_v17.contract import load_bank
from experiments.spatiotemporal_v17.anchor_probe import refreshed_anchor
from .anchors import market_anchors
from .contract import OUT,ROOT


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build():
    start=time.monotonic(); names=['original','refreshed_12m','market_level','market_trend']
    if any((OUT/name).exists() for name in names): raise FileExistsError('refusing to overwrite v18 anchor artifacts')
    b=load_bank('F0',variant='causal_no_mrt'); frame=export.load_source(); out={name:{} for name in names}; counts={}
    for part in ('train','val'):
        p=b[part]; ids=p['row_id']; raw=frame.iloc[ids]
        assert np.array_equal(raw.SOURCE_ROW_ID.to_numpy(),ids) and np.allclose(raw.MONTHLY_RENT.to_numpy(),p['y_sgd'])
        assert np.array_equal(raw.MONTH_INDEX.to_numpy(),p['month']) and np.array_equal(p['month']-p['origin'],p['horizon'])
        assert np.allclose((p['residual']+p['anchor_ksgd'])*1000,p['y_sgd'],atol=.001)
        q=pd.DataFrame({'origin':p['origin'],'horizon':p['horizon'],'town':p['town'],'flat_type':p['flat_type']})
        market=market_anchors(frame,q); common=market['market_level']; counts[part]={'n':len(ids),'no_history':int(common['missing'].sum()),'origin_min':int(p['origin'].min()),'origin_max':int(p['origin'].max())}
        for name in names:
            anchor=market[name]['anchor_sgd'] if name in market else (p['anchor_ksgd']*1000 if name=='original' else refreshed_anchor(p['rent_sequence'],p['sequence_mask'],p['anchor_ksgd']*1000,12))
            arrays={'row_id':ids,'anchor_sgd':np.asarray(anchor,dtype=np.float32),**{k:common[k] for k in ('count','recency','level','missing','uncertainty_sgd')}}
            assert np.isfinite(arrays['anchor_sgd']).all() and np.isfinite(arrays['uncertainty_sgd']).all()
            for key,value in arrays.items(): out[name][part+'_'+key]=value
    assert counts['train']['n']==30000 and counts['val']['n']==5000 and counts['val']['origin_min']==counts['val']['origin_max']==38
    source_paths=[source_contract.SOURCE, ROOT/'experiments/market_anchor_v18/anchors.py', ROOT/'experiments/market_anchor_v18/build.py',ROOT/'experiments/spatiotemporal_v17/anchor_probe.py']
    source_hashes={str(Path(p).relative_to(ROOT)):sha(p) for p in source_paths}
    for name in names:
        temp=Path(tempfile.mkdtemp(prefix='.'+name+'-',dir=OUT)); np.savez_compressed(temp/'arrays.npz',**out[name]); digest=sha(temp/'arrays.npz')
        manifest={'anchor_id':name,'status':'auditing','stage':'orig/screen','population_counts':counts,'array_sha256':digest,'v17_bank_sha256':b['manifest']['array_sha256'],'source_hashes':source_hashes,'units':'SGD','reliability_fields':['count','recency','level','missing','uncertainty_sgd'],'reliability_definition':'Identical market 12-month group descriptors across all anchors; count is town-flat-type count, recency/level refer to chosen fallback level, missing indicates no history. level 0 town, 1 flat type, 2 global, 3 no history. uncertainty is past observed rent SD at chosen level.', 'asof':'per query origin: source months <= origin and >= origin-11, with every trend estimated separately for that origin; all labels on later months excluded; horizon >0. Original and refreshed controls use frozen causal v17 data.', 'model':'market_level: 12-month transaction-weighted group mean with prior10 to flat-type, flat-type prior10 to global. market_trend: weighted linear slope in trailing12 months, town slope prior50 to flat-type and flat-type prior50 to global; forecast intercept and slope per origin, capped ±20% of past smoothed level. Original is cumulative v17; refreshed_12m is local block-to-town prior10.', 'source_provenance':'No additional data. Current mall/school snapshot availability remains unverified in downstream F0 features; market anchors use past rents, town and flat type.', 'elapsed_seconds':time.monotonic()-start}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)); os.rename(temp,OUT/name)
    print(json.dumps({'anchors':names,'counts':counts,'seconds':time.monotonic()-start},indent=2))

if __name__=='__main__': build()

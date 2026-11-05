"""Audit frozen banks and write provenance diagnostics from fit rows only."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.fusion_contract_v2 import export, contract
from .contract import load_bank,OUT,ROOT


def audit(stage='screen',split='orig',variant='snapshot'):
    root=OUT/split/stage; base=root if variant=='snapshot' else root/variant; reference=load_bank('F1',split,stage,variant); t,v=reference['train'],reference['val']; frame=export.load_source(); f=frame.iloc[t['row_id']]
    for name in ('F0','F1','F2','F3','F4','F5'):
        b=load_bank(name,split,stage,variant); m=b['manifest']; path=base/name/'arrays.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m['array_sha256']
        for part in ('train','val'):
            x=b[part]; ref=reference[part]
            assert np.array_equal(x['row_id'],ref['row_id']) and np.array_equal(x['origin'],ref['origin'])
            assert x['x_num'].shape[1]==len(b['numeric_names']) and x['x_cat'].shape[1]==len(b['categorical_names'])
            assert np.isfinite(x['x_num']).all() and np.isfinite(x['rent_sequence']).all()
            for j,c in enumerate(b['cardinalities']): assert x['x_cat'][:,j].min()>=0 and x['x_cat'][:,j].max()<c
        assert not np.intersect1d(b['train']['row_id'],b['val']['row_id']).size
    cols=['YEARS_SINCE_LEASE_COMMENCEMENT','BUILDING_AGE_YEARS']; ages=f[cols].apply(pd.to_numeric,errors='coerce'); ok=ages.notna().all(axis=1)
    diff=ages.loc[ok,cols[0]]-ages.loc[ok,cols[1]]
    groups=frame.iloc[np.r_[t['row_id'],v['row_id']]].copy(); groups['population']=['train']*len(t['row_id'])+['val']*len(v['row_id'])
    groups.groupby(['population','MONTH_INDEX','TOWN_CLEAN','FLAT_TYPE_CLEAN'],dropna=False).size().rename('rows').reset_index().to_csv(base/'subgroup_counts.csv',index=False)
    state=json.loads((root/'fit_state.json').read_text()); spatial=load_bank('F3',split,stage,variant)
    result={'banks_verified':True,'train_rows':len(t['row_id']),'validation_rows':len(v['row_id']),'cold_start_train':int((t['origin']<0).sum()),'validation_unknown_fraction':{name:float((spatial['val']['x_cat'][:,i]==0).mean()) for i,name in enumerate(spatial['categorical_names'])},'fit_missing_fraction':{c:float(f[c].isna().mean()) for c in contract.STATIC_NUMERIC_COLUMNS},'fit_constant_numeric': [n for j,n in enumerate(reference['numeric_names']) if np.all(t['x_num'][:,j]==t['x_num'][0,j])], 'fit_pruned':state['pruned'],'lease_age_minus_building_age':{'both_present':int(ok.sum()),'equal_fraction':float((diff==0).mean()),'min':float(diff.min()),'max':float(diff.max())},'provenance':'MRT, malls and schools are current snapshots; no historical opening dates. MAX_FLOOR is building maximum, not unit floor. Opening-date validity cannot be verified with source.', 'variant':variant,'warning':('Current MRT snapshot violates origin-time availability; banks are reference only.' if variant=='snapshot' else 'MRT snapshot excluded uniformly; mall and school historical availability remains unverified.')}
    if variant=='causal_no_mrt':
        from .no_mrt import REMOVED
        for name in ('F0','F1','F2','F3','F4','F5'):
            bank=load_bank(name,split,stage,variant); parent=load_bank(name,split,stage)
            assert not set(REMOVED).intersection(bank['numeric_names'])
            keep=[i for i,n in enumerate(parent['numeric_names']) if n not in REMOVED]
            for part in ('train','val'):
                assert np.array_equal(bank[part]['x_num'],parent[part]['x_num'][:,keep])
                for key in set(parent[part])-{'x_num'}: assert np.array_equal(bank[part][key],parent[part][key])
        result['uniform_removal_verified']=True
    (base/'audit.json').write_text(json.dumps(result,indent=2)); return result
if __name__=='__main__': print(json.dumps(audit(),indent=2))

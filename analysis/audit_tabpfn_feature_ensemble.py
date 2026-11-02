"""Independent label-mutation audit of the newly reused spatial fields."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analysis.compare_tabpfn_feature_ensemble import OUT,ROOT,LOCAL,ORIGINS,load_prepared,load_bundle,check_sources,raw_frame
from analysis.compare_local_economy_landscape import immutable,write_json
from analysis.experiments.local_market_fields import historical_market_features
from analysis.data import sha256


def main():
    check_sources()
    immutable(OUT/'independent_audit_plan.json',dict(code_sha256=sha256(Path(__file__)),no_model_or_weight_selection=True,
        test='Spatial numerical replay and poison every unavailable fitting label at query and context pseudo-origins'))
    raw=raw_frame(); raw['_MONTH']=pd.PeriodIndex(raw.month,freq='M')
    links=pd.read_csv(LOCAL/'transaction_building_links.csv',index_col=0); raw['building_id']=links.loc[raw.index,'building_id']
    buildings=pd.read_csv(LOCAL/'canonical_buildings.csv').set_index('building_id',drop=False)
    cfg=json.loads((ROOT/'analysis/configs/local_economy_landscape.json').read_text())
    main=load_bundle('2024-06'); result=[]
    for origin in ORIGINS:
        b=load_bundle(origin); a=load_prepared(origin,'spatial'); x,q=b['X_train'],b['X_query']
        columns=[c for c in a['X_train'] if c not in x]; assert len(columns)==12
        assert x.index.is_unique and q.index.is_unique and set(x.index).isdisjoint(q.index)
        past=raw.loc[x.index]
        assert past.month.max()<origin<=raw.loc[q.index,'month'].min()
        selected=raw.loc[q.index].groupby('month',sort=True).head(2).index
        specifications=[('query',pd.Period(origin,'M')-1,selected)]
        assignments=main['train_temporal_origins'].loc[x.index]
        for cutoff in [pd.Period('2021-06','M'),pd.Period(origin,'M')-2]:
            ids=assignments.index[assignments.forecast_origin.eq(cutoff)][:4]
            if len(ids): specifications.append(('train',cutoff,ids))
        for split,cutoff,ids in specifications:
            query=raw.loc[ids].copy(); query['MONTHLY_RENT']=1e12
            origins=pd.Series(cutoff,index=ids)
            observed=historical_market_features(past,query,origins,buildings,cfg)[columns]
            expected=a['X_'+split].loc[ids,columns]
            np.testing.assert_allclose(observed,expected,atol=1e-8,rtol=0,equal_nan=True)
            mutated=past.copy(); unavailable=mutated._MONTH>cutoff-cfg['history_publication_lag_months']
            mutated.loc[unavailable,'MONTHLY_RENT']=-1e12
            poisoned=historical_market_features(mutated,query,origins,buildings,cfg)[columns]
            np.testing.assert_allclose(observed,poisoned,atol=0,rtol=0,equal_nan=True)
            result.append(dict(origin=origin,split=split,feature_origin=str(cutoff),last_permitted_rental_month=str(cutoff-1),
                rows=len(ids),altered_unavailable_labels=int(unavailable.sum()),max_replay_difference=float(np.nanmax(np.abs(observed.to_numpy()-expected.to_numpy()))),
                current_query_targets_ignored=True,poisoning_unavailable_context_labels_has_no_effect=True))
            print('AUDIT',origin,split,cutoff,len(ids),'passed',flush=True)
    write_json(OUT/'independent_feature_audit.json',dict(verified=True,checks=result,columns=12,no_model_changes=True,
        inherited_limitation='Static MRT geography timing remains uncertified and unchanged'))
    check_sources(); print('INDEPENDENT SPATIAL AUDIT PASSED',flush=True)


if __name__=='__main__': main()

"""Check real Kaggle rows at the March overlap and long-horizon cutoffs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from analysis.train_tabpfn_feature_ensemble_submission import OUT,ROOT,DATA,LOCAL,full_base,load_member,source_rows,verify_sources
from analysis.compare_local_economy_landscape import immutable,write_json
from analysis.experiments.local_market_fields import historical_market_features
from analysis.experiments.calendar_encoding import encoded_features
from analysis.data import sha256


def main():
    verify_sources(); immutable(OUT/'test_feature_audit_plan.json',dict(code_sha256=sha256(Path(__file__)),
        purpose='Replay sampled true test rows and poison unavailable rental labels; never fit or select on test targets'))
    b=full_base(); spatial=load_member('spatial'); temporal=load_member('temporal')
    links=pd.read_csv(LOCAL/'transaction_building_links.csv',index_col=0)
    buildings=pd.read_csv(LOCAL/'canonical_buildings.csv').set_index('building_id',drop=False)
    train=source_rows(pd.read_csv(DATA/'train.csv'),links,0); test=source_rows(pd.read_csv(DATA/'test.csv'),links,len(train))
    cfg=json.loads((ROOT/'analysis/configs/local_economy_landscape.json').read_text())
    names=json.loads((OUT/'settings.json').read_text())['spatial_columns']; records=[]
    for origin in ['2025-02','2025-03']:
        candidates=test.loc[np.asarray(b['query_origin'])==origin]
        query=candidates.groupby('_MONTH',sort=True).head(3).copy(); query['MONTHLY_RENT']=1e12
        origins=pd.Series(pd.Period(origin,'M'),index=query.index)
        actual=historical_market_features(train,query,origins,buildings,cfg)[names]
        expected=spatial['X_query'].loc[query.index,names]
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0,equal_nan=True)
        latest=pd.Period(origin,'M')-1; poisoned=train.copy(); mask=poisoned._MONTH>latest
        poisoned.loc[mask,'MONTHLY_RENT']=-1e12
        changed=historical_market_features(poisoned,query,origins,buildings,cfg)[names]
        np.testing.assert_allclose(actual,changed,atol=0,rtol=0,equal_nan=True)
        records.append(dict(feature_origin=origin,latest_spatial_rental_month=str(latest),test_rows=len(query),
            query_months=sorted(query._MONTH.astype(str).unique()),unavailable_labels_poisoned=int(mask.sum()),
            maximum_replay_difference=float(np.nanmax(np.abs(actual.to_numpy()-expected.to_numpy()))),
            query_targets_ignored=True,unavailable_training_targets_ignored=True))
        print('REAL TEST FEATURE AUDIT',origin,len(query),'passed',flush=True)
    # Verify future actual dates survive the inherited temporal forecast cap.
    months=pd.PeriodIndex(b['query_month'],freq='M'); origins=pd.PeriodIndex(b['query_origin'],freq='M')
    np.testing.assert_array_equal(b['X_query'].TEMP_FORECAST_HORIZON,months.asi8-origins.asi8)
    reconstructed=encoded_features(b['X_query'],b['query_month'],'cyclical',calendar=True)
    pd.testing.assert_frame_equal(reconstructed,temporal['X_query'])
    later=months>pd.Period('2026-01','M'); assert later.any()
    assert (b['X_query'].loc[later,'TEMP_FORECAST_HORIZON']>10).all()
    write_json(OUT/'test_feature_audit.json',dict(verified=True,checks=records,actual_future_calendar_and_horizon_preserved=True,
        temporal_matrix_replay_exact=True,test_labels_never_used=True,all_context_rows_retained=True))
    verify_sources(); print('ALL REAL TEST FEATURE AUDITS PASSED',flush=True)


if __name__=='__main__': main()

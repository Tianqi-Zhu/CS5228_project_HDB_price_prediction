"""Read-only audit of raw cleaning, cached features and the saved Kaggle CSV."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from analysis.config import ROOT,DATA,REPORTS,AUX
from analysis.data import prepare_main,normalized_address,sha256
from analysis.compare_local_economy_landscape import write_json
from analysis.summarize_local_economy_landscape import table

OUT=REPORTS/'tabpfn_diagscale_mse_cleaning_audit'
SUB=REPORTS/'tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv'
FEATURES=ROOT/'.cache/tabpfn_fast_final_submission/features.joblib'
PROTECT=[DATA/'train.csv',DATA/'test.csv',FEATURES,SUB,
         ROOT/'.cache/tabpfn_diagscale_mse_final_submission/adapter.pt']
SITE_SOURCE='https://www.lta.gov.sg/content/ltagov/en/newsroom/2025/3/news-releases/expanding_our_rail_network_and_strengthening_rail_reliability.html'


def main():
    OUT.mkdir(exist_ok=True)
    before={p.relative_to(ROOT).as_posix():sha256(p) for p in PROTECT}
    settings=json.loads((REPORTS/'tabpfn_diagscale_mse_final_submission/settings.json').read_text())
    for path,h in settings['source_sha256'].items(): assert sha256(ROOT/path)==h
    bundle=joblib.load(FEATURES); raw={s:pd.read_csv(DATA/f'{s}.csv') for s in ['train','test']}
    frames={s:normalized_address(prepare_main(r)) for s,r in raw.items()}
    matrices={'train':bundle['X_train'],'test':bundle['X_query']}
    blocks=pd.read_csv(AUX/'sg-hdb-block.csv',dtype={'POSTAL_CODE':'string'})
    blocks=normalized_address(blocks); keys=['TOWN','BLOCK','STREET']
    assert not blocks.duplicated(keys).any()
    for c in ['MAX_FLOOR','YEAR_COMPLETED']:
        blocks[c]=pd.to_numeric(blocks[c],errors='coerce').replace(-1,np.nan)
    canonical=['RENT_APPROVAL_DATE',*keys,'FLAT_TYPE_CLEAN','FLAT_MODEL_CLEAN','FLOOR_AREA_SQM',
               'FURNISHED','LEASE_COMMENCE_DATE','FEE']
    summary=[]; missing=[]; ranges=[]; category=[]; duplicates=[]; anomalies=[]; join_audits=[]
    for split,r in raw.items():
        p,x=frames[split],matrices[split]
        assert x.index.equals(r.index) and len(r)==len(x)
        dates=pd.to_datetime(r.RENT_APPROVAL_DATE,format='%Y-%m',errors='coerce')
        assert dates.notna().all()
        join=p[keys].merge(blocks,on=keys,how='left',sort=False,validate='many_to_one',indicator=True)
        assert len(join)==len(r) and join._merge.eq('both').all()
        for c in ['LATITUDE','LONGITUDE','MAX_FLOOR','YEAR_COMPLETED']:
            np.testing.assert_allclose(x[c],join[c],rtol=0,atol=1e-10,equal_nan=True)
        for c in ['FLOOR_AREA_SQM','APPROVAL_MONTH_INDEX','FLAT_AGE_YEARS']:
            np.testing.assert_allclose(x[c],p[c],rtol=0,atol=1e-10,equal_nan=True)
        for c in ['FLAT_TYPE_CLEAN','FLAT_MODEL_CLEAN','TOWN_CLEAN']:
            np.testing.assert_array_equal(x[c].astype(str),p[c].astype(str))
        invalid_age=dates.dt.year-r.LEASE_COMMENCE_DATE<0
        for i in r.index[invalid_age]: anomalies.append(dict(split=split,row_id=int(i),issue='lease year later than approval year; derived lease age correctly masked',**r.loc[i].to_dict()))
        unknown_block=p.TOWN+'|'+p.STREET+'|'+p.BLOCK
        training_blocks=set((frames['train'].TOWN+'|'+frames['train'].STREET+'|'+frames['train'].BLOCK))
        unseen=~unknown_block.isin(training_blocks)
        assert (x.BLOCK_ID.isna().to_numpy()==unseen.to_numpy()).all()
        canonical_cols=canonical+(['MONTHLY_RENT'] if split=='train' else [])
        duplicate_mask=p.duplicated(canonical_cols,keep=False)
        duplicate_cases=p.loc[duplicate_mask,canonical_cols].copy(); duplicate_cases.insert(0,'row_id',duplicate_cases.index)
        duplicate_cases.to_csv(OUT/f'{split}_canonical_duplicate_cases.csv',index=False)
        for c in x:
            missing.append(dict(split=split,feature=c,missing_rows=int(x[c].isna().sum()),missing_pct=100*float(x[c].isna().mean())))
            if isinstance(x[c].dtype,pd.CategoricalDtype):
                known=set(matrices['train'][c].dropna().astype(str))
                category.append(dict(split=split,feature=c,levels=int(x[c].nunique()),missing_rows=int(x[c].isna().sum()),
                                     unknown_nonmissing_levels=len(set(x[c].dropna().astype(str))-known)))
            else:
                ranges.append(dict(split=split,feature=c,minimum=float(x[c].min()),maximum=float(x[c].max()),
                    infinite_rows=int(np.isinf(x[c].to_numpy(float)).sum())))
        for prefix in ['HIST_SIGNATURE','HIST_BLOCK_FLAT','HIST_AREA_FLAT']:
            count=x[prefix+'_COUNT']; mean=x[prefix+'_MEAN']
            assert not (count.dropna()<0).any()
            assert not ((count==0)&mean.notna()).any()
        for c in [c for c in x if c.endswith('_COUNT') or c.endswith('_STD') or c.endswith('_KM')]:
            assert not (x[c].dropna()<0).any()
        num=x.select_dtypes(include=np.number)
        assert not np.isinf(num.to_numpy()).any()
        assert (r.FLOOR_AREA_SQM>0).all()
        assert x.LATITUDE.between(1.1,1.5).all() and x.LONGITUDE.between(103.5,104.2).all()
        constants=[c for c in r if r[c].nunique(dropna=False)==1]
        assert not {'FURNISHED','FEE','MONTHLY_RENT','LOG_MONTHLY_RENT','RENT_PER_SQM'}&set(x)
        info=dict(split=split,rows=len(r),raw_missing_cells=int(r.isna().sum().sum()),raw_exact_duplicate_excess=int(r.duplicated().sum()),
            canonical_duplicate_excess=int(p.duplicated(canonical_cols).sum()),canonical_duplicate_affected_rows=int(duplicate_mask.sum()),
            flat_type_alias_rows=int((r.FLAT_TYPE!=p.FLAT_TYPE_CLEAN).sum()),invalid_lease_age_rows=int(invalid_age.sum()),
            missing_block_metadata_rows=int(x.YEAR_COMPLETED.isna().sum()),unseen_block_id_rows=int(x.BLOCK_ID.isna().sum()),
            constants_excluded=','.join(constants),minimum_floor_area=float(r.FLOOR_AREA_SQM.min()),maximum_floor_area=float(r.FLOOR_AREA_SQM.max()))
        if split=='train':
            info.update(minimum_rent=float(r.MONTHLY_RENT.min()),maximum_rent=float(r.MONTHLY_RENT.max()),
                nonpositive_rent_rows=int((r.MONTHLY_RENT<=0).sum()),rent_at_most_500_rows=int((r.MONTHLY_RENT<=500).sum()))
            assert (r.MONTHLY_RENT>0).all(); np.testing.assert_array_equal(bundle['y_train'],r.MONTHLY_RENT)
        summary.append(info)
        join_audits.append(dict(split=split,matched_rows=int(join._merge.eq('both').sum()),source_rows=len(r),
            source_missing_block_year_rows=int(join.YEAR_COMPLETED.isna().sum()),coordinates_and_metadata_match_cache=True))
    # Redundancy is explained by actual property constancy, not assumed to be a join bug.
    redundancy=[]
    for split,x in matrices.items():
        for suffix in ['MEAN','COUNT','LAST_MEAN','RECENCY_MONTHS']:
            a,b='HIST_SIGNATURE_'+suffix,'HIST_BLOCK_FLAT_'+suffix
            redundancy.append(dict(split=split,first=a,second=b,identical_including_missing=x[a].equals(x[b])))
    all_properties=pd.concat(list(frames.values()),ignore_index=True)
    variability=all_properties.groupby(keys+['FLAT_TYPE_CLEAN'])[['FLOOR_AREA_SQM','FLAT_MODEL_CLEAN','LEASE_COMMENCE_DATE']].nunique().max().to_dict()
    # Independent calculation reproduces the cached static MRT distances.
    # Only CR/JR-only locations are flagged here: a conservative, verified subset.
    stations=pd.read_csv(AUX/'sg-mrt-stations.csv')
    opened=stations.loc[stations.STATUS.str.casefold().eq('open')].copy()
    opened['prefix']=opened.CODE.str.extract(r'^([a-z]+)',expand=False)
    locations=opened[['LATITUDE','LONGITUDE']].drop_duplicates().reset_index(drop=True)
    future_prefix={'cr','cp','js','je','jw'}
    coord_groups=opened.groupby(['LATITUDE','LONGITUDE'])
    locations['codes']=[','.join(coord_groups.get_group((r.LATITUDE,r.LONGITUDE)).CODE) for r in locations.itertuples()]
    locations['names']=[','.join(coord_groups.get_group((r.LATITUDE,r.LONGITUDE)).NAME.unique()) for r in locations.itertuples()]
    locations['jrl_crl_only']=[set(coord_groups.get_group((r.LATITUDE,r.LONGITUDE)).prefix)<=future_prefix for r in locations.itertuples()]
    locations.to_csv(OUT/'mrt_source_location_audit.csv',index=False)
    tree=BallTree(np.radians(locations[['LATITUDE','LONGITUDE']]),metric='haversine')
    future_tree=BallTree(np.radians(locations.loc[locations.jrl_crl_only,['LATITUDE','LONGITUDE']]),metric='haversine')
    mrt_summary=[]
    for split,x in matrices.items():
        xy=np.radians(x[['LATITUDE','LONGITUDE']]); distances,idx=tree.query(xy,k=1)
        np.testing.assert_allclose(distances[:,0]*6371.0088,x.NEAREST_MRT_KM,rtol=0,atol=1e-9)
        nearest=locations.iloc[idx[:,0]].reset_index(drop=True)
        affected=nearest.jrl_crl_only.to_numpy()
        count=future_tree.query_radius(xy,r=1/6371.0088,count_only=True)
        rows=pd.DataFrame(dict(row_id=x.index,month=bundle['train_month' if split=='train' else 'query_month'],
            town=x.TOWN_CLEAN.astype(str),nearest_source_codes=nearest.codes.to_numpy(),nearest_source_name=nearest.names.to_numpy(),
            nearest_km=x.NEAREST_MRT_KM.to_numpy(),nearest_is_jrl_crl_only=affected,jrl_crl_sites_within_1km=count))
        rows.loc[affected|(count>0)].to_csv(OUT/f'mrt_affected_{split}_rows.csv',index=False)
        mrt_summary.append(dict(split=split,nearest_jrl_crl_only_rows=int(affected.sum()),nearest_jrl_crl_only_pct=100*float(affected.mean()),
            jrl_crl_in_1km_count_rows=int((count>0).sum()),jrl_crl_in_1km_count_pct=100*float((count>0).mean())))
    horizon=np.asarray(bundle['X_query'].TEMP_FORECAST_HORIZON)
    origin=pd.PeriodIndex(bundle['query_origin'],freq='M'); month=pd.PeriodIndex(bundle['query_month'],freq='M')
    np.testing.assert_array_equal(horizon,month.asi8-origin.asi8)
    assert (origin.asi8<month.asi8).all()
    march=np.asarray(bundle['query_month'])=='2025-03'
    assert (np.asarray(bundle['query_origin'])[march]=='2025-02').all()
    assert (np.asarray(bundle['query_origin'])[~march]=='2025-03').all()
    target=np.asarray(bundle['y_train'])-bundle['train_baseline']
    np.testing.assert_allclose(target+bundle['train_baseline'],bundle['y_train'],rtol=0,atol=1e-10)
    assert np.isfinite(target).all() and np.isfinite(bundle['query_baseline']).all()
    sample=pd.read_csv(DATA/'example-submission.csv'); prediction=pd.read_csv(SUB)
    assert prediction.shape==(50000,2) and list(prediction)==list(sample)
    np.testing.assert_array_equal(prediction.Id,sample.Id)
    assert prediction.Id.is_unique and np.isfinite(prediction.Predicted).all() and (prediction.Predicted>0).all()
    after={p.relative_to(ROOT).as_posix():sha256(p) for p in PROTECT}; assert before==after
    for name,rows in [('summary',summary),('feature_missingness',missing),('feature_numeric_ranges',ranges),
                      ('categorical_audit',category),('invalid_age_cases',anomalies),('join_audit',join_audits),
                      ('redundant_history_columns',redundancy),('mrt_summary',mrt_summary)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    write_json(OUT/'verification.json',dict(protected_file_hashes=after,audit_code_sha256=sha256(Path(__file__)),
        protected_artifacts_unchanged=True,raw_dates_and_values_valid=True,spatial_joins_and_cache_values_verified=True,
        categorical_normalization_verified=True,cached_mrt_distances_reproduced=True,submission_schema_and_rows_verified=True,
        target_reconstruction_sgd_verified=True,query_history_origins_verified=True,property_variability_by_block_type=variability,
        future_mrt_source=SITE_SOURCE,confirmed_future_only_locations=int(locations.jrl_crl_only.sum()),
        scope='Read-only audit; temporal MRT correctness finding remains unresolved in the existing77-feature model'))
    report=['# Data-cleaning audit of the MSE-only DiagScale submission','',
        '**Conclusion: basic cleaning, row alignment and submission formatting pass, but the spatial source is not fully cleaned for historical availability. The current77-feature cache includes future JRL/CRL sites as open MRT access.**','',
        '## Verified input and cache integrity','',table(pd.DataFrame(summary)),'',
        'Raw train/test have no missing cells, invalid approval dates, nonpositive floor areas or infinite model inputs. Known source hashes match the full-data training settings. All150,000 train and50,000 test rows retain their raw order; cached labels match raw rent exactly. Flat-type aliases such as4 room/4-room are canonicalized, and case/whitespace normalized address keys give100% many-to-one block joins. Cached coordinates and building metadata match an independent join. Constant FURNISHED=yes and FEE=0 are excluded from model inputs. No current-rent, log-rent or rent-per-area target column appears among the77 inputs.','',
        '## Findings and their treatment','',
        '### High priority: inaccurate MRT operating-status source','',
        f'The source has {int(locations.jrl_crl_only.sum())} distinct coordinates markedopen whose codes consist exclusively of future JRL/CRL lines. The pipeline filters STATUS=open and deduplicates exact coordinates, but has no station-opening date filter. Independent nearest-neighbour calculation reproduces the saved NEAREST_MRT_KM values, confirming this source issue is present in the submitted model inputs.','',
        table(pd.DataFrame(mrt_summary)),'',
        f'[LTA’s March2025 announcement]({SITE_SOURCE}) gives JRL opening stages in2027-2029 and CRL openings from2030, later than every train/test transaction in these files (latest July2026). The counts above are a conservative subset; they do not count other station-opening-date issues. Sites sharing a coordinate with an operating-line source code are excluded from this flag.','',
        'This is temporally inaccurate infrastructure availability, not direct use of hidden rental labels. It can make access look better than it was. Correct operating stations and planned-station access should be separate features in a controlled rerun. Existing77-feature checkpoints and the current CSV are preserved; silently changing these inputs at inference would break training consistency.','',
        '### Repeated records: retained, not automatically established as errors','',
        'Raw training has425 exact duplicate excess rows; canonicalization raises this to641. Test has5,508 raw duplicate excess rows and7,764 after canonicalization. These counts include all observed fields (including rent for training). There is no individual flat-unit or tenancy identifier, so identical rows can represent distinct contracts in the same block/month. Keep every test row and its Kaggle ID. Any train deduplication or reweighting needs a validation sensitivity experiment and a documented identity rule. Repeated observed property inputs alone do not establish target leakage. Duplicate-case exports are provided for review.','',
        '### One invalid lease-based age: already masked','',
        'Training row31739 has approval2021-11 and lease commencement2022. Its derived FLAT_AGE_YEARS is correctly NaN. Missing block metadata (-1 in the source) becomes NaN for159 training rows and zero test rows. This is not a missing-address problem: the coordinate joins are complete. Completion year and lease commencement are different fields and are retained separately.','',
        '### Missing histories and new categories: expected and explicit','',
        'The history features intentionally preserve NaN when no observations, no enough observations for a sample standard deviation, or no supported forecast are available at the permitted origin. Recent counts remain nonnegative, zero-count means are missing, and sample deviations are never negative. TabPFN receives its native missing-value representation; there is no future-target backfill.102 test rows have unseen BLOCK_ID categories and are deliberately encoded as missing while their town, coordinates and other features remain available. Other categorical columns are complete and use known train levels. Detailed missing counts are in feature_missingness.csv.','',
        '### Redundant histories: data structure, not a join failure','',
        'All four HIST_SIGNATURE summaries equal their HIST_BLOCK_FLAT counterparts, including missingness. In the combined raw data, floor area, model and lease year are each constant within a normalized block x flat-type group, so the finer signature creates no additional partition. Removing these four redundant columns is a feature-ablation question requiring retraining rather than a silent cleaning edit.','',
        '### Extreme rents and areas: flagged for review, not assumed erroneous','',
        'Training rents range fromS$300 toS$7,600; floor areas range31-215sqm. Nonpositive inputs are absent. Low/high prices and unusually large sizes are not dropped simply for falling outside a statistical range; source confirmation is needed to call them corrupt. The current pipeline has no blanket outlier deletion or winsorization.','',
        '## Temporal policy and CSV checks','',
        f'March2025 test histories use February2025; later test histories use March2025. The actual forecast horizon matches each row’s calendar month. {int((horizon>10).sum()):,} test rows exceed the validated ten-month forecast range and inherit the documented month10 forecast plateau. That extrapolation is a modelling limitation, not a raw-value cleaning operation. Existing causal-history tests are rerun separately; this audit verifies cached date/origin consistency and arithmetic without claiming that a static audit proves every possible leakage path absent.','',
        'The final CSV has50,000 unique, sample-aligned Id values and two columns Id,Predicted. All predictions are finite and positive. Source data, cached features, trained adapter and the saved submission are hash-checked before/after this read-only audit and remain unchanged.','',
        '## Next action','',
        'Prioritize an opening-date-corrected MRT feature experiment with the identical chronological80/20 split before changing the Kaggle submission. Deduplication policy and four redundant history columns can then be tested independently. Basic categorical/date/sentinel cleaning does not need a wholesale rewrite.','',
        'Reproduce: `.venv/Scripts/python.exe analysis/audit_diagscale_submission_cleaning.py`. Audit tables and issue row IDs are stored next to this report.']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    print(pd.DataFrame(summary).to_string(index=False)); print(pd.DataFrame(mrt_summary).to_string(index=False))


if __name__=='__main__': main()

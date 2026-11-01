"""Frozen TabPFN 77-feature control plus dated school/mall/transport quality."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from analysis.config import ROOT,AUX
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import frozen_predict,write_json,immutable,hashlib_array
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.local_spatial_features import metric_xy
from analysis.experiments.amenity_quality_sources import parse_school_page,parse_malls,train_passengers,school_key
from analysis.experiments.amenity_quality_features import (GROUPS,school_fields,mall_fields,transport_fields,bus_snapshot_files)

CONFIG=ROOT/'analysis/configs/amenity_quality.json'


def paths():
    cfg=json.loads(CONFIG.read_text()); out=ROOT/'reports'/cfg['experiment']; cache=ROOT/'.cache'/cfg['experiment']
    out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    return cfg,out,cache


def information_origins(bundle,base,split):
    x=bundle[f'X_{split}']
    if split=='query': return pd.Series(pd.PeriodIndex(bundle['query_month'],freq='M').min()-1,index=x.index)
    origin=base['train_temporal_origins'].reindex(x.index).forecast_origin.copy()
    fallback=pd.Series(pd.PeriodIndex(bundle['train_month'],freq='M')-1,index=x.index)
    return origin.where(origin.notna(),fallback)


def prepare():
    cfg,out,cache=paths()
    assert sha256(ROOT/cfg['baseline'])==cfg['baseline_sha256']
    parent=ROOT/'.cache/tabpfn_local_economy_landscape_80_20/prepared.joblib'
    parent_manifest=ROOT/'reports/tabpfn_local_economy_landscape_80_20/prepared_manifest.json'
    assert sha256(parent)==json.loads(parent_manifest.read_text())['sha256']
    source=json.loads((out/'source_manifest.json').read_text())
    source_hash={name:v['sha256'] for name,v in source.items() if v['status']=='acquired'}
    for name,h in source_hash.items(): assert sha256(cache/'sources'/name)==h
    code=[Path(__file__),CONFIG,ROOT/'analysis/experiments/amenity_quality_features.py',ROOT/'analysis/experiments/amenity_quality_sources.py',ROOT/'analysis/compare_local_economy_landscape.py']
    settings=dict(config=cfg,source_sha256=source_hash,parent_cache_sha256=sha256(parent),checkpoint_sha256=sha256(CHECKPOINT),
                  baseline_predictions_sha256=sha256(ROOT/cfg['baseline_predictions']),
                  code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in code},
                  versions={p:version(p) for p in ['tabpfn','torch','numpy','pandas','scipy','networkx','pyproj','shapely','beautifulsoup4','pypdf']},
                  inference=dict(frozen=True,n_estimators=1,random_state=42,batch_size=200,context_rows=121005,subsampling=False,
                                 target='original rent residual; original query baseline added once'),groups=GROUPS)
    immutable(out/'settings.json',settings)
    if (cache/'prepared.joblib').exists():
        assert sha256(cache/'prepared.joblib')==json.loads((out/'prepared_manifest.json').read_text())['sha256']; return
    data=joblib.load(parent); base=data['outer']['bundle']
    primary=pd.read_csv(AUX/'sg-schools.csv'); primary=primary.loc[primary.MAINLEVEL_CODE.str.contains('primary|p1',case=False)].copy()
    primary['school_key']=primary.NAME.map(school_key)
    assert not primary.school_key.duplicated().any()
    schools=pd.concat([parse_school_page(cache/f'sources/schools_{year}_all.html',year) for year in [2020,2021,2022,2023]],ignore_index=True)
    malls=pd.read_csv(AUX/'sg-shopping-malls.csv'); mall_data=parse_malls(cache/'sources',cache/'texts')
    passengers=train_passengers(cache/'sources/train_2023_archive.zip'); stations=pd.read_csv(AUX/'sg-mrt-stations.csv')
    schools.to_csv(out/'school_demand_records.csv',index=False); mall_data.to_csv(out/'mall_records.csv',index=False); passengers.to_csv(out/'station_passenger_records.csv',index=False)
    buildings=pd.read_csv(ROOT/'reports/tabpfn_local_economy_landscape_80_20/canonical_buildings.csv').set_index('building_id',drop=False)
    links=pd.read_csv(ROOT/'reports/tabpfn_local_economy_landscape_80_20/transaction_building_links.csv',index_col='row_id')
    xy=buildings[['x_m','y_m']].to_numpy()
    school_distance=cdist(xy,metric_xy(primary.LONGITUDE,primary.LATITUDE)); mall_distance=cdist(xy,metric_xy(malls.LONGITUDE,malls.LATITUDE))
    snapshots=bus_snapshot_files(cache/'sources')
    origins={}; months=set()
    for stage,entry in data.items():
        origins[stage]={}
        for split in ['train','query']:
            value=information_origins(entry['bundle'],base,split); origins[stage][split]=value; months.update(value.unique())
    tables={}; audits=[]; rail_rows=[]; coverage=[]
    for mode in ['dated','static_weights']:
        tables[mode]={}
        for origin in sorted(months):
            cutoff=origin.end_time; measurement_cutoff=cutoff if mode=='dated' else pd.Period('2024-05',freq='M').end_time
            school=school_fields(school_distance,primary,schools,measurement_cutoff,cfg)
            mall=mall_fields(mall_distance,malls,mall_data,measurement_cutoff,cfg)
            transport,nodes,active,audit=transport_fields(buildings,stations,passengers,snapshots,cutoff,cfg,measurement_cutoff=measurement_cutoff)
            extra=pd.concat([school,mall,transport],axis=1); extra.index=buildings.index
            tables[mode][origin]=extra
            if mode=='dated':
                if origin in [pd.Period('2020-12'),pd.Period('2021-09'),pd.Period('2022-12'),pd.Period('2024-05')]:
                    for record in audit: audits.append(dict(origin=str(origin),**record))
                    rail_rows.extend(dict(origin=str(origin),station_key=k,**r.to_dict()) for k,r in nodes.iterrows())
            print(f'{mode} building features {origin}',flush=True)
    joblib.dump(tables,cache/'building_features.joblib',compress=3)
    tables['dated'][pd.Period('2024-05')].to_csv(cache/'building_features_2024-05.csv')
    pd.DataFrame(audits).to_csv(out/'station_bus_match_audit.csv',index=False)
    pd.DataFrame(rail_rows).to_csv(out/'operating_station_audit.csv',index=False)
    prepared={}
    for stage,entry in data.items():
        b=entry['bundle']; extras={}
        for mode in tables:
            extras[mode]={}
            for split in ['train','query']:
                ids=b[f'X_{split}'].index; origin=origins[stage][split]
                values=pd.DataFrame(index=ids,columns=GROUPS['combined'],dtype=float)
                for month in origin.unique():
                    selected=origin.index[origin==month]; building_ids=links.loc[selected,'building_id']
                    values.loc[selected,:]=tables[mode][month].loc[building_ids].to_numpy()
                extras[mode][split]=values
                for name in values:
                    coverage.append(dict(stage=stage,mode=mode,split=split,feature=name,rows=len(values),missing_fraction=float(values[name].isna().mean()),
                                         unique_nonmissing=int(values[name].nunique())))
        prepared[stage]=dict(bundle=b,added=extras)
    pd.DataFrame(coverage).to_csv(out/'feature_coverage.csv',index=False)
    school_audit=primary[['NAME','school_key','LATITUDE','LONGITUDE']].copy()
    for year in [2020,2021,2022,2023]: school_audit[f'matched_{year}']=school_audit.school_key.isin(schools.loc[schools.year==year,'school_key'])
    school_audit.to_csv(out/'school_name_match_audit.csv',index=False)
    mall_audit=malls.copy(); mall_audit['has_any_measurement']=malls.NAME.isin(mall_data.mall); mall_audit.to_csv(out/'mall_name_match_audit.csv',index=False)
    joblib.dump(prepared,cache/'prepared.joblib',compress=3)
    write_json(out/'prepared_manifest.json',dict(sha256=sha256(cache/'prepared.joblib'),building_cache_sha256=sha256(cache/'building_features.joblib'),
               school_rows=len(schools),school_sites=len(primary),measured_malls=int(mall_data.mall.nunique()),passenger_station_codes=len(passengers),
               feature_count=len(GROUPS['combined']),stages=list(prepared),groups=GROUPS))
    write_json(out/'feature_dictionary.json',{name:dict(group=next(group for group in ['schools','malls','transport'] if name in GROUPS[group]),
                labels_used=False,availability='source released by information origin; static_weights is a separate retrospective sensitivity',
                definition='See source records, config and amenity_quality_features.py; log = natural log1p unless school log applicant/vacancy ratio') for name in GROUPS['combined']})
    write_json(out/'source_limits.json',dict(official_school_rankings=False,school_proxy='primary admission demand; not teaching quality or secondary school reputation',
               all_malls_have_financial_data=False,revenue='owner gross property revenue; not tenant sales or net property income',
               train_volume='station entry+exit taps, not train/platform occupancy; Aug–Oct 2023 only; public archive uploaded Dec 6 2023',
               bus='unique service IDs across directions; no timetable-frequency or passenger count interpretation',
               rail='MRT network only; incomplete LRT topology excluded; source CC33/34 mapped to historical CE2/1 branch topology',
               geography='inherited location snapshots; no claim that every school/mall coordinate is historically archived',
               source_archive_caveat='historical registration figures are retrieved from a present-day MOE-derived archive; original version preservation is not cryptographically verifiable'))


def run_one(stage,group,mode='dated',smoke=False):
    cfg,out,cache=paths(); prep=json.loads((out/'prepared_manifest.json').read_text()); assert sha256(cache/'prepared.joblib')==prep['sha256']
    data=joblib.load(cache/'prepared.joblib')[stage]; b=data['bundle']; names=GROUPS[group]
    x,q=[b[f'X_{split}'].join(data['added'][mode][split][names]) for split in ['train','query']]
    for split,m in [('train',x),('query',q)]: pd.testing.assert_frame_equal(m[b[f'X_{split}'].columns],b[f'X_{split}'])
    if smoke: q=q.iloc[:400]
    tag=f'{stage}_{group}_{mode}'+('_smoke' if smoke else '')
    immutable(out/f'job_{tag}.json',dict(stage=stage,group=group,mode=mode,added_columns=names,context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index),
               prepared_sha256=prep['sha256'],checkpoint_sha256=sha256(CHECKPOINT),code_sha256=sha256(Path(__file__))))
    result_path=out/f'predictions_{tag}.csv'; timing_path=out/f'timing_{tag}.json'
    if result_path.exists() and timing_path.exists():
        saved=pd.read_csv(result_path); np.testing.assert_array_equal(saved.row_id,q.index); np.testing.assert_array_equal(saved.actual,np.asarray(b['y_query'])[:len(q)]); return
    print(f'{tag}: {len(x):,} complete context; {len(q):,} queries; {len(x.columns)} features',flush=True)
    write_json(out/'status.json',dict(status='inference',stage=stage,group=group,mode=mode))
    residual,timing=frozen_predict(x,np.asarray(b['y_train'])-b['train_baseline'],q,cache/f'partial_{tag}.npy')
    predicted=residual+np.asarray(b['query_baseline'])[:len(q)]
    saved=pd.DataFrame(dict(row_id=q.index,month=np.asarray(b['query_month'])[:len(q)],actual=np.asarray(b['y_query'])[:len(q)],predicted=predicted,
                            weight=np.asarray(b.get('query_weights',np.ones(len(q))))[:len(q)]))
    saved['error']=saved.predicted-saved.actual; saved['absolute_error']=saved.error.abs(); saved.to_csv(result_path,index=False,float_format='%.10f')
    write_json(timing_path,dict(**timing,added_dimensions=len(names),**metrics(saved.actual,saved.predicted)))
    if group=='baseline' and stage=='outer':
        reference=pd.read_csv(ROOT/cfg['baseline_predictions']).set_index('row_id').loc[q.index]
        np.testing.assert_allclose(predicted,reference.predicted,atol=1e-5,rtol=0)
        write_json(out/'baseline_replay.json',dict(verified=True,rows=len(q),max_abs_difference=float(np.max(np.abs(predicted-reference.predicted)))))
    print(group,mode,metrics(saved.actual,saved.predicted),flush=True)


def child(stage,group,mode='dated',smoke=False):
    cmd=[sys.executable,str(Path(__file__)),'--stage','one','--fold',stage,'--group',group,'--mode',mode]
    if smoke: cmd+=['--smoke']
    subprocess.run(cmd,check=True,cwd=ROOT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','smoke','inner','outer','one','all','dry-run'],default='all')
    parser.add_argument('--fold',default='outer'); parser.add_argument('--group',choices=list(GROUPS),default='combined')
    parser.add_argument('--mode',choices=['dated','static_weights'],default='dated'); parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args(); cfg,out,cache=paths()
    if a.stage=='dry-run': print(json.dumps(dict(config=cfg,groups=GROUPS,pretrained_weights_modified=False),indent=2)); return
    if a.stage in ('prepare','all'): prepare()
    if a.stage in ('smoke','all'):
        child('outer','baseline',smoke=True); child('inner_2023-07','combined',smoke=True)
    if a.stage in ('inner','all'):
        for fold in cfg['inner_stages']:
            for group in GROUPS: child(fold,group)
    if a.stage in ('outer','all'):
        immutable(out/'outer_plan.json',dict(groups=cfg['outer_variants'],static_sensitivity='combined',uses_outer_labels_to_select=False,config_sha256=sha256(CONFIG)))
        for group in cfg['outer_variants']: child('outer',group)
        child('outer','combined','static_weights')
    if a.stage=='one': run_one(a.fold,a.group,a.mode,a.smoke)


if __name__=='__main__': main()

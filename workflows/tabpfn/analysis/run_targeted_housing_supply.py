"""Archived annual-report housing supply ablation, with explicit availability."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fitz
import joblib
import numpy as np
import pandas as pd
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable,write_json
import analysis.run_targeted_spatial_economic as core

TOWNS=['ang mo kio','bedok','bishan','bukit batok','bukit merah','bukit panjang','choa chu kang','clementi',
       'geylang','hougang','jurong east','jurong west','kallang/whampoa','pasir ris','punggol','queenstown',
       'sembawang','sengkang','serangoon','tampines','tengah','toa payoh','woodlands','yishun','central area','bukit timah','marine parade']
SOURCES=core.OUT/'sources'
DOCUMENTS=[
    dict(year=2019,file='hdb2019.pdf',available='2020-01',pages=[15,16],
         url='https://links.sgx.com/FileOpen/HDB_v19_keystatistic_20191024.ashx?AnncID=IWPHODMKZ4RZEZAF&App=ArchiveAnnouncement&FileID=583813',
         timing='Archived October 2019 issuer filing; conservative January 2020 use'),
    dict(year=2021,file='hdb2021.pdf',available='2023-01',pages=[8],
         url='https://links.sgx.com/1.0.0/corporate-announcements/CXHXEE7KATRRGPM3/738439_HDB_key-statistics.pdf',
         timing='Retrieved issuer filing broadcast 11 November 2022; do not assume the 2021 financial period implies availability in 2021'),
    dict(year=2022,file='verified_key_statistics_2022.pdf',available='2023-01',pages=[8],
         url='https://assets.hdb.gov.sg/about-us/news-and-publications/annual-report/2022/ebooks/Key%20Statistics%20FY21.pdf',
         timing='HDB FY2021 annual-report release 30 October 2022; conservative January 2023 use'),
    dict(year=2023,file='hdb2023_new.pdf',available='2024-01',pages=[8],
         url='https://www.hdb.gov.sg/-/media/hdb-pulse/reports/annual-reports-and-financial-statements/HDB-Key-Statistics-FY22.pdf',
         timing='FY2022 report issued 31 October 2023; conservative January 2024 use, nine months after fiscal end'),
]
FEATURES=['SUPPLY_COMPLETIONS_STOCK_RATIO','SUPPLY_OBSERVATION_AGE_MONTHS','SUPPLY_UNAVAILABLE']


def numeric_rows(page,n,minimum_x=0):
    words=sorted([w for w in page.get_text('words') if re.fullmatch(r'[\d,]+',w[4]) and w[0]>minimum_x],key=lambda w:w[1])
    groups=[]
    for w in words:
        if not groups or w[1]-groups[-1][0][1]>2:groups.append([w])
        else:groups[-1].append(w)
    return [[int(w[4].replace(',','')) for w in sorted(g,key=lambda w:w[0])] for g in groups if len(g)==n]


def extract_supply():
    records=[];audit=[]
    for spec in DOCUMENTS:
        path=SOURCES/spec['file'];doc=fitz.open(path)
        if spec['year']==2019:
            left,right=[numeric_rows(doc[p],9) for p in spec['pages']]
            assert len(left)==len(right)==28
            values=[a+b for a,b in zip(left,right)];stock_column=-4;completed_column=-3
        else:
            values=numeric_rows(doc[8],17,minimum_x=150);stock_column=-3;completed_column=-1
        assert len(values)==28,(spec['file'],len(values))
        extracted=[]
        for town,v in zip(TOWNS+['TOTAL'],values):
            row=dict(year=spec['year'],town=town,stock=v[stock_column],completed=v[completed_column],
                     available=spec['available'],period_end=f"{spec['year']}-03",source_file=spec['file'])
            extracted.append(row)
        assert sum(r['stock'] for r in extracted[:-1])==extracted[-1]['stock']
        assert sum(r['completed'] for r in extracted[:-1])==extracted[-1]['completed']
        # These independent published totals also detect column-order changes:
        # FY2018/19 prints completions BEFORE construction; later reports reverse it.
        assert extracted[-1]['completed']=={2019:15748,2021:8234,2022:13627,2023:23782}[spec['year']]
        audit.append(dict(**spec,sha256=sha256(path),towns=27,stock=extracted[-1]['stock'],completed=extracted[-1]['completed'],totals_reconcile=True,completion_column_verified_visually=True))
        records.extend(extracted[:-1])
    frame=pd.DataFrame(records)
    frame.to_csv(core.OUT/'housing_supply_vintages.csv',index=False)
    write_json(core.OUT/'housing_supply_sources.json',dict(documents=audit,
        publication_policy='Conservative explicit availability months, tested against each existing per-row feature origin minus one month. No observation is assigned to its fiscal end before publication.',
        snapshot_gaps='2019/20 and verified pre-November2022 circulation of FY2020/21 were unavailable; retain the older 2019 snapshot until a documented newer vintage is available, expose its age. No interpolation/backfill.',
        definition='Annual completed HDB residential units divided by total town HDB stock. Includes subsidized home-ownership/public-rental housing; NOT measured additions to open-market rental supply.',
        timing_sources=['https://www.hdb.gov.sg/hdb-pulse/news/2022/hdb-annual-report-fy2021',
                        'https://links.sgx.com/1.0.0/corporate-announcements/CXHXEE7KATRRGPM3/01b6f1d81396670e2a8585fec46d235ddaf4a927f5d8ef8915621d498e0e6ecc',
                        'https://www.channelnewsasia.com/watch/hdb-records-deficit-s538b-last-financial-year-video-3886826']))
    return frame


def supply_features(query,cutoffs,vintages):
    result=pd.DataFrame(np.nan,index=query.index,columns=FEATURES)
    cutoffs=pd.Series(pd.PeriodIndex(cutoffs,freq='M'),index=query.index)
    for cutoff in sorted(cutoffs.dropna().unique()):
        q=query.loc[cutoffs.eq(cutoff)]
        known=vintages[pd.PeriodIndex(vintages.available,freq='M')<=cutoff]
        known=known.sort_values('year').drop_duplicates('town',keep='last').set_index('town')
        matched=known.reindex(q.TOWN_CLEAN)
        stock,completed=matched.stock.to_numpy(float),matched.completed.to_numpy(float)
        value=np.divide(completed,stock,out=np.full(len(q),np.nan),where=stock>0)
        result.loc[q.index,FEATURES[0]]=value
        years=matched.year.to_numpy(float)
        result.loc[q.index,FEATURES[1]]=np.where(np.isfinite(years),cutoff.ordinal-(years-1970)*12-2,np.nan)
        result.loc[q.index,FEATURES[2]]=(~np.isfinite(value)).astype(float)
    result[FEATURES[2]]=result[FEATURES[2]].fillna(1.)
    return result


def check_sources():
    core.check_sources()
    settings=json.loads((core.OUT/'B1_settings.json').read_text())
    for key in ['source_sha256','code_sha256']:
        for p,h in settings[key].items():assert sha256(core.ROOT/p)==h,p


def prepare():
    core.check_sources();v=extract_supply();frame=core.raw_frame()
    immutable(core.OUT/'B1_settings.json',dict(variant='B1',features=FEATURES,documents=DOCUMENTS,
        source_sha256={(SOURCES/s['file']).relative_to(core.ROOT).as_posix():sha256(SOURCES/s['file']) for s in DOCUMENTS},
        code_sha256={Path(__file__).relative_to(core.ROOT).as_posix():sha256(Path(__file__))},
        inference='Same frozen architecture, unchanged original inputs/targets/contexts, separately reapply earlier-only ensemble weighting',
        no_early_backfill=True,no_new_submission=True))
    coverage=[]
    for origin in core.ORIGINS:
        for member in core.MEMBERS:
            path=core.CACHE/f'features_{origin}_B1_{member}.joblib'
            if path.exists():core.load_prepared(origin,f'B1_{member}');continue
            b=core.original_bundle(origin,member)
            for split in ('train','query'):
                x=b[f'X_{split}'];_,cutoffs=core.row_cutoffs(x)
                f=supply_features(frame.loc[x.index],cutoffs,v);z=x.join(f)
                z.columns=pd.Index(list(z.columns),dtype=x.columns.dtype)
                pd.testing.assert_frame_equal(z[x.columns],x)
                b[f'X_{split}']=z
                coverage.extend(dict(origin=origin,member=member,split=split,feature=c,rows=len(z),missing=int(z[c].isna().sum())) for c in f)
            joblib.dump(b,path,compress=3)
            write_json(core.OUT/f'features_{origin}_B1_{member}.json',dict(sha256=sha256(path),features=list(b['X_train']),
                context_rows=len(b['X_train']),query_rows=len(b['X_query']),original_columns_targets_baselines_unchanged=True))
        print('PREPARED B1',origin,flush=True)
    if coverage:pd.DataFrame(coverage).to_csv(core.OUT/'B1_feature_coverage.csv',index=False)


def worker(origin,member):
    prior=core.prior
    prior.OUT,prior.CACHE=core.OUT,core.CACHE
    prior.check_sources,prior.load_prepared=check_sources,core.load_prepared
    prior.infer(origin,member)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','prepare','one'],default='all');p.add_argument('--origin');p.add_argument('--member');a=p.parse_args()
    if a.stage=='one':worker(a.origin,a.member);return
    prepare()
    if a.stage=='prepare':return
    import subprocess,os
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for key in ['TABPFN_MAX_BATCHED_ESTIMATOR_ROWS','TABPFN_MAX_BATCHED_ESTIMATOR_CELLS']:env.pop(key,None)
    for origin in core.ORIGINS:
        for m in core.MEMBERS:
            member=f'B1_{m}'
            with (core.OUT/f'inference_{origin}_{member}.log').open('a',encoding='utf-8') as log:
                code=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','one','--origin',origin,'--member',member],cwd=core.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'Inference failed {origin} {member}')
            print('COMPLETED',origin,member,flush=True)


if __name__=='__main__':main()

"""Canonicalize published bus-stop abbreviations before transport inference.

The original prepared artifact and initial audit remain immutable. Only the
same-name service-count column changes; school, mall and geometric bus inputs
are asserted identical. Corrected model outputs use distinct filenames.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
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
from analysis.compare_amenity_quality import paths,information_origins,child
from analysis.compare_local_economy_landscape import frozen_predict,write_json,immutable,hashlib_array
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.amenity_quality_features import GROUPS,rail_graph,station_bus_counts,bus_snapshot_files
from analysis.experiments.local_spatial_features import metric_xy

UPDATED='AQ_MRT_NAMED_BUS_SERVICES_LOG'
EXPANSIONS={'nth':'north','sth':'south','upp':'upper','resvr':'reservoir','gdns':'gardens','gdn':'garden',
            'rd':'road','ave':'avenue','cres':'crescent','lor':'lorong','blvd':'boulevard','pl':'place','lk':'link','cir':'circle'}


def canonical_stop_name(name):
    name=re.sub(r"\bw['’]?lands\b",'woodlands',name,flags=re.I)
    name=re.sub(r"\bs['’]?goon\b",'serangoon',name,flags=re.I)
    name=re.sub(r"\bc['’]?wealth\b",'commonwealth',name,flags=re.I)
    name=re.sub(r'\b6th\b','sixth',name,flags=re.I)
    name=re.sub(r'\bholland\s+v\b','holland village',name,flags=re.I)
    for old,new in EXPANSIONS.items(): name=re.sub(r'\b'+old+r'\b',new,name,flags=re.I)
    return name


def prepare():
    cfg,out,cache=paths(); dest=cache/'prepared_bus_aliases.joblib'; manifest=out/'bus_name_refinement.json'
    if dest.exists():
        info=json.loads(manifest.read_text()); assert sha256(dest)==info['sha256']; assert sha256(Path(__file__))==info['code_sha256']; return
    parent_hash=sha256(cache/'prepared.joblib'); data=joblib.load(cache/'prepared.joblib'); base=data['outer']['bundle']
    buildings=pd.read_csv(ROOT/'reports/tabpfn_local_economy_landscape_80_20/canonical_buildings.csv').set_index('building_id')
    links=pd.read_csv(ROOT/'reports/tabpfn_local_economy_landscape_80_20/transaction_building_links.csv',index_col='row_id')
    stations=pd.read_csv(AUX/'sg-mrt-stations.csv'); snapshots=bus_snapshot_files(cache/'sources'); fixed=[]; changed=[]
    folder=cache/'canonical_bus_names'; folder.mkdir(exist_ok=True)
    for stamp,stop_path,services_path in snapshots:
        raw=json.loads(stop_path.read_text()); names={}
        for stop,values in raw.items():
            values=values.copy(); old=values[2]; values[2]=canonical_stop_name(old); names[stop]=values
            if old!=values[2]: changed.append(dict(snapshot=str(stamp),stop_id=stop,original_name=old,canonical_name=values[2]))
        path=folder/stop_path.name; path.write_text(json.dumps(names),encoding='utf8'); fixed.append((stamp,path,services_path))
    tables=joblib.load(cache/'building_features.joblib'); replacements={}; all_audit=[]; station_profiles=[]
    for mode,months in tables.items():
        replacements[mode]={}
        for origin,extra in months.items():
            cutoff=origin.end_time; source_cutoff=cutoff if mode=='dated' else pd.Period('2024-05','M').end_time
            nodes,_,_=rail_graph(stations,cutoff)
            available=[s for s in fixed if s[0]<=source_cutoff]; snapshot=available[-1] if available else None
            named,nearby,audit=station_bus_counts(nodes,snapshot,cfg)
            nearest=cdist(buildings[['x_m','y_m']],metric_xy(nodes.LONGITUDE,nodes.LATITUDE)).argmin(axis=1)
            np.testing.assert_allclose(np.log1p(nearby[nearest]),extra.AQ_MRT_NEARBY_BUS_SERVICES_LOG,equal_nan=True,atol=1e-12)
            replacements[mode][origin]=pd.Series(np.log1p(named[nearest]),index=buildings.index)
            if mode=='dated' and origin==pd.Period('2024-05'):
                all_audit.extend(audit)
                profile=nodes.copy(); profile['same_name_unique_bus_services']=named; profile['nearby_400m_unique_bus_services']=nearby
                station_profiles=profile
            print('Canonical bus names',mode,origin,flush=True)
    for stage,entry in data.items():
        for mode in entry['added']:
            for split,extra in entry['added'][mode].items():
                original=extra.copy(); origins=information_origins(entry['bundle'],base,split)
                for origin in origins.unique():
                    ids=origins.index[origins==origin]; sites=links.loc[ids,'building_id']
                    extra.loc[ids,UPDATED]=replacements[mode][origin].loc[sites].to_numpy()
                pd.testing.assert_frame_equal(extra.drop(columns=UPDATED),original.drop(columns=UPDATED))
    joblib.dump(data,dest,compress=3)
    pd.DataFrame(changed).to_csv(out/'bus_stop_abbreviation_audit.csv',index=False)
    pd.DataFrame(all_audit).to_csv(out/'station_bus_match_audit_canonical_2024-05.csv',index=False)
    station_profiles.to_csv(out/'station_weight_assignments_canonical_2024-05.csv')
    write_json(manifest,dict(sha256=sha256(dest),parent_prepared_sha256=parent_hash,code_sha256=sha256(Path(__file__)),updated_column=UPDATED,
             expansions=EXPANSIONS,extra_rules=["W'lands → Woodlands","S'goon → Serangoon","C'wealth → Commonwealth","6th → Sixth","Holland V → Holland Village"],
             nonupdated_columns_asserted_identical=True,target_labels_used=False,
             canonical_file_sha256={p.name:sha256(p) for p in folder.glob('*.json')}))


def run(stage,group,mode='dated'):
    cfg,out,cache=paths(); info=json.loads((out/'bus_name_refinement.json').read_text()); assert sha256(cache/'prepared_bus_aliases.joblib')==info['sha256']
    data=joblib.load(cache/'prepared_bus_aliases.joblib')[stage]; b=data['bundle']; names=GROUPS[group]
    x,q=[b[f'X_{split}'].join(data['added'][mode][split][names]) for split in ['train','query']]
    for split,m in [('train',x),('query',q)]: pd.testing.assert_frame_equal(m[b[f'X_{split}'].columns],b[f'X_{split}'])
    tag=f'{stage}_{group}_canonical_{mode}'; path=out/f'predictions_{tag}.csv'; timepath=out/f'timing_{tag}.json'
    immutable(out/f'job_{tag}.json',dict(stage=stage,group=group,mode=mode,prepared_sha256=info['sha256'],code_sha256=sha256(Path(__file__)),
              context_ids_sha256=hashlib_array(x.index),query_ids_sha256=hashlib_array(q.index),checkpoint_sha256=sha256(CHECKPOINT)))
    if path.exists() and timepath.exists(): return
    print(tag,len(x),'full context',len(q),'queries',len(x.columns),'features',flush=True)
    write_json(out/'status.json',dict(status='inference',stage=stage,group=group,mode=mode,name_matching='canonical public abbreviations'))
    residual,timing=frozen_predict(x,np.asarray(b['y_train'])-b['train_baseline'],q,cache/f'partial_{tag}.npy')
    prediction=residual+b['query_baseline']; result=pd.DataFrame(dict(row_id=q.index,month=b['query_month'],actual=b['y_query'],predicted=prediction,weight=b.get('query_weights',np.ones(len(q)))))
    result['error']=result.predicted-result.actual; result['absolute_error']=result.error.abs(); result.to_csv(path,index=False,float_format='%.10f')
    write_json(timepath,dict(**timing,added_dimensions=len(names),**metrics(result.actual,result.predicted)))
    print(tag,metrics(result.actual,result.predicted),flush=True)


def launch(stage,group,mode='dated'):
    subprocess.run([sys.executable,str(Path(__file__)),'--stage','one','--fold',stage,'--group',group,'--mode',mode],check=True,cwd=ROOT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=['prepare','inner','outer','one','all'],default='all')
    parser.add_argument('--fold',default='outer'); parser.add_argument('--group',choices=['transport','combined'],default='combined'); parser.add_argument('--mode',choices=['dated','static_weights'],default='dated')
    a=parser.parse_args(); cfg,out,cache=paths()
    if a.stage in ['prepare','all']: prepare()
    if a.stage in ['inner','all']:
        for fold in cfg['inner_stages']:
            for group in ['transport','combined']: launch(fold,group)
    if a.stage in ['outer','all']:
        # Identical school/mall feature blocks reuse their verified predictions.
        for group in ['schools','malls']: child('outer',group)
        for group in ['transport','combined']: launch('outer',group)
        launch('outer','combined','static_weights')
    if a.stage=='one': run(a.fold,a.group,a.mode)

"""Dated, capacity/demand-weighted amenity features for frozen TabPFN."""
from __future__ import annotations
import json
import re
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from analysis.experiments.local_spatial_features import metric_xy
from analysis.experiments.amenity_quality_sources import school_key

SCHOOL_COLUMNS=['AQ_SCHOOL_NEAREST_LOG_DEMAND','AQ_SCHOOL_MAX_PERCENTILE_1KM','AQ_SCHOOL_EXCESS_ACCESS_LOG',
                'AQ_SCHOOL_COVERAGE_1KM','AQ_SCHOOL_NEAREST_UNKNOWN']
MALL_COLUMNS=['AQ_MALL_NEAREST_LOG_NLA','AQ_MALL_NLA_ACCESS_LOG','AQ_MALL_GROSS_REVENUE_ACCESS_LOG',
              'AQ_MALL_FOOTFALL_ACCESS_LOG','AQ_MALL_COVERAGE_2KM','AQ_MALL_REVENUE_COVERAGE_2KM','AQ_MALL_NEAREST_RECORD_AGE_YEARS']
TRANSPORT_COLUMNS=['AQ_MRT_NEAREST_OPERATIONAL_KM','AQ_MRT_NEAREST_LINE_COUNT','AQ_MRT_NEAREST_HARMONIC_CENTRALITY',
                   'AQ_MRT_NEAREST_CBD_HOPS','AQ_MRT_NAMED_BUS_SERVICES_LOG','AQ_MRT_NEARBY_BUS_SERVICES_LOG',
                   'AQ_MRT_NEAREST_DAILY_TAPS_LOG','AQ_MRT_BUS_SOURCE_UNKNOWN','AQ_MRT_PASSENGER_SOURCE_UNKNOWN']
GROUPS={'baseline':[],'schools':SCHOOL_COLUMNS,'malls':MALL_COLUMNS,'transport':TRANSPORT_COLUMNS,
        'combined':SCHOOL_COLUMNS+MALL_COLUMNS+TRANSPORT_COLUMNS}


def historical_latest(records,key,cutoff):
    if records.empty: return records
    allowed=records.loc[pd.to_datetime(records.available_at)<=pd.Timestamp(cutoff)]
    return allowed.sort_values(['available_at','year']).drop_duplicates(key,keep='last')


def school_fields(distance,schools,records,cutoff,cfg):
    latest=historical_latest(records,'school_key',cutoff).set_index('school_key')
    values=schools.school_key.map(latest.log_demand).to_numpy(float)
    percentile=latest.log_demand.rank(method='average',pct=True)
    ranks=schools.school_key.map(percentile).to_numpy(float)
    near=distance.argmin(axis=1); within=distance<=cfg['school_radius_m']; known=np.isfinite(values)
    weights=np.exp(-distance/cfg['school_decay_m'])*within
    maximum=np.max(np.where(within & known[None,:],ranks[None,:],-np.inf),axis=1)
    maximum[~np.isfinite(maximum)]=np.nan
    total=within.sum(axis=1); covered=(within & known[None,:]).sum(axis=1)
    coverage=np.divide(covered,total,out=np.full(len(distance),np.nan),where=total>0)
    # Centred percentiles avoid arbitrary 'elite school' scores and retain a
    # separate source-coverage measurement; popularity is not teaching quality.
    access=weights@np.where(np.isfinite(ranks),np.maximum(ranks-.5,0)*2,0)
    access[(covered==0)&(total>0)]=np.nan
    return pd.DataFrame(np.column_stack([values[near],maximum,np.log1p(access),coverage,(~known[near]).astype(float)]),columns=SCHOOL_COLUMNS)


def mall_fields(distance,malls,records,cutoff,cfg):
    latest=historical_latest(records,'mall',cutoff).set_index('mall')
    near=distance.argmin(axis=1)
    def values(name): return malls.NAME.map(latest[name]).to_numpy(float)
    nla,rev,foot=values('nla_sqm'),values('revenue_sgd_m'),values('footfall_m')
    within=distance<=cfg['mall_radius_m']; known=np.isfinite(nla)
    count=within.sum(axis=1)
    coverage=np.divide((within & known[None,:]).sum(axis=1),count,out=np.full(len(distance),np.nan),where=count>0)
    weights=np.exp(-distance/cfg['mall_decay_m'])*within
    def weighted_access(v):
        mask=np.isfinite(v); support=(within & mask[None,:]).sum(axis=1)
        access=weights@np.where(mask,v,0); access[(support==0)&(count>0)]=np.nan
        return np.log1p(access)
    revenue_coverage=np.divide((within & np.isfinite(rev)[None,:]).sum(axis=1),count,out=np.full(len(distance),np.nan),where=count>0)
    dates=pd.to_datetime(malls.NAME.map(latest.available_at))
    age=(pd.Timestamp(cutoff)-dates).dt.days.to_numpy(float)/365.25
    return pd.DataFrame(np.column_stack([np.log1p(nla[near]),weighted_access(nla),weighted_access(rev),weighted_access(foot),coverage,revenue_coverage,age[near]]),columns=MALL_COLUMNS)


def station_key(name):
    return re.sub(r'[^a-z0-9]','',re.sub(r'\s+(mrt/lrt|mrt|lrt)\s+station$','',str(name).lower()))


def operational_date(code):
    code=code.lower()
    m=re.fullmatch(r'([a-z]+)(\d+)',code)
    if not m: return None
    prefix,n=m.group(1),int(m.group(2))
    if prefix=='ns': return None if n==6 else pd.Timestamp('2019-11-02' if n==12 else '2019-01-01')
    if prefix=='ew' or prefix=='cg': return pd.Timestamp('2019-01-01')
    if prefix=='ne': return pd.Timestamp('2025-03-16' if n==18 else '2019-01-01')
    if prefix=='cc': return None if n in (18,30,31,32) else pd.Timestamp('2019-01-01')
    if prefix=='dt': return pd.Timestamp('2025-03-07') if n==4 else (None if n>35 else pd.Timestamp('2019-01-01'))
    if prefix=='te':
        if n<=3: return pd.Timestamp('2020-01-31')
        if n<=9: return pd.Timestamp('2021-08-28')
        if n in (10,21) or n>29: return None
        if n<=22: return pd.Timestamp('2022-11-13')
        return pd.Timestamp('2024-06-23')
    return None  # Future lines, RTS, and incomplete LRT topology are excluded.


def rail_graph(stations,cutoff):
    frame=stations.copy(); frame['opened']=frame.CODE.map(operational_date)
    frame=frame.loc[frame.opened.notna() & (frame.opened<=pd.Timestamp(cutoff))].copy()
    frame['station_key']=frame.NAME.map(station_key)
    frame['line']=frame.CODE.str.extract(r'^([a-z]+)',expand=False)
    frame.loc[frame.line=='cg','line']='ew'
    nodes=frame.groupby('station_key').agg(NAME=('NAME','first'),LATITUDE=('LATITUDE','mean'),LONGITUDE=('LONGITUDE','mean'),line_count=('line','nunique'))
    graph=nx.Graph(); graph.add_nodes_from(nodes.index)
    for prefix in ('ns','ew','ne','dt','te','cc','cg'):
        rows=frame.loc[frame.CODE.str.match(r'^'+prefix+r'\d+$')].copy()
        rows['order']=rows.CODE.str.extract(r'(\d+)$',expand=False).astype(int)
        if prefix=='cc': rows=rows.loc[rows.order<=29]
        keys=rows.sort_values('order').station_key.tolist()
        graph.add_edges_from(zip(keys,keys[1:]))
    # Historical Circle Line branch: the source's CC34/CC33 labels correspond
    # to CE1/CE2 during the evaluation period, not the later closed loop.
    for left,right in [('promenade','bayfront'),('bayfront','marinabay')]:
        if left in graph and right in graph: graph.add_edge(left,right)
    # Changi Airport branch starts at Tanah Merah (CG0/EW4 in this source).
    if not nx.is_connected(graph): raise ValueError('Operational MRT graph is unexpectedly disconnected')
    h=nx.harmonic_centrality(graph)
    nodes['harmonic']=[h[k]/max(1,len(graph)-1) for k in nodes.index]
    cbd=nx.single_source_shortest_path_length(graph,'rafflesplace')
    nodes['cbd_hops']=[cbd[k] for k in nodes.index]
    return nodes,frame,graph


def bus_alias(name):
    name=re.sub(r'\b(bukit)\b','bt',str(name).lower())
    name=re.sub(r'\bpark\b','pk',name)
    name=re.sub(r'\b(jalan)\b','jln',name)
    return re.sub('[^a-z0-9]','',name)


def bus_snapshot_files(source_dir):
    rows=[]; source_dir=Path(source_dir)
    for label in ('first','2021','2022','2023'):
        commit_path=source_dir/('bus_first_commit.json' if label=='first' else f'bus_commit_{label}.json')
        if not commit_path.exists(): continue
        commits=json.loads(commit_path.read_text())
        if not commits: continue
        stamp=pd.Timestamp(commits[0]['commit']['committer']['date']).tz_localize(None)
        stops_path=source_dir/f'bus_{label}_stops.json'; services_path=source_dir/f'bus_{label}_services.json'
        if stops_path.exists() and services_path.exists():
            rows.append((stamp,stops_path,services_path))
    return sorted(rows,key=lambda v:v[0])


def station_bus_counts(nodes,snapshot,cfg):
    if snapshot is None: return np.full(len(nodes),np.nan),np.full(len(nodes),np.nan),[]
    _,stop_path,services_path=snapshot
    stop_dict=json.loads(stop_path.read_text()); service_dict=json.loads(services_path.read_text())
    # Later public snapshots append a road-name field; the first three fields
    # retain longitude/latitude/stop description across archive versions.
    stops=pd.DataFrame([(str(k),*v[:3]) for k,v in stop_dict.items()],columns=['stop_id','lon','lat','name'])
    associations={stop:set() for stop in stops.stop_id}
    for service,data in service_dict.items():
        for route in data['routes']:
            for stop in route:
                if str(stop) in associations: associations[str(stop)].add(str(service))
    distances=cdist(metric_xy(nodes.LONGITUDE,nodes.LATITUDE),metric_xy(stops.lon,stops.lat))
    named,nearby,audit=[],[],[]
    stop_alias=stops.name.map(bus_alias).tolist()
    for i,(key,node) in enumerate(nodes.iterrows()):
        eligible=np.flatnonzero(distances[i]<=cfg['station_named_bus_radius_m'])
        alias=bus_alias(re.sub(r'\s+(mrt/lrt|mrt|lrt)\s+station$','',node.NAME))
        matched=[j for j in eligible if alias in stop_alias[j] and ('stn' in stop_alias[j] or 'int' in stop_alias[j] or 'ter' in stop_alias[j])]
        close=np.flatnonzero(distances[i]<=cfg['station_bus_radius_m'])
        named_services=set().union(*(associations[stops.stop_id.iloc[j]] for j in matched))
        nearby_services=set().union(*(associations[stops.stop_id.iloc[j]] for j in close))
        named.append(len(named_services)); nearby.append(len(nearby_services))
        for j in matched:
            audit.append(dict(station_key=key,station=node.NAME,stop_id=stops.stop_id.iloc[j],stop_name=stops.name.iloc[j],distance_m=float(distances[i,j]),source_commit_date=str(snapshot[0])))
    return np.asarray(named,float),np.asarray(nearby,float),audit


def transport_fields(buildings,stations,passengers,snapshots,cutoff,cfg,*,measurement_cutoff=None):
    nodes,active,graph=rail_graph(stations,cutoff)
    cutoff=pd.Timestamp(cutoff); measurement_cutoff=pd.Timestamp(measurement_cutoff) if measurement_cutoff is not None else cutoff
    distance=cdist(buildings[['x_m','y_m']],metric_xy(nodes.LONGITUDE,nodes.LATITUDE))
    nearest=distance.argmin(axis=1); node_keys=nodes.index.to_numpy()
    eligible=[s for s in snapshots if s[0]<=measurement_cutoff]
    snapshot=eligible[-1] if eligible else None
    named,nearby,audit=station_bus_counts(nodes,snapshot,cfg)
    code_to_key=dict(zip(active.CODE.str.upper(),active.station_key)); code_to_key.update({'CE1':'bayfront','CE2':'marinabay'})
    volume={}
    if measurement_cutoff>=pd.Timestamp(cfg['ridership_available_at']):
        for code,value in passengers[['PT_CODE','daily_taps']].itertuples(index=False,name=None):
            matches={code_to_key[c] for c in code.split('/') if c in code_to_key}
            if len(matches)==1: volume[next(iter(matches))]=value
    taps=np.array([volume.get(k,np.nan) for k in node_keys])
    result=pd.DataFrame(np.column_stack([distance[np.arange(len(distance)),nearest]/1000,nodes.line_count.to_numpy()[nearest],
                nodes.harmonic.to_numpy()[nearest],nodes.cbd_hops.to_numpy()[nearest],np.log1p(named[nearest]),np.log1p(nearby[nearest]),
                np.log1p(taps[nearest]),np.isnan(named[nearest]).astype(float),np.isnan(taps[nearest]).astype(float)]),columns=TRANSPORT_COLUMNS)
    return result,nodes,active,audit

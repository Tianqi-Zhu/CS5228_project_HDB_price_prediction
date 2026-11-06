"""Recompute appendix explanations from shipped evidence; zero model fitting.

python -m analysis.appendix_review run --output-dir output/appendix-review/run-1
python -m analysis.appendix_review verify
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from analysis.appendix_statistics import score,validate_identity,paired_intervals,blend_moments

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'reproduction/appendix_diagnostics'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.12g',lineterminator='\n')

def inputs():
    manifest=json.loads((INPUT/'source_manifest.json').read_text(encoding='utf-8'))
    bundle=manifest['compact_prediction_bundle']
    if sha(ROOT/bundle['path'])!=bundle['sha256']:raise ValueError('Changed frozen prediction bundle')
    for item in manifest['imports']:
        if item.get('destination') and sha(ROOT/item['destination'])!=item['sha256']:
            raise ValueError('Imported source changed: '+item['destination'])
    bank=np.load(ROOT/bundle['path'],allow_pickle=False)
    raw=pd.read_csv(ROOT/'data/train.csv')
    expected=next(x['sha256'] for x in manifest['imports'] if x['source_path']=='data/train.csv')
    if sha(ROOT/'data/train.csv')!=expected:raise ValueError('Training data differs from pinned source')
    for w in ['Y','J']:
        validate_identity(bank[w+'__row'],bank[w+'__actual'],bank[w+'__month'],raw)
        if not (bank[w+'__month']>=dict(Y='2024-04',J='2024-06')[w]).all():raise ValueError('Origin mismatch')
    np.testing.assert_allclose(bank['J__F2'],(bank['J__tree']+bank['J__tabpfn'])/2,atol=0,rtol=0)
    full=(bank['full__tree']+bank['full__tabpfn'])/2
    if len(full)!=50000 or not np.array_equal(bank['full__row'],np.arange(50000)):raise ValueError('Full query IDs invalid')
    if not np.isfinite(full).all() or (full<=0).any():raise ValueError('Invalid full predictions')
    difference=float(np.max(abs(full-bank['full__export'])))
    if difference>.5e-6+1e-10:raise ValueError('Full candidate reconstruction differs')
    return raw,bank,{'status':'passed','paired_rows':28995,'F2_full_rows':50000,
                     'full_max_abs_difference_sgd':difference,'full_tolerance_sgd':.5e-6+1e-10,
                     'F1_is_F2_member':False,'source_commit':manifest['main_commit'],
                     'fresh_GPU_inference':False,'new_models_fitted':0}

def x11(out,raw,bank):
    y=bank['J__actual'];p={'F1':bank['J__F1'],'tree7':bank['J__tree'],
                         'TabPFN87':bank['J__tabpfn'],'F2':bank['J__F2']}
    metrics=[{'window':'J','model':name,**score(y,v)} for name,v in p.items()]
    table(out/'X11/member_ablation.csv',metrics)
    dump(out/'X11/error_moments.json',blend_moments(y,p['tree7'],p['TabPFN87']))
    rows=[]
    labels={'month':bank['J__month'],
            'support24_36':np.array([f'24={bool(a)}|36={bool(b)}' for a,b in zip(bank['J__support24'],bank['J__support36'])])}
    for dimension,group in labels.items():
        for value in np.unique(group):
            mask=group==value
            for name,prediction in p.items():
                rows.append({'window':'J','dimension':dimension,'group':str(value),'model':name,**score(y[mask],prediction[mask])})
    table(out/'X11/group_metrics.csv',rows)
    dump(out/'X11/blend_identity.json',{'status':'passed','formula':'0.5*tree7 + 0.5*TabPFN87',
         'F1_is_a_member':False,'new_weight_search':False,'Y_status':'missing origin-appropriate F2 predictions',
         'interpretation':'Observed ensemble RMSE improvement is assessed with actual component losses; correlation alone is not evidence of improvement.'})

def x12(out,raw,bank):
    from analysis.pipeline.auxiliary import _normalized_keys
    ids=bank['J__row'];keys=_normalized_keys(raw.iloc[ids])
    buildings=keys.fillna('__MISSING__').astype(str).agg('\x1f'.join,axis=1).to_numpy()
    rows=[]
    for method,labels in [('moving_month_block',bank['J__month']),('building_cluster',buildings)]:
        rows+=paired_intervals(bank['J__actual'],bank['J__F2'],bank['J__F1'],labels,method)
    table(out/'X12/paired_intervals.csv',rows)
    dump(out/'X12/resampling_spec.json',{'status':'J completed; Y unavailable','candidate':'F2','reference':'F1',
        'seed':42,'resamples_per_method':3000,'percentiles':[.025,.975],'month_block':3,
        'month_algorithm':'Non-circular starts 0..T-3, sample ceil(T/3) blocks, concatenate and truncate to T months; keep all rows and multiplicity',
        'building_algorithm':'Normalized Town/Block/Street; draw K of K buildings with replacement; keep all rows and multiplicity',
        'rows':28995,'months':10,'buildings':int(len(np.unique(buildings))),
        'not_included':['training randomness','adaptive selection uncertainty','unknown future regimes'],
        'Y_missing':'No origin-appropriate F2 annual predictions; Y and J must not be pooled',
        'crossing_zero_is_valid_result':True,'not_an_independent_test':True})

def x08(out):
    from sklearn.neighbors import BallTree
    from analysis.pipeline.auxiliary import build_auxiliary_features,EARTH_RADIUS_KM
    mrt=pd.read_csv(ROOT/'data/auxiliary/sg-mrt-stations.csv')
    opened=mrt[mrt.STATUS.astype(str).str.casefold().eq('open')].copy()
    opened['prefix']=opened.CODE.astype(str).str.lower().str.extract(r'^([a-z]+)',expand=False)
    locations=[]
    for (lat,lon),group in opened.groupby(['LATITUDE','LONGITUDE'],sort=False):
        locations.append({'LATITUDE':lat,'LONGITUDE':lon,'codes':'|'.join(sorted(group.CODE.astype(str))),
                          'jrl_crl_only':set(group.prefix)<=set(['cr','cp','js','je','jw'])})
    places=pd.DataFrame(locations);table(out/'X08/location_flags.csv',locations)
    coords=places[['LATITUDE','LONGITUDE']].to_numpy(float)
    full_tree=BallTree(np.radians(coords),metric='haversine')
    reduced=coords[~places.jrl_crl_only.to_numpy()]
    reduced_tree=BallTree(np.radians(reduced),metric='haversine')
    flagged_tree=BallTree(np.radians(coords[places.jrl_crl_only]),metric='haversine')
    rows=[]
    for split in ['train','test']:
        frame=build_auxiliary_features(pd.read_csv(ROOT/f'data/{split}.csv'))
        point=np.radians(frame[['LATITUDE','LONGITUDE']].to_numpy(float))
        if not np.isfinite(point).all():raise ValueError('Coordinate join incomplete')
        old,index=full_tree.query(point,k=1);new=reduced_tree.query(point,k=1)[0][:,0]*EARTH_RADIUS_KM
        old=old[:,0]*EARTH_RADIUS_KM
        np.testing.assert_allclose(old,frame.MRT_DIST_KM,rtol=0,atol=1e-9)
        affected=places.jrl_crl_only.to_numpy()[index[:,0]]
        in1=flagged_tree.query_radius(point,r=1/EARTH_RADIUS_KM,count_only=True)
        delta=new-old
        if np.any(delta < -1e-9):raise ValueError('Removing locations decreased nearest distance')
        row={'split':split,'rows':len(frame),'nearest_flagged_rows':int(affected.sum()),
             'flagged_within_1km_rows':int((in1>0).sum()),'distance_changed_rows':int((delta>1e-9).sum()),
             'mean_nearest_distance_increase_km':float(delta.mean()),'max_increase_km':float(delta.max()),
             'sensitivity_type':'Feature-only counterfactual; no refit or performance claim'}
        rows.append(row)
    archived=pd.read_csv(ROOT/'experiment/appendix/T01/tabpfn/tabpfn_diagscale_mse_cleaning_audit/mrt_summary.csv').set_index('split')
    for r in rows:
        assert r['nearest_flagged_rows']==archived.loc[r['split'],'nearest_jrl_crl_only_rows']
        assert r['flagged_within_1km_rows']==archived.loc[r['split'],'jrl_crl_in_1km_count_rows']
    assert int(places.jrl_crl_only.sum())==34
    table(out/'X08/feature_sensitivity.csv',rows)
    dump(out/'X08/availability_audit.json',{'status':'Source/status audit independently reproduced; historical-date sensitivity refit not run',
        'flagged_source_locations':34,'policy':'Supplied STATUS=open snapshot; exact coordinate deduplication',
        'code_prefixes':['cr','cp','js','je','jw'],'complete_opening_dates_available':False,
        'remaining_locations_certified_historically_open':False,'model_inputs_changed':False,'models_refitted':0,
        'scope':'Flags reproduce the archived JRL/CRL subset; removing only these locations does not reconstruct all historic facility dates',
        'source_hashes':{p:sha(ROOT/p) for p in ['data/auxiliary/sg-mrt-stations.csv','data/auxiliary/sg-hdb-block.csv']}})

def x09(out):
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    month=pd.PeriodIndex(test.RENT_APPROVAL_DATE,freq='M').asi8
    origin=pd.Period('2025-04',freq='M').ordinal
    horizon=month-origin+1
    groups={'same-month context':horizon<=0,'1-10 month':(horizon>=1)&(horizon<=10),'11-16 month':horizon>10}
    table(out/'X09/query_horizons.csv',[{'group':g,'rows':int(m.sum()),'min_month':str(test.loc[m,'RENT_APPROVAL_DATE'].min()),
             'max_month':str(test.loc[m,'RENT_APPROVAL_DATE'].max())} for g,m in groups.items()])
    assert int((horizon<=0).sum())==380
    dump(out/'X09/cutoff_checks.json',{'status':'Query/context boundary audit complete; matched final long-horizon refits not run',
        'latest_training_month':str(train.RENT_APPROVAL_DATE.max()),'test_months':int(test.RENT_APPROVAL_DATE.nunique()),
        'first_future_month':'2025-04','same_month_rows':380,'validated_final_pair_horizon_months':10,
        'no_Y_prediction_slicing_to_fake_L':True,'F1_F2_L_results_available':False,
        'missing':['Fixed 2023-12 origin F1/F2 predictions from pre-origin fits/selection','Full February-2025 strict-context submission refits if claimed']})

def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    scores=pd.read_csv(out/'X11/member_ablation.csv')
    fig,axes=plt.subplots(1,2,figsize=(8,3.0),layout='constrained')
    axes[0].bar(scores.model,scores.rmse_sgd,color=['#607589','#60826b','#ad744d','#385c7c'])
    axes[0].set(ylim=(438,446),ylabel='RMSE (SGD)',title='J: saved component/pipeline scores')
    axes[0].tick_params(axis='x',rotation=20)
    monthly=pd.read_csv(out/'X11/group_metrics.csv');monthly=monthly[monthly.dimension=='month']
    a=monthly[monthly.model=='F2'].set_index('group');b=monthly[monthly.model=='F1'].set_index('group')
    delta=a.rmse_sgd-b.rmse_sgd
    axes[1].bar(delta.index,delta,color=['#385c7c' if v<0 else '#ad744d' for v in delta])
    axes[1].axhline(0,color='black',lw=.6);axes[1].set(ylabel='F2 minus F1 RMSE (SGD)',title='J: monthly differences')
    axes[1].tick_params(axis='x',rotation=55)
    for suffix in ['png','pdf']:fig.savefig(out/f'X11/complementarity.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    rows=pd.read_csv(out/'X12/paired_intervals.csv');rows=rows[rows.metric=='rmse_sgd']
    fig,ax=plt.subplots(figsize=(6,2.5),layout='constrained')
    for i,r in enumerate(rows.itertuples()):
        ax.plot([r.p025,r.p975],[i,i],lw=2,color='#385c7c');ax.plot(r.candidate_minus_reference,i,'o',color='#ad744d')
    ax.axvline(0,color='black',lw=.7,ls='--');ax.set_yticks(range(len(rows)),['3-month moving blocks','Building clusters'])
    ax.set(xlabel='F2 minus F1 RMSE (SGD)',title='J: descriptive fixed-prediction 95% intervals')
    for suffix in ['png','pdf']:fig.savefig(out/f'X12/paired_intervals.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('--output-dir',type=Path,required=True)
    commands.add_parser('verify');args=p.parse_args(argv)
    raw,bank,identity=inputs()
    if args.command=='verify':print(json.dumps(identity,indent=2));return 0
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('Use a new directory below repository output/')
    protected=[ROOT/'data/train.csv',ROOT/'data/test.csv',INPUT/'inputs/predictions.npz',
               *sorted((ROOT/'data/auxiliary').glob('*.csv'))]
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    out.mkdir(parents=True)
    x11(out,raw,bank);x12(out,raw,bank);x08(out);x09(out);plot(out)
    dump(out/'X13/component_replay.json',identity)
    if before!={p.relative_to(ROOT).as_posix():sha(p) for p in protected}:raise ValueError('Scientific inputs changed')
    dump(out/'execution.json',{'status':'passed','new_models_fitted':0,'inputs_unchanged':True,'input_hashes':before,
        'completed':['X11 J member/pipeline/group/MSE analysis','X12 J month/building paired sensitivity',
                     'X08 independent MRT subset/feature audit','X09 query-horizon/context audit','X13 saved F2 component reconstruction'],
        'not_run':['X07 matched seed refits','X08 historical-date matched model refits','X09 final-pipeline L refits','X12 Y (F2 predictions absent)','X13 fresh Conda install/full GPU training'],
        'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,
        'code_hashes':{name:sha(ROOT/name) for name in ['analysis/appendix_review.py','analysis/appendix_statistics.py']}})
    print(json.dumps({'status':'passed','output':str(out),'new_models_fitted':0},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())

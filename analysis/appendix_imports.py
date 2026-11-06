"""Import pinned, read-only evidence for appendix analysis; never train or fetch.

git show reads committed blobs, so local main drafts and report files are excluded.
Existing archive bytes are never overwritten unless they are identical.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import subprocess
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'reproduction/appendix_diagnostics'
MAIN='d075c81a4719cad57f8f46d290760324afa14c6c'
IMPORTS=[]

def blob(commit,path):
    content=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)
    IMPORTS.append({'source_commit':commit,'source_path':path,'sha256':hashlib.sha256(content).hexdigest()})
    return content

def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=data:
        raise ValueError('Refuse to overwrite differing evidence: '+str(path))
    path.write_bytes(data)

def compact_arrays(arrays):
    buffer=BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(arrays.items()):
            stream=BytesIO();np.save(stream,np.asarray(value),allow_pickle=False)
            info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,stream.getvalue())
    return buffer.getvalue()

def archive(ref,target,paths):
    commit=subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()
    for path in paths:
        data=blob(commit,path)
        dest=ROOT/target/path
        write_once(dest,data)
        IMPORTS[-1]['destination']=dest.relative_to(ROOT).as_posix()

def main():
    bank=np.load(BytesIO(blob(MAIN,'experiment/main/M00/inputs/validation.npz')),allow_pickle=False)
    raw_blob=blob(MAIN,'data/train.csv')
    assert hashlib.sha256((ROOT/'data/train.csv').read_bytes()).digest()==hashlib.sha256(raw_blob).digest()
    raw=pd.read_csv(BytesIO(raw_blob))
    recent=pd.read_csv(BytesIO(blob(MAIN,'reports/zch_anniversary_tabpfn_blend/predictions_june.csv')),float_precision='round_trip')
    np.testing.assert_array_equal(recent.row_id,bank['J__row'])
    np.testing.assert_array_equal(recent.actual,bank['J__actual'])
    arrays={k:bank[k] for w in ['Y','J'] for k in [w+'__row',w+'__actual',w+'__F1',w+'__support24',w+'__support36']}
    arrays.update(J__tree=recent.zch_original.to_numpy(),J__tabpfn=recent.tabpfn87.to_numpy())
    arrays['J__F2']=(arrays['J__tree']+arrays['J__tabpfn'])/2
    for w in ['Y','J']:
        arrays[w+'__month']=raw.loc[arrays[w+'__row'],'RENT_APPROVAL_DATE'].to_numpy(dtype='U7')
    folder='reports/zch_anniversary_tabpfn_blend/'
    frames=[pd.read_csv(BytesIO(blob(MAIN,folder+name)),float_precision='round_trip') for name in
            ['predictions_zch_full.csv','predictions_tabpfn87_full.csv','submission_zch_tabpfn87_equal_150k.csv']]
    for f in frames:np.testing.assert_array_equal(f.iloc[:,0],np.arange(50000))
    for key,f in zip(['full__tree','full__tabpfn','full__export'],frames):arrays[key]=f.iloc[:,1].to_numpy()
    arrays['full__row']=np.arange(50000)
    assert np.max(abs((arrays['full__tree']+arrays['full__tabpfn'])/2-arrays['full__export']))<=.5e-6+1e-10
    content=compact_arrays(arrays)
    write_once(HERE/'inputs/predictions.npz',content)
    for path,name in [('experiment/main/M00/inputs/metadata.json','main_metadata.json'),
                      ('experiment/main/M07/evaluation/f2_manifest.json','f2_identity.json'),
                      ('experiment/main/M08/evaluation/protocol_checks.json','paired_protocol.json'),
                      ('experiment/main/M09/evaluation/verification.json','main_replay_scope.json')]:
        write_once(HERE/'inputs'/name,blob(MAIN,path))
        IMPORTS[-1]['destination']=(HERE/'inputs'/name).relative_to(ROOT).as_posix()
    archive('origin/rnn-lstm-market-features','experiment/appendix/X05/external_studies/rnn_lstm',[
        'docs/market_latent.md','docs/rnn_lstm_report.md',
        'reports/experiments/rnn_lstm_summary/latest/rental_metrics.csv',
        'reports/experiments/rnn_lstm_summary/latest/results.json',
        'reports/experiments/rnn_lstm_summary/cutoffs/protocol.json',
        'reports/experiments/rnn_lstm_summary/cutoffs/cutoff_comparison.csv',
        'analysis/experiments/market_latent.py','analysis/experiments/market_latent_diagnostics.py'])
    archive('origin/experiment/dynamic-rental-surface-plan','experiment/appendix/X05/external_studies/dynamic_surface',[
        'reports/dynamic_market_index/report.md','reports/dynamic_market_index/results.json',
        'reports/dynamic_market_index/pooled_scores.csv','reports/dynamic_market_index/fold_scores.csv',
        'reports/factored_rental_surface/report.md','reports/factored_rental_surface/results.json',
        'reports/factored_rental_surface/hybrid_pooled_scores.csv','reports/factored_rental_surface/hybrid_fold_scores.csv',
        'reports/factored_rental_surface/surface_forecast_scores.csv',
        'analysis/experiments/dynamic_market_index.py','analysis/experiments/factored_rental_surface.py'])
    archive('origin/ziyuan——spatialfeatures','experiment/appendix/X01/spatial_controls',[
        'reports/spatial_features/amenities_adjusted/amenities_adjusted_summary_zh.md',
        'reports/spatial_features/amenities_adjusted/amenity_bin_effects.csv',
        'reports/spatial_features/amenities_adjusted/analysis_manifest.json',
        'reports/spatial_features/amenity_interactions/analysis_manifest.json',
        'reports/spatial_features/amenity_interactions/cluster_covariance.csv',
        'reports/spatial_features/amenity_interactions/mrt_mall_interaction_heatmap.png',
        'reports/spatial_features/amenity_interactions/mrt_mall_interaction_summary_zh.md',
        'reports/spatial_features/amenity_interactions/mrt_mall_joint_cells.csv',
        'reports/spatial_features/tables/adjusted_correlations.csv',
        'reports/spatial_features/feature_dictionary.csv'])
    record={'main_commit':MAIN,'imports':IMPORTS,
            'compact_prediction_bundle':{'path':(HERE/'inputs/predictions.npz').relative_to(ROOT).as_posix(),
                'sha256':hashlib.sha256(content).hexdigest(),'lossless_numeric_subset':True,'rows_J':28995,'rows_Y':35158,'full_query_rows':50000},
            'scope':'Pinned source and saved predictions; raw code snapshots are not standalone executable integrations',
            'scientific_training_performed':False}
    dest=HERE/'source_manifest.json'
    dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'imports':len(IMPORTS),'prediction_bytes':len(content),'new_models_fitted':0}))

if __name__=='__main__':main()

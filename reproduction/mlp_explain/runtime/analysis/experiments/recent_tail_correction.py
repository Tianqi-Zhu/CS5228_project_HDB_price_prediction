"""Earlier-selected historical shrinkage diagnostics, reusing frozen OOF models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.experiments.historical_risk_correction import GRAINS, historical_risk_features, select_candidate
from analysis.experiments.neural_ensemble_uncertainty import RentIntervalCalibrator, interval_metrics


def raw_history_delta(fitting, queries, prediction):
    if fitting.RENT_APPROVAL_DATE.max() >= queries.RENT_APPROVAL_DATE.min():
        raise ValueError('History must precede queries')
    result = np.full(len(queries), np.nan)
    for grain in ['SIGNATURE', 'BLOCK_TYPE']:
        keys = GRAINS[grain]
        means = fitting.groupby(keys, observed=True, dropna=False).MONTHLY_RENT.mean()
        values = means.reindex(pd.MultiIndex.from_frame(queries[keys])).to_numpy(float)
        result = np.where(np.isfinite(result), result, values)
    return np.where(np.isfinite(result), result-prediction, 0.)


def policy_correction(spec, features, raw_delta):
    if spec['kind'] == 'identity':
        return np.zeros(len(features))
    if spec['kind'] == 'recent_mean':
        delta = features.SIGNATURE_12M_mean_minus_prediction.fillna(features.BLOCK_TYPE_12M_mean_minus_prediction).fillna(0).to_numpy()
    elif spec['kind'] == 'low_tail':
        share = features.SIGNATURE_12M_low_share.fillna(features.BLOCK_TYPE_12M_low_share).fillna(0).to_numpy()
        low = features.SIGNATURE_12M_q10_minus_prediction.fillna(features.BLOCK_TYPE_12M_q10_minus_prediction).fillna(0).to_numpy()
        delta = share*np.minimum(low, 0.)
    elif spec['kind'] == 'raw_mean':
        delta = raw_delta
    else:
        raise ValueError('Unknown policy')
    return np.clip(spec['weight']*delta, -600., 600.)


def main():
    from analysis.pipeline.config import DATA
    from analysis.pipeline.data import check_output_path, sha256
    from analysis.pipeline.auxiliary import build_auxiliary_features
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('reports/experiments/neural_ensemble_uncertainty/run-2026-11-02'))
    parser.add_argument('--oof-dir', type=Path, default=Path('reports/experiments/historical_risk_correction/run-2026-11-03'))
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args(); out = check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('New output directory required')
    out.mkdir(parents=True, exist_ok=True)
    before = sha256(DATA/'train.csv')
    specs = [{'name':'identity', 'kind':'identity'}]
    for kind, weights in [('recent_mean',[.25,.5,1.]), ('low_tail',[.5,1.,2.]), ('raw_mean',[.1,.25,.5])]:
        specs += [{'name':f'{kind}_{w}', 'kind':kind, 'weight':w} for w in weights]
    protocol = {'candidates':specs,'selection':'minimum December-2023 RMSE, .25% gain required else identity',
                'prediction_cap_sgd':600,'source_dir':str(a.source_dir),'oof_dir':str(a.oof_dir),
                'source_sha256':before,'code_sha256':sha256(Path(__file__)),
                'limits':'Second exploratory round motivated by inspected outer case. No outer labels choose weights; all candidates are descriptive diagnostics, not new outer-selected winners.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for c in ['TOWN','BLOCK','STREET']:frame[c]=frame[c].astype(str).str.casefold().str.strip()
    fit=frame[frame.RENT_APPROVAL_DATE<'2023-10'];q=frame[(frame.RENT_APPROVAL_DATE>='2023-10')&(frame.RENT_APPROVAL_DATE<'2024-01')]
    features=pd.read_csv(a.oof_dir/'correction_oof_features.csv',index_col='source_row').loc[q.index]
    mean=features.prediction.to_numpy(); raw_delta=raw_history_delta(fit,q.drop(columns='MONTHLY_RENT'),mean)
    tune=q.RENT_APPROVAL_DATE.eq('2023-12').to_numpy();scores=[]
    for spec in specs:
        pred=mean+policy_correction(spec,features,raw_delta);err=pred[tune]-q.MONTHLY_RENT.to_numpy()[tune]
        scores.append({'name':spec['name'],'rmse_sgd':float(np.sqrt(np.mean(err**2))),'mae_sgd':float(np.mean(abs(err)))})
    pd.DataFrame(scores).to_csv(out/'december_selection_scores.csv',index=False)
    name=select_candidate(scores);selected=next(s for s in specs if s['name']==name)
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    print('Frozen second-round selection',name,flush=True)
    saved=pd.concat([pd.read_csv(a.source_dir/f'hybrid_{s}_predictions.csv') for s in ['calibration','outer','test']],ignore_index=True)
    test=build_auxiliary_features(pd.read_csv(DATA/'test.csv'))
    for c in ['TOWN','BLOCK','STREET']:test[c]=test[c].astype(str).str.casefold().str.strip()
    test.index=np.arange(len(frame),len(frame)+len(test));queries=pd.concat([frame,test]).loc[saved.source_row].drop(columns='MONTHLY_RENT')
    fit=frame[frame.RENT_APPROVAL_DATE<'2024-01'];mean=saved.prediction_sgd.to_numpy();std=saved.ensemble_std_sgd.to_numpy()
    members=saved[[c for c in saved if c.startswith('member_')]].to_numpy()
    features=historical_risk_features(fit,queries,mean,std,members[:,:3].mean(1)-members[:,3:].mean(1))
    raw_delta=raw_history_delta(fit,queries,mean)
    cal=saved.partition.eq('calibration').to_numpy();outer=saved.partition.eq('outer').to_numpy();y=saved.actual_sgd.to_numpy()
    ids=[123066,123120,121146];case_mask=saved.source_row.isin(ids).to_numpy()
    results=[]
    # Candidate list and deployed candidate are already frozen; these outer scores do not reselect.
    for spec in specs:
        pred=mean+policy_correction(spec,features,raw_delta)
        err=pred[outer]-y[outer]
        results.append({'name':spec['name'],'selected_earlier':spec['name']==name,'rmse_sgd':float(np.sqrt(np.mean(err**2))),
                        'mae_sgd':float(np.mean(abs(err))),'case_prediction':float(pred[case_mask][0]),
                        'case_rmse':float(np.sqrt(np.mean((pred[case_mask]-y[case_mask])**2)))})
    pd.DataFrame(results).to_csv(out/'predeclared_candidate_outer_diagnostics.csv',index=False)
    pred=mean+policy_correction(selected,features,raw_delta)
    calibrator=RentIntervalCalibrator().fit(y[cal],pred[cal],std[cal])
    (out/'frozen_calibration.json').write_text(json.dumps(calibrator.metadata(),indent=2))
    output=saved[['source_row','month','partition','flat_type','Id','months_after_fit']].copy()
    output['base_prediction_sgd']=mean;output['prediction_sgd']=pred;output['ensemble_std_sgd']=std;output['actual_sgd']=y
    for level,(lower,upper) in calibrator.intervals(pred,std).items():output[f'pi{level}_lower_sgd'],output[f'pi{level}_upper_sgd']=lower,upper
    for split in ['calibration','outer','test']:
        piece=output[output.partition.eq(split)].copy()
        if split=='test':piece=piece.drop(columns='actual_sgd');piece['Id']=piece.Id.astype('int64')
        piece.to_csv(out/f'{split}_predictions.csv',index=False)
    output[case_mask].to_csv(out/'jurong_west_case.csv',index=False)
    pd.DataFrame([{'model':name,**interval_metrics(y[outer],pred[outer],std[outer],calibrator)}]).to_csv(out/'outer_metrics.csv',index=False)
    assert sha256(DATA/'train.csv')==before
    print(pd.DataFrame(scores).to_string(index=False),flush=True)
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':main()

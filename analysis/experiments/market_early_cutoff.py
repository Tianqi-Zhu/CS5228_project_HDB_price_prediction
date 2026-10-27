"""Earlier frozen encoders: trade training history for more rental training rows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from analysis.pipeline.config import DATA, EXPERIMENT_REPORTS
from analysis.pipeline.data import check_output_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=EXPERIMENT_REPORTS/'market_early_cutoff')
    args=parser.parse_args()
    out=check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()):parser.error('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'lookback':3,'seeds':[42,43,44],'cutoffs':{'2022-01':'2021-10','2022-07':'2022-04'},
              'validation':'2024-04..2025-03',
              'scenario':'Monthly updated; actual prior-month rental summaries required',
              'interpretation':'Exploratory reused holdout, not unbiased selection. Each cutoff changes encoder history and rental fitting size. Compare to its own no-latent baseline.',
              'exports':'Both cutoffs, fixed seed 42; earlier published CSVs remain unchanged.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]; expected=None
    raw=pd.read_csv(DATA/'train.csv',usecols=['RENT_APPROVAL_DATE'])
    for cutoff,tune in protocol['cutoffs'].items():
        print('Earlier encoder:',cutoff,flush=True)
        dest=out/cutoff
        command=[sys.executable,'-m','analysis.experiments.market_latent','--lookback','3',
                 '--encoder-cutoff',cutoff,'--encoder-tune-start',tune,'--output-dir',str(dest)]
        with (out/f'{cutoff}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        scores=pd.read_csv(dest/'rental_metrics.csv')
        predictions=pd.read_csv(dest/'validation_predictions.csv')
        ids=predictions.source_row.tolist()
        if expected is None:expected=ids
        assert ids==expected
        for r in scores.itertuples():
            e=predictions[r.variant]-predictions.MONTHLY_RENT
            np.testing.assert_allclose(np.sqrt(np.mean(e**2)),r.rmse_sgd,atol=1e-9)
        meta=json.loads((dest/'results.json').read_text())
        for kind in ['rnn','lstm']:
            part=scores[scores.variant.str.startswith(kind+'_')]
            latent=pd.read_csv(dest/f'{kind}_42_latent.csv')
            colmap={f'{kind}_42_z{i}':f'MARKET_{kind.upper()}_Z{i}' for i in range(1,4)}
            exported=latent.rename(columns={'prediction_month':'RENT_APPROVAL_DATE',**colmap})
            features=exported[['RENT_APPROVAL_DATE',*colmap.values()]]
            assert features.RENT_APPROVAL_DATE.is_unique and features.RENT_APPROVAL_DATE.min()==cutoff
            assert np.isfinite(features[list(colmap.values())]).all().all()
            dates=pd.PeriodIndex(features.RENT_APPROVAL_DATE,freq='M')
            assert (pd.PeriodIndex(latent.input_end_month,freq='M')==dates-1).all()
            features.to_csv(out/f'{kind}_features_from_{cutoff}.csv',index=False)
            pd.DataFrame({'RENT_APPROVAL_DATE':features.RENT_APPROVAL_DATE,
                          'input_start_month':(dates-3).astype(str),'input_end_month':(dates-1).astype(str),
                          'encoder_training_end':str(pd.Period(cutoff,'M')-1)}).to_csv(out/f'{kind}_windows_from_{cutoff}.csv',index=False)
            matched=raw.RENT_APPROVAL_DATE.isin(features.RENT_APPROVAL_DATE).sum()
            rows.append({'cutoff':cutoff,'model':kind,'encoder_windows':meta['encoder_results'][0]['fit_windows'],
                         'feature_months':len(features),'matched_rental_rows':int(matched),
                         'n_fit':int(part.n_fit.iloc[0]),'n_validation':int(part.n_validation.iloc[0]),
                         'mean_rmse':part.rmse_sgd.mean(),'std_rmse':part.rmse_sgd.std(),
                         'baseline_rmse':scores.iloc[0].rmse_sgd,
                         'delta_rmse_vs_baseline':part.rmse_sgd.mean()-scores.iloc[0].rmse_sgd})
    result=pd.DataFrame(rows)
    result.to_csv(out/'cutoff_comparison.csv',index=False)
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()

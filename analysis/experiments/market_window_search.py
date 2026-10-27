"""Select lookback using Oct 2023–Mar 2024; evaluate winners on later months."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from analysis.pipeline.config import EXPERIMENT_REPORTS
from analysis.pipeline.data import check_output_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=EXPERIMENT_REPORTS/'market_window_search')
    args=parser.parse_args()
    out=check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty directory')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'windows':[3,6,9,12],'seeds':[42,43,44],
              'encoder_target_start':'2022-01','encoder_fit_end':'2022-09',
              'encoder_tune':'2022-10..2022-12','rental_selection_fit':'2023-01..2023-09',
              'window_selection':'2023-10..2024-03','outer_evaluation':'2024-04..2025-03',
              'selection_rule':'minimum mean rental RMSE over the three seeds, separately per architecture',
              'note':'Outer period previously inspected in other experiments; not a pristine test set. Monthly-updated scenario.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def run(window,destination,select):
        cmd=[sys.executable,'-m','analysis.experiments.market_latent',
             '--lookback',str(window),'--encoder-target-start','2022-01',
             '--output-dir',str(destination),'--validation-start','2023-10' if select else '2024-04']
        if select:cmd+=['--data-end','2024-04']
        with (out/(destination.name+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[]
    expected_ids=None
    for window in protocol['windows']:
        print('Selection lookback:',window,flush=True)
        dest=out/f'select_{window}m';run(window,dest,True)
        predictions=pd.read_csv(dest/'validation_predictions.csv')
        assert predictions.RENT_APPROVAL_DATE.max()=='2024-03'
        ids=predictions.source_row.to_list()
        if expected_ids is None:expected_ids=ids
        assert expected_ids==ids
        scores=pd.read_csv(dest/'rental_metrics.csv')
        for kind in ['rnn','lstm']:
            part=scores[scores.variant.str.startswith(kind+'_')]
            rows.append({'model':kind,'lookback_months':window,'mean_rmse':part.rmse_sgd.mean(),
                         'std_rmse':part.rmse_sgd.std(),'min_rmse':part.rmse_sgd.min(),
                         'max_rmse':part.rmse_sgd.max(),'mean_mae':part.mae_sgd.mean(),
                         'baseline_rmse':scores.iloc[0].rmse_sgd})
    table=pd.DataFrame(rows);table.to_csv(out/'window_selection.csv',index=False)
    chosen={kind:int(g.loc[g.mean_rmse.idxmin(),'lookback_months']) for kind,g in table.groupby('model')}
    (out/'selected_windows.json').write_text(json.dumps(chosen,indent=2))
    print('Selected on earlier months:',chosen,flush=True)
    for window in sorted(set(chosen.values())):
        run(window,out/f'final_{window}m',False)
    final=[]
    for kind,window in chosen.items():
        dest=out/f'final_{window}m'
        scores=pd.read_csv(dest/'rental_metrics.csv')
        part=scores[scores.variant.str.startswith(kind+'_')]
        final.append({'model':kind,'lookback_months':window,'mean_rmse':part.rmse_sgd.mean(),
                      'std_rmse':part.rmse_sgd.std(),'min_rmse':part.rmse_sgd.min(),
                      'max_rmse':part.rmse_sgd.max(),'baseline_rmse':scores.iloc[0].rmse_sgd})
        # Fixed seed, not the seed with the lowest evaluation score.
        latent=pd.read_csv(dest/f'{kind}_42_latent.csv')
        columns={f'{kind}_42_z{i}':f'MARKET_{kind.upper()}_Z{i}' for i in range(1,4)}
        latent=latent.rename(columns={'prediction_month':'RENT_APPROVAL_DATE',**columns})
        latent[['RENT_APPROVAL_DATE',*columns.values()]].to_csv(out/f'{kind}_selected_features.csv',index=False)
    pd.DataFrame(final).to_csv(out/'selected_outer_metrics.csv',index=False)
    print(table.to_string(index=False),flush=True)
    print(pd.DataFrame(final).to_string(index=False),flush=True)


if __name__=='__main__':main()

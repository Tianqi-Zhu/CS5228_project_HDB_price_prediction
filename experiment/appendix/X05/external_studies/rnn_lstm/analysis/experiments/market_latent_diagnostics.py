"""Diagnose saved latent-market models without retraining or using test labels."""
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from analysis.pipeline.config import DATA
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.evaluation import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=__import__('pathlib').Path)
    args=parser.parse_args(); out=args.run_dir
    config=json.loads((out/'results.json').read_text())['configuration']
    raw=pd.read_csv(DATA/'train.csv'); frame=build_auxiliary_features(raw)
    fitting=frame[(frame.RENT_APPROVAL_DATE>=config['encoder_cutoff'])&(frame.RENT_APPROVAL_DATE<config['validation_start'])].copy()
    validation=frame[frame.RENT_APPROVAL_DATE>=config['validation_start']].copy()
    saved=pd.read_csv(out/'validation_predictions.csv').set_index('source_row')
    rows=[]; slices=[]; concentration=[]
    for name in ['spatial_baseline',*[f'{kind}_{seed}' for kind in ['rnn','lstm'] for seed in config['seeds']]]:
        fit=fitting.copy();val=validation.copy()
        if name!='spatial_baseline':
            latent=pd.read_csv(out/f'{name}_latent.csv').set_index('prediction_month').drop(columns='input_end_month')
            fit=fit.join(latent,on='RENT_APPROVAL_DATE'); val=val.join(latent,on='RENT_APPROVAL_DATE')
        bundle=joblib.load(out/f'{name}_rent.joblib')
        def predict(part):
            xx=part[bundle['features']].copy()
            for c,dtype in bundle['category_dtypes'].items():xx[c]=xx[c].astype('string').astype(dtype)
            with threadpool_limits(limits=2):return bundle['model'].predict(xx)
        train_pred=predict(fit); pred=predict(val)
        np.testing.assert_allclose(pred,saved.loc[val.index,name].to_numpy(),rtol=1e-10,atol=1e-8)
        train_metrics=metrics(fit.MONTHLY_RENT.to_numpy(),train_pred)
        val_metrics=metrics(val.MONTHLY_RENT.to_numpy(),pred)
        rows.append({'model':name,**{'train_'+k:v for k,v in train_metrics.items()},**{'validation_'+k:v for k,v in val_metrics.items()}})
        errors=pred-val.MONTHLY_RENT.to_numpy()
        vv=val[['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN','TOWN_CLEAN','MONTHLY_RENT']].copy()
        vv['prediction']=pred
        vv['forecast_quarter']=((pd.PeriodIndex(vv.RENT_APPROVAL_DATE,freq='M').asi8-pd.Period(config['validation_start'],freq='M').ordinal)//3+1)
        q10,q90=fit.MONTHLY_RENT.quantile([.1,.9])
        vv['rent_band']=pd.cut(vv.MONTHLY_RENT,[-np.inf,q10,q90,np.inf],labels=['below_train_p10','train_p10_to_p90','above_train_p90'])
        total=(errors**2).sum()
        for col in ['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN','rent_band','forecast_quarter']:
            for group,g in vv.groupby(col,observed=True):
                e=g.prediction-g.MONTHLY_RENT
                slices.append({'model':name,'group_type':col,'group':str(group),'n':len(g),
                               **metrics(g.MONTHLY_RENT.to_numpy(),g.prediction.to_numpy()),
                               'squared_error_share':float((e**2).sum()/total)})
        worst=np.sort(errors**2)[::-1]
        concentration.append({'model':name,'top_5pct_sse_share':float(worst[:int(np.ceil(len(worst)*.05))].sum()/total),
                              'median_absolute_error':float(np.median(abs(errors))),
                              'p90_absolute_error':float(np.quantile(abs(errors),.9)),
                              'bias_removed_rmse_diagnostic_only':float(np.sqrt(np.mean((errors-errors.mean())**2))),
                              'train_rent_p10':float(q10),'train_rent_p90':float(q90)})
    pd.DataFrame(rows).to_csv(out/'train_validation_gap.csv',index=False)
    pd.DataFrame(slices).to_csv(out/'error_slices.csv',index=False)
    pd.DataFrame(concentration).to_csv(out/'error_concentration.csv',index=False)
    # Descriptive variability for identical observed core fields, including month/address.
    keys=['RENT_APPROVAL_DATE','TOWN','BLOCK','STREET','FLAT_TYPE','FLAT_MODEL','FLOOR_AREA_SQM','LEASE_COMMENCE_DATE']
    groups=raw[raw.RENT_APPROVAL_DATE>=config['validation_start']].groupby(keys,dropna=False).MONTHLY_RENT.agg(['size','std','min','max'])
    repeated=groups[groups['size']>=2]
    context={'repeated_observed_groups':len(repeated),'rows_in_repeated_groups':int(repeated['size'].sum()),
             'median_within_group_rent_range':float((repeated['max']-repeated['min']).median()),
             'note':'Descriptive only; different units/contracts can share these fields. Not an irreducible-error estimate.',
             'train_target_std':float(fitting.MONTHLY_RENT.std()),'validation_target_std':float(validation.MONTHLY_RENT.std())}
    (out/'diagnostic_context.json').write_text(json.dumps(context,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(concentration).to_string(index=False))
    print(context)


if __name__=='__main__':main()

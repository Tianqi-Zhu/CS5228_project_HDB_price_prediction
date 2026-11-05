"""Describe current forecast errors without turning query labels into inputs."""
import json
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json

OUT=ROOT/'reports/experiments/fourth_wave'
ART=ROOT/'output/models/fourth_wave'


def main():
    raw,hashes=frame_data()
    rows=[]
    for stage in ('outer','june'):
        pred=pd.read_csv(ROOT/f'output/models/third_wave/ablations/{stage}/models/predictions.csv')
        source=raw.iloc[pred.source_row.to_numpy(int)].copy().reset_index(drop=True)
        np.testing.assert_array_equal(pred.actual,source.MONTHLY_RENT)
        source['prediction']=pred.prediction.to_numpy()
        source['rent_band']=pd.cut(source.MONTHLY_RENT,[0,1800,2500,3200,4000,5000,np.inf]).astype(str)
        for group in ('RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN','PLANNING_AREA_CLEAN','rent_band'):
            for name,part in source.groupby(group):
                metrics=score(part.MONTHLY_RENT,part.prediction)
                rows.append({'stage':stage,'group':group,'value':name,'rows':len(part),**metrics,
                    'squared_error_share':float(np.square(part.prediction-part.MONTHLY_RENT).sum()/
                        np.square(pred.prediction-pred.actual).sum())})
    OUT.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'baseline_error_groups.csv',index=False)
    write_json(OUT/'plan.json',{'target_rmse_sgd':430.,'baseline_june_rmse_sgd':442.9024826164729,
        'required_rmse_reduction_pct':100*(1-430/442.9024826164729),
        'required_mse_reduction_pct':100*(1-(430/442.9024826164729)**2),
        'scope':'train.csv + supplied auxiliary; same chronological queries; no external rental targets',
        'proposed_routes':['Forecast information-age augmentation with calendar-cohort features',
            'Spatial comparable rents restricted to prior calendar cohorts',
            'Earlier-selected model/feature and convex ensemble extensions'],
        'kaggle_pairing':'User has no paired local/Kaggle scores; do not assume a constant 20 SGD offset.',
        'source_sha256':hashes})
    print(table.loc[(table.stage=='june')&table.group.isin(['FLAT_TYPE_CLEAN','rent_band'])].to_string(index=False))


if __name__=='__main__':
    main()

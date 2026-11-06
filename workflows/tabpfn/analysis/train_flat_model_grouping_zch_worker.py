"""Refit the two direct flat-model category variants using isolated zch code."""
import gc
import json
from pathlib import Path
import sys
import time
CURRENT=Path(__file__).resolve().parents[1]
ZCH=CURRENT / ".cache/zch-reproduction"
sys.path.insert(0,str(ZCH))
import numpy as np
import pandas as pd
from flat_model_groups import VARIANTS, apply_model_group
from analysis.experiments.frontier_search import ROOT, frame_data
from analysis.experiments.second_wave_anniversary import data
from analysis.experiments.second_wave_cohort_extensions import fit as fit_anniversary
from analysis.experiments.second_wave_normalized import fit as fit_normalized
from analysis.pipeline.data import sha256

OUT=CURRENT / "reports/flat_model_grouping_ensemble"
WEIGHTS={"normalized_market/compact_ratio15_market_last":.13094,"anniversary/broad_lgb15":.09659,
    "anniversary/broad_lgb63":.10352,"anniversary/exact_lgb15":.17311,"anniversary/exact_lgb63":.00658,
    "anniversary/window_cat6":.41380,"anniversary/window_lgb63":.07546}


def main():
    assert ROOT.resolve()==ZCH.resolve()
    summary=json.loads((OUT / "statistics_summary.json").read_text())
    assert sha256(ROOT / "data/train.csv")==summary['source_sha256']
    base=data("2024-06","2025-04")
    source,hashes=frame_data()
    raw_fit=source.loc[source.RENT_APPROVAL_DATE.lt("2024-06")]
    raw_query=source.loc[source.RENT_APPROVAL_DATE.ge("2024-06")]
    actual=np.asarray(base['y_val']).copy()
    base['raw_fit']=raw_fit
    base['reference']={method:raw_fit.groupby(['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform(method).to_numpy(float) for method in ['mean','median']}
    base['y_val']=np.zeros(len(actual))
    assert len(base['y_fit'])==121005 and len(actual)==28995
    for variant in VARIANTS:
        bundle=dict(base)
        bundle['x_fit'],bundle['x_val']=apply_model_group(base['x_fit'],base['x_val'],variant,raw_fit.FLAT_MODEL,raw_query.FLAT_MODEL)
        for column in base['x_fit']:
            if column!='FLAT_MODEL_CLEAN' or variant=='original_plus_group':
                pd.testing.assert_series_equal(bundle['x_fit'][column],base['x_fit'][column])
                pd.testing.assert_series_equal(bundle['x_val'][column],base['x_val'][column])
        combined=np.zeros(len(actual))
        records={}
        for name,weight in WEIGHTS.items():
            group,member=name.split('/')
            record_path=ROOT / 'output/experiment_runs/round_02_anniversary_history' / group / 'screen' / member / 'results.json'
            record=json.loads(record_path.read_text())
            spec,rounds=record['spec'],int(record['iterations'])
            path=OUT / f'predictions_zch_{variant}_{member}.csv'
            begun=time.perf_counter()
            if path.exists():
                saved=pd.read_csv(path)
                np.testing.assert_array_equal(saved.row_id,base['row_val'])
                np.testing.assert_array_equal(saved.actual,actual)
                prediction=saved.predicted.to_numpy(float)
            else:
                print(f'TREES {variant}: fitting {member}, {rounds} rounds',flush=True)
                if group=='normalized_market':model,prediction,count=fit_normalized(spec,bundle,rounds)
                else:model,prediction,count,_=fit_anniversary(spec,bundle,rounds)
                assert count==rounds and np.isfinite(prediction).all()
                pd.DataFrame(dict(row_id=base['row_val'],month=base['date_val'],actual=actual,predicted=prediction)).to_csv(path,index=False,float_format='%.17g')
                del model
                gc.collect()
            combined+=weight*prediction
            records[name]=dict(weight=weight,iterations=rounds,spec=spec,spec_sha256=sha256(record_path),prediction_sha256=sha256(path),seconds=time.perf_counter()-begun)
        output=OUT / f'predictions_zch_{variant}.csv'
        pd.DataFrame(dict(row_id=base['row_val'],month=base['date_val'],actual=actual,predicted=combined)).to_csv(output,index=False,float_format='%.17g')
        (OUT / f'zch_provenance_{variant}.json').write_text(json.dumps(dict(members=records,source_hashes=hashes,context_rows=len(raw_fit),validation_targets_blinded=True,prediction_sha256=sha256(output)),indent=2)+'\n',encoding='utf-8')
        print(f'TREES complete {variant}',flush=True)


if __name__=='__main__':main()

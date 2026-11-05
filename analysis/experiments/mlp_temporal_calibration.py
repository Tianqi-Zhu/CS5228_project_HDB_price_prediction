"""Select calibration on a later inner subwindow; freeze before outer evaluation."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from analysis.pipeline.data import check_output_path
from analysis.pipeline.evaluation import metrics


def design(pred,kind):
    x=np.asarray(pred,dtype=float)
    if kind=='spline':return np.column_stack([x,*[np.maximum(x-t,0) for t in [2000,3000,4000]]])
    return x[:,None]


def train(pred,y,kind):
    if kind=='identity':return None
    if kind=='bias':return float(np.mean(y-pred))
    model=make_pipeline(StandardScaler(),Ridge(alpha=100)) if kind=='spline' else LinearRegression()
    model.fit(design(pred,kind),y)
    return model


def apply(pred,model,kind):
    if kind=='identity':return np.asarray(pred)
    if kind=='bias':return np.asarray(pred)+model
    p=model.predict(design(pred,kind))
    return (p+pred)/2 if kind=='shrink_affine' else p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--blend-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);a=parser.parse_args()
    out=check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()):parser.error('Empty directory required')
    out.mkdir(parents=True,exist_ok=True)
    inner=pd.read_csv(a.blend_dir/'inner_predictions.csv')
    weights=json.loads((a.blend_dir/'selected.json').read_text())
    pred=(weights['tabm_weight']*inner.tabm+weights['mlp_weight']*inner.mlp).to_numpy()
    y=inner.MONTHLY_RENT.to_numpy();early=inner.RENT_APPROVAL_DATE<'2024-01';late=~early
    results=[]
    for kind in ['identity','bias','affine','shrink_affine','spline']:
        model=train(pred[early],y[early],kind)
        calibrated=apply(pred[late],model,kind)
        results.append({'calibrator':kind,**metrics(y[late],calibrated)})
    scores=pd.DataFrame(results);scores.to_csv(out/'selection_metrics.csv',index=False)
    winner=str(scores.loc[scores.rmse_sgd.idxmin(),'calibrator'])
    model=train(pred,y,winner)
    # Both choices are already frozen before opening the outer prediction table.
    (out/'selected.json').write_text(json.dumps({'calibrator':winner,'weights':weights},indent=2))
    joblib.dump(model,out/'calibrator.joblib')
    outer=pd.read_csv(a.blend_dir/'predictions.csv')
    p=apply(outer.prediction.to_numpy(),model,winner)
    outer['uncalibrated']=outer.prediction;outer.prediction=p
    outer.to_csv(out/'predictions.csv',index=False)
    row={'model':'inner_temporal_calibration','calibrator':winner,'n_validation':len(outer),
         **metrics(outer.actual.to_numpy(),p)}
    pd.DataFrame([row]).to_csv(out/'outer_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'source':str(a.blend_dir),'calibration_fit':'2023-10..2023-12',
       'calibration_selection':'2024-01..2024-03','calibration_refit':'2023-10..2024-03',
       'outer':'2024-04..2025-03','caveats':['Base blend weights already selected using all inner months.',
       'Outer period previously inspected in earlier experiments; not a pristine test set.']},indent=2))
    print(scores.to_string(index=False));print(row)


if __name__=='__main__':main()

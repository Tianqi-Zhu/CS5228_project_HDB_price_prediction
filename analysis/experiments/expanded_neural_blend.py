"""Constrained multi-family neural blend selected on earlier predictions only."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.optimize import minimize
from analysis.pipeline.config import DATA
from analysis.pipeline.data import check_output_path,sha256
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.evaluation import metrics
from analysis.experiments.tabm_rent_search import fit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()):p.error('Empty directory required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);before=sha256(DATA/'train.csv')
    root=Path('reports/experiments');inc=root/'mlp_family_blend/run-2026-10-30'
    specs=[('mixture',root/'tabm_rent_search/round5-mixture-fixed-2026-10-30'),
           ('raw_time',root/'tabm_rent_search/round6-rawtime-2026-10-30'),
           ('per_sqm',root/'tabm_rent_search/round4-2026-10-30')]
    frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for c in ['TOWN','STREET','BLOCK']:frame[c]=frame[c].astype(str).str.casefold().str.strip()
    frame['STREET_ID']=frame.TOWN+'|'+frame.STREET;frame['BUILDING_ID']=frame.STREET_ID+'|'+frame.BLOCK
    f=frame[frame.RENT_APPROVAL_DATE<'2023-10'];v=frame[(frame.RENT_APPROVAL_DATE>='2023-10')&(frame.RENT_APPROVAL_DATE<'2024-04')]
    previous=pd.read_csv(inc/'inner_predictions.csv').set_index('source_row')
    previous=previous.loc[v.index]
    coef=json.loads((inc/'selected.json').read_text())
    inner=pd.DataFrame({'source_row':v.index,'month':v.RENT_APPROVAL_DATE,'actual':v.MONTHLY_RENT})
    inner['incumbent']=coef['tabm_weight']*previous.tabm.to_numpy()+coef['mlp_weight']*previous.mlp.to_numpy()
    names=['incumbent']
    for name,path in specs:
        choice=json.loads((path/'selected.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
        spec=next(s for s in protocol['specifications'] if s['name']==choice['model'])
        print('Inner refit',name,choice['epochs'],flush=True)
        _,_,_,_,_,prediction=fit(f,v,spec,epochs=choice['epochs'])
        inner[name]=prediction;names.append(name)
    inner.to_csv(out/'inner_predictions.csv',index=False)
    x=inner[names].to_numpy();y=inner.actual.to_numpy();sd=float(y.std())
    errors=(x-y[:,None])/sd;penalty=.001
    initial=np.array([1.,0.,0.,0.])
    objective=lambda weights:float(np.mean((errors@weights)**2)+penalty*np.sum((weights-initial)**2))
    fitted=minimize(objective,initial,method='SLSQP',bounds=[(0,1)]*len(names),
                     constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-12,'maxiter':300})
    if not fitted.success:raise RuntimeError(fitted.message)
    weights=np.maximum(fitted.x,0);weights/=weights.sum()
    chosen={name:float(w) for name,w in zip(names,weights)}
    (out/'selected.json').write_text(json.dumps({'weights':chosen,'inner_rmse':float(np.sqrt(np.mean((x@weights-y)**2))),
         'selection':'SLSQP nonnegative sum-one weights, normalized MSE plus .001 L2 shrinkage toward incumbent'},indent=2))
    # All weights frozen before outer files are read.
    outer=pd.read_csv(inc/'predictions.csv').set_index('source_row');preds=[outer.prediction.to_numpy()]
    for name,path in specs:
        other=pd.read_csv(path/'predictions.csv').set_index('source_row')
        assert outer.index.equals(other.index);np.testing.assert_array_equal(outer.actual,other.actual)
        preds.append(other.prediction.to_numpy())
    predicted=np.column_stack(preds)@weights
    exported=pd.DataFrame({'source_row':outer.index,'month':outer.month,'actual':outer.actual,'prediction':predicted})
    exported.to_csv(out/'predictions.csv',index=False)
    row={'model':'constrained_neural_family_blend','n_validation':len(outer),**metrics(outer.actual.to_numpy(),predicted)}
    pd.DataFrame([row]).to_csv(out/'outer_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'source_runs':[str(inc),*[str(path) for _,path in specs]],
        'inner':'2023-10..2024-03','outer':'2024-04..2025-03','caveat':'Adaptive research on reused outer holdout; weights use only inner labels.'},indent=2))
    assert sha256(DATA/'train.csv')==before
    print(chosen,flush=True);print(row,flush=True)


if __name__=='__main__':main()

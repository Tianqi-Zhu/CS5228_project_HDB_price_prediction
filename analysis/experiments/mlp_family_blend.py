"""Choose neural-family blend weights on earlier months, then evaluate fixed weights."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from analysis.pipeline.config import DATA
from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.data import check_output_path,sha256
from analysis.pipeline.evaluation import metrics
from analysis.experiments.tabm_rent_search import fit as fit_tabm
from analysis.experiments.embedding_rent_search import fit_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tabm-dir',type=Path,required=True);parser.add_argument('--mlp-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();out=check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()):parser.error('Empty directory required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    before=sha256(DATA/'train.csv');frame=build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for c in ['TOWN','STREET','BLOCK']:frame[c]=frame[c].astype(str).str.casefold().str.strip()
    frame['STREET_ID']=frame.TOWN+'|'+frame.STREET;frame['BUILDING_ID']=frame.STREET_ID+'|'+frame.BLOCK
    f=frame[frame.RENT_APPROVAL_DATE<'2023-10'];v=frame[(frame.RENT_APPROVAL_DATE>='2023-10')&(frame.RENT_APPROVAL_DATE<'2024-04')]
    tabm_protocol=json.loads((a.tabm_dir/'protocol.json').read_text())
    selections=pd.read_csv(a.tabm_dir/'selection_metrics.csv')
    if len(selections)!=len(tabm_protocol['specifications']):parser.error('TabM inner selection still in progress')
    output=v[['RENT_APPROVAL_DATE','MONTHLY_RENT']].copy();output['source_row']=v.index
    tabm_preds=[]
    for spec in tabm_protocol['specifications']:
        epoch=int(selections[selections.model==spec['name']].iloc[0].epochs)
        print('Inner refit TabM',spec['name'],epoch,flush=True)
        _,_,_,_,_,pred=fit_tabm(f,v,spec,epochs=epoch)
        tabm_preds.append(pred);output[spec['name']]=pred
    output['tabm']=np.mean(tabm_preds,axis=0)
    mlp_protocol=json.loads((a.mlp_dir/'protocol.json').read_text())
    choice=json.loads((a.mlp_dir/'selected.json').read_text())
    spec=next(s for s in mlp_protocol['specifications'] if s['name']==choice['model'])
    print('Inner refit MLP',spec['name'],choice['epochs'],flush=True)
    _,_,_,_,_,pred, _=fit_model(f,v,spec,epochs=choice['epochs'])
    output['mlp']=pred;output.to_csv(out/'inner_predictions.csv',index=False)
    trials=[]
    for weight in [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]:
        pred=weight*output.tabm+(1-weight)*output.mlp
        trials.append({'tabm_weight':weight,**metrics(v.MONTHLY_RENT.to_numpy(),pred.to_numpy())})
    scored=pd.DataFrame(trials);scored.to_csv(out/'inner_blend_metrics.csv',index=False)
    best=scored.loc[scored.rmse_sgd.idxmin()];weight=float(best.tabm_weight)
    # Freeze the choice before opening any outer predictions.
    (out/'selected.json').write_text(json.dumps({'tabm_weight':weight,'inner_rmse':float(best.rmse_sgd),'mlp_weight':1-weight},indent=2))
    tabm=pd.read_csv(a.tabm_dir/'predictions.csv').set_index('source_row')
    mlp=pd.read_csv(a.mlp_dir/'predictions.csv').set_index('source_row')
    assert tabm.index.equals(mlp.index)
    np.testing.assert_array_equal(tabm.actual,mlp.actual)
    predicted=weight*tabm.prediction+(1-weight)*mlp.prediction
    exported=pd.DataFrame({'source_row':tabm.index,'month':tabm.month,'actual':tabm.actual,'prediction':predicted})
    exported.to_csv(out/'predictions.csv',index=False)
    row={'model':'inner_selected_neural_family_blend','tabm_weight':weight,'n_validation':len(exported),
         **metrics(tabm.actual.to_numpy(),predicted.to_numpy())}
    pd.DataFrame([row]).to_csv(out/'outer_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'tabm_run':str(a.tabm_dir),'mlp_run':str(a.mlp_dir),
        'inner_period':'2023-10..2024-03','outer_period':'2024-04..2025-03',
        'weight_selection':'Earlier-month RMSE only, fixed grid, frozen before reading outer predictions.',
        'limits':'Outer period previously inspected; adaptive experiment families; exploratory, not blind test.'},indent=2))
    assert sha256(DATA/'train.csv')==before
    print(row,flush=True)


if __name__=='__main__':main()

"""Rebuild holdout predictions from saved preprocessing and model checkpoint."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from experiments.feature_transformer_v4 import macro_v4 as MV
from experiments.feature_transformer_v4 import macro_ablation as MA
from experiments.feature_transformer_v4.data import StageState, build_tensors
from experiments.feature_transformer_v4.training import TrainConfig, build_for
from experiments.macro_encoders_v3 import contract as MC
from experiments.macro_encoders_v3.scaling import MacroScalers, ScalerState
from experiments.neural_fusion_v3.macro_bridge import build_stage_macro
from experiments.neural_fusion_v2 import contract as C
from experiments.cross_attention_v11.model import CrossModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/cross_attention_v11'

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('arm', choices=('concat','summary','sequence')); args=parser.parse_args()
    torch.set_num_threads(2)
    folder=OUT/args.arm
    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    expected=np.load(folder/'holdout_predictions.npz')
    contract_dir=MA.CONTRACT_ROOT/MA.CONTRACT_ID
    manifest=C.load_manifest(contract_dir)
    pop=C.load_population(contract_dir,manifest,'orig','A','holdout')
    state=StageState.from_dict(saved['feature_state']); hold=build_tensors(pop,state)
    meta=saved['macro_scaler_meta']
    scalers=MacroScalers(**{name:ScalerState.from_dict(saved['macro_scalers'][name]) for name in ('dbs','dbs_scalar','coe','coe_scalar')}, **meta)
    mm=MC.load_manifest(MC.latest_v3_contract_dir())
    sm=build_stage_macro(mm,'orig','A',scalers=scalers)
    config=TrainConfig(**{k:v for k,v in saved['config'].items() if k in TrainConfig().__dict__})
    base=build_for(state.numeric_names,state.cardinalities(),context_dim=hold.query_context.shape[1],config=config)
    original=MV.build_with_macro(base,lambda m:MA._source_factory(sm,mm,m,48),'both')
    model=original if args.arm=='concat' else CrossModel(original,args.arm)
    model.load_state_dict(saved['state_dict']); model.eval()
    outputs=[]
    with torch.no_grad():
        for batch in MA.MacroDeviceTensors(hold,torch.device('cpu'),sm).batches(512): outputs.append(model(batch).numpy())
    prediction=hold.to_sgd(np.concatenate(outputs))
    assert np.array_equal(hold.row_id,expected['row_id'])
    assert np.array_equal(pop.arrays.origin,expected['origin'])
    delta=float(np.abs(prediction-expected['predicted_sgd']).max())
    if delta >= .01: raise AssertionError(f'replay error {delta} SGD')
    result={'arm':args.arm,'epoch':saved['epoch'],'rows':len(hold),'max_abs_delta_sgd':delta,'feature_scaler_reconstructed':True,'macro_scalers_reconstructed':True}
    (folder/'independent_replay.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__': main()

"""Supplemental immutable inputs and fresh initialization evidence."""
import hashlib, json, pickle
import torch
from experiments.cross_attention_matched467_v12 import run as R
from experiments.feature_transformer_v4.data import fit_stage_state, build_tensors
from experiments.feature_transformer_v4.training import TrainConfig, build_for, set_seed
from experiments.feature_transformer_v4 import macro_v4 as MV, macro_ablation as MA
from experiments.macro_encoders_v3 import contract as MC
from experiments.neural_fusion_v3.macro_bridge import build_stage_macro
from experiments.neural_fusion_v2 import contract as C

def main():
    torch.set_num_threads(2)
    cd=MA.CONTRACT_ROOT/MA.CONTRACT_ID; md=MC.latest_v3_contract_dir(); manifest=C.load_manifest(cd)
    pop=C.load_population(cd,manifest,'orig','B','fit'); state=fit_stage_state(pop,'B/fit'); fit=build_tensors(pop,state)
    mm=MC.load_manifest(md); sm=build_stage_macro(mm,'orig','B')
    hist=torch.load(R.HIST/'checkpoint.pt',map_location='cpu',weights_only=False)
    cfg=TrainConfig(**{k:v for k,v in hist['config'].items() if k in TrainConfig().__dict__})
    set_seed(42); base=build_for(state.numeric_names,state.cardinalities(),context_dim=fit.query_context.shape[1],config=cfg)
    model=MV.build_with_macro(base,lambda m:MA._source_factory(sm,mm,m,48),'both'); rng=R.rng_get()
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items(): h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    paths=[md/'macro/orig/stageB.npz',md/'samples/orig/stageB.csv',md/'features/orig/stageB.npz']
    paths += [R.ROOT/p for p in json.loads((R.OUT/'root_original_guard.json').read_text()) if p.startswith('experiments/')]
    obj={'hashes':{str(p.relative_to(R.ROOT)):R.sha(p) for p in paths},'initial_tensor_sha256':h.hexdigest(),'postbuild_rng_sha256':hashlib.sha256(pickle.dumps(rng)).hexdigest(),'backend':'cpu','threads':2,'torch':torch.__version__}
    R.dump(R.OUT/'supplemental_provenance.json',obj)
if __name__=='__main__': main()

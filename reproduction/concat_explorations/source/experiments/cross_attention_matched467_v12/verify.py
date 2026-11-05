"""Verify all artifacts and inputs independently of the queue."""
import json
import numpy as np
import torch
from experiments.cross_attention_matched467_v12 import run as R
from experiments.feature_transformer_v4.training import metrics

def validate_checkpoint(checkpoint, registration, arm):
    assert checkpoint['arm'] == arm
    assert checkpoint['fingerprint'] == registration['fingerprint']
    assert checkpoint['config'] == registration['config']
    assert checkpoint['epoch'] == 23 and checkpoint['steps'] == 3979

def check_inputs():
    for file in ('root_original_guard.json','registration.json','supplemental_provenance.json'):
        data=json.loads((R.OUT/file).read_text()); hashes=data if file=='root_original_guard.json' else data['hashes']
        for p,h in hashes.items(): assert R.sha(R.ROOT/p)==h,(file,p)
    reg=json.loads((R.OUT/'registration.json').read_text()); extra=json.loads((R.OUT/'supplemental_provenance.json').read_text())
    assert reg['backend']==extra['backend']=='cpu' and reg['threads']==extra['threads']==2 and reg['torch']==extra['torch']==torch.__version__

def main():
    check_inputs(); assert json.loads((R.OUT/'guard_verification.json').read_text())['unchanged']
    registration=json.loads((R.OUT/'registration.json').read_text())
    historical=torch.load(R.HIST/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert registration['config'] == historical['config']
    historical_macro=json.loads((R.HIST/'recovery_state.json').read_text())['stage_macro']['B']
    torch.set_num_threads(2)
    cd=R.MA.CONTRACT_ROOT/R.MA.CONTRACT_ID; manifest=R.C.load_manifest(cd)
    pop=R.C.load_population(cd,manifest,'orig','B','validation')
    mm=R.MC.load_manifest(R.MC.latest_v3_contract_dir())
    reference=np.load(R.HIST/'validation_predictions.npz'); first=None; results={}
    for a in R.ARMS:
        d=R.OUT/a; ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False); p=np.load(d/'validation_predictions.npz'); s=json.loads((d/'summary.json').read_text())
        validate_checkpoint(ck,registration,a)
        assert ck['feature_state'] == historical['state_b']
        assert ck['macro_scalers'] == historical_macro['scalers']
        assert ck['macro_scaler_meta'] == historical_macro['scalers_meta']
        assert s['arm'] == a and s['epoch'] == 23 and s['steps'] == 3979
        assert ck['epoch']==23 and ck['steps']==3979 and len(ck['curve'])==23
        assert [x['epoch'] for x in ck['curve']] == list(range(1,24))
        assert all(x['steps']==173*x['epoch'] for x in ck['curve'])
        assert np.array_equal(p['origin'],pop.arrays.origin)
        assert len(p['row_id']) == registration['validation_rows'] == 35158
        assert len(np.unique(p['row_id'])) == 35158
        assert np.isfinite(p['predicted_sgd']).all()
        for k in ('row_id','actual_sgd','horizon'): assert np.array_equal(reference[k],p[k])
        if first is None: first={k:p[k] for k in ('row_id','origin','actual_sgd','horizon')}
        else:
            for k in first: assert np.array_equal(first[k],p[k])
        actual=metrics(p['actual_sgd'],p['predicted_sgd'])
        for k,v in actual.items(): assert abs(v-s['overall'][k])<1e-8
        for h in np.unique(p['horizon']):
            m=p['horizon']==h; result=metrics(p['actual_sgd'][m],p['predicted_sgd'][m])
            for k,v in result.items(): assert abs(v-s['horizon'][str(h)][k])<1e-8
        st=R.StageState.from_dict(ck['feature_state']); val=R.build_tensors(pop,st)
        scalers=R.MacroScalers(**{n:R.ScalerState.from_dict(ck['macro_scalers'][n]) for n in ('dbs','dbs_scalar','coe','coe_scalar')},**ck['macro_scaler_meta'])
        sm=R.build_stage_macro(mm,'orig','B',scalers=scalers)
        config=R.TrainConfig(**{k:v for k,v in ck['config'].items() if k in R.TrainConfig().__dict__})
        base=R.build_for(st.numeric_names,st.cardinalities(),context_dim=val.query_context.shape[1],config=config)
        original=R.MV.build_with_macro(base,lambda m:R.MA._source_factory(sm,mm,m,48),'both')
        model=R.model_for(original,a); model.load_state_dict(ck['state_dict'])
        assert sum(t.numel() for t in model.parameters()) == s['parameters'] == registration['parameters'][a] == 87297
        delta=float(np.max(np.abs(R.predict(model,val,sm)-p['predicted_sgd'])))
        assert delta<.01 and s['replay_max_abs_delta_sgd']<.01; results[a]={'recomputed':actual,'replay_delta':delta}
    R.dump(R.OUT/'artifact_verification.json',{'passed':True,'arms':results,'original_and_input_hashes_unchanged':True,'identities_match':True})
if __name__=='__main__': main()

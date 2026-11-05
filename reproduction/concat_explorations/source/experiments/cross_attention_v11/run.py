"""A-only, same-input three-arm screening; no B-stage access."""
import copy, hashlib, json, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from experiments.feature_transformer_v4 import macro_v4 as MV
from experiments.feature_transformer_v4 import macro_ablation as MA
from experiments.feature_transformer_v4 import device as DV
from experiments.feature_transformer_v4.data import fit_stage_state, build_tensors
from experiments.feature_transformer_v4.training import TrainConfig, build_for, set_seed, metrics
from experiments.macro_encoders_v3 import contract as MC
from experiments.neural_fusion_v3.macro_bridge import build_stage_macro
from experiments.neural_fusion_v2 import contract as C
from experiments.cross_attention_v11.model import CrossModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/cross_attention_v11'

def main():
    torch.set_num_threads(2)
    config = TrainConfig(pooling='cls', layers=3, schedule_epochs=60)
    contract_dir = MA.CONTRACT_ROOT / MA.CONTRACT_ID
    manifest = C.load_manifest(contract_dir)
    mm = MC.load_manifest(MC.latest_v3_contract_dir())
    fitpop = C.load_population(contract_dir, manifest, 'orig', 'A', 'fit')
    holdpop = C.load_population(contract_dir, manifest, 'orig', 'A', 'holdout')
    state = fit_stage_state(fitpop, 'A/fit')
    fit, hold = build_tensors(fitpop, state), build_tensors(holdpop, state)
    sm = build_stage_macro(mm, 'orig', 'A')
    assert len(fit) == 52310 and len(hold) == 35882
    assert set(fit.row_id).isdisjoint(hold.row_id)
    assert np.allclose(hold.to_sgd(hold.target), hold.actual_sgd, atol=.01)
    for pop in (fitpop, holdpop):
        idx = np.array([sm.position[int(i)] for i in pop.arrays.row_id])
        assert np.all(sm.macro_cutoff_month[idx] <= pop.arrays.origin)
    set_seed(42)
    base = build_for(state.numeric_names, state.cardinalities(), context_dim=fit.query_context.shape[1], config=config)
    original = MV.build_with_macro(base, lambda m: MA._source_factory(sm, mm, m, 48), 'both')
    initial = copy.deepcopy(original.state_dict())
    def model_for(arm):
        m = copy.deepcopy(original)
        m.load_state_dict(initial)
        return m if arm == 'concat' else CrossModel(m, arm)
    models = {name: model_for(name) for name in ('concat', 'summary', 'sequence')}
    param = {name: sum(p.numel() for p in m.parameters()) for name,m in models.items()}
    assert len(set(param.values())) == 1
    for m in models.values():
        src = m if isinstance(m, MV.MacroOnFeatureTransformer) else m.original
        assert all(torch.equal(v, initial[k]) for k,v in src.state_dict().items())
    batch = next(MA.MacroDeviceTensors(fit.subset(np.arange(8)), torch.device('cpu'), sm).batches(8))
    for m in models.values():
        m.eval(); y=m(batch); assert torch.isfinite(y).all()
        y.sum().backward(); m.zero_grad()
    sequence_source = models['sequence'].original.source
    with torch.no_grad():
        from experiments.cross_attention_v11.model import sequence_states
        t=sequence_source.tensors
        for enc,x,mask in ((sequence_source.encoders.dbs,t.dbs_seq,t.dbs_mask),(sequence_source.encoders.coe,t.coe_seq,t.coe_mask)):
            x=x[:8];mask=mask[:8]
            states,valid=sequence_states(enc,x,mask)
            summary,_=enc(x,mask)
            pos=torch.where(mask.bool(),torch.arange(mask.shape[1]),-1).max(1).values.clamp(min=0)
            assert torch.allclose(states[torch.arange(len(x)),pos],summary,atol=1e-6)
    hashes = {}
    for p in [ROOT/'experiments/cross_attention_v11/model.py', ROOT/'experiments/cross_attention_v11/run.py', contract_dir/'manifest.json', contract_dir/'features/orig/stageA.npz', contract_dir/'samples/orig/stageA.csv', MC.latest_v3_contract_dir()/'manifest.json']:
        hashes[str(p.relative_to(ROOT))] = C.sha256_file(p)
    payload={'config':config.to_dict(),'parameters':param,'backend':'cpu, torch '+torch.__version__,'threads':2,'numeric_features':state.numeric_names,'categorical_features':state.categorical_names,'state':state.to_dict(),'fit_signature':fit.parity_signature(),'hold_signature':hold.parity_signature(),'hashes':hashes,'macro_contract':mm.get('contract_id'),'fit_rows':len(fit),'hold_rows':len(hold),'common_initialization':True,'summary_last_state_parity':True}
    (OUT/'preflight.json').write_text(json.dumps(payload,indent=2))
    print('PREFLIGHT',json.dumps({k:payload[k] for k in ('parameters','backend','fit_rows','hold_rows','summary_last_state_parity')}),flush=True)
    device=torch.device('cpu')
    for arm in ('concat','summary','sequence'):
        set_seed(42); model=models[arm]; opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
        arm_dir=OUT/arm; arm_dir.mkdir(exist_ok=True)
        best=float('inf'); stale=0; curve=[]; start=time.perf_counter()
        for epoch in range(1,61):
            begin=time.perf_counter(); model.train(); total=0
            for features,target in MA.MacroDeviceTensors(fit,device,sm).batches(512,shuffle=True,seed=42000+epoch,with_target=True):
                opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model(features),target)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(target)
            model.eval(); preds=[]
            with torch.no_grad():
                for features in MA.MacroDeviceTensors(hold,device,sm).batches(512): preds.append(model(features).numpy())
            pred=hold.to_sgd(np.concatenate(preds)); score=metrics(hold.actual_sgd,pred); lr=opt.param_groups[0]['lr']; sched.step()
            record={'epoch':epoch,'train_mse':total/len(fit),'lr':lr,'seconds':time.perf_counter()-begin,**score}
            curve.append(record); (arm_dir/'curve.json').write_text(json.dumps(curve,indent=2))
            print(arm,json.dumps(record),flush=True)
            if epoch>=5:
                if score['rmse_sgd']<best-1e-9:
                    best=score['rmse_sgd']; stale=0
                    torch.save({'arm':arm,'config':config.to_dict(),'state_dict':model.state_dict(),'feature_state':state.to_dict(),'macro_scalers':{n:getattr(sm.scaler,n).to_dict() for n in ('dbs','dbs_scalar','coe','coe_scalar')},'macro_scaler_meta':{'contract_id':sm.scaler.contract_id,'split_id':sm.scaler.split_id,'stage':sm.scaler.stage,'policy':sm.scaler.policy},'epoch':epoch,'score':score},arm_dir/'checkpoint.pt')
                    np.savez_compressed(arm_dir/'holdout_predictions.npz',row_id=hold.row_id,origin=holdpop.arrays.origin,horizon=hold.horizon,actual_sgd=hold.actual_sgd,predicted_sgd=pred)
                else:
                    stale+=1
                    if stale>=10: break
        saved=torch.load(arm_dir/'checkpoint.pt',map_location='cpu',weights_only=False)
        reload_model=model_for(arm); reload_model.load_state_dict(saved['state_dict']); reload_model.eval(); out=[]
        with torch.no_grad():
            for features in MA.MacroDeviceTensors(hold,device,sm).batches(512): out.append(reload_model(features).numpy())
        reload_pred=hold.to_sgd(np.concatenate(out)); saved_pred=np.load(arm_dir/'holdout_predictions.npz'); delta=float(np.max(np.abs(reload_pred-saved_pred['predicted_sgd']))); assert delta < .01
        bands={str(h):metrics(hold.actual_sgd[hold.horizon==h], reload_pred[hold.horizon==h]) for h in np.unique(hold.horizon)}
        summary={'arm':arm,'selected_epoch':saved['epoch'],'overall':saved['score'],'horizon':bands,'seconds':time.perf_counter()-start,'reload_delta_sgd':delta,'parameters':param[arm]}
        (arm_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'STATUS').write_text('COMPLETED\n')

if __name__ == '__main__':
    try: main()
    except Exception:
        (OUT/'STATUS').write_text('FAILED\n'+traceback.format_exc()); raise

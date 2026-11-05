"""Fixed-budget historical B architecture comparison; resume at epoch boundaries."""
import argparse, copy, hashlib, json, random, time, traceback
from pathlib import Path
import numpy as np
import torch
from experiments.feature_transformer_v4 import macro_v4 as MV, macro_ablation as MA
from experiments.feature_transformer_v4.data import fit_stage_state, build_tensors, StageState
from experiments.feature_transformer_v4.training import TrainConfig, build_for, set_seed, metrics
from experiments.macro_encoders_v3 import contract as MC
from experiments.macro_encoders_v3.scaling import MacroScalers, ScalerState
from experiments.neural_fusion_v3.macro_bridge import build_stage_macro
from experiments.neural_fusion_v2 import contract as C
from experiments.cross_attention_v11.model import CrossModel
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/cross_attention_matched467_v12'
HIST=ROOT/'reports/feature_transformer_v4/runs/v4-macro-primary-s42/arm_both'
ARMS=('concat','summary','sequence'); EPOCHS=23

def batch_seed(epoch): return 42000+epoch
def epoch_range(done): return range(done+1,EPOCHS+1)
def scheduler(opt): return torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
def rng_get(): return (random.getstate(),np.random.get_state(),torch.get_rng_state())
def rng_set(s): random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2])
def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str)); tmp.replace(path)
def save(path,obj):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)
def sha(p): return C.sha256_file(p)
def model_for(original,arm):
    m=copy.deepcopy(original)
    return m if arm=='concat' else CrossModel(m,arm)
def scaler_dict(sm): return {n:getattr(sm.scaler,n).to_dict() for n in ('dbs','dbs_scalar','coe','coe_scalar')}
def predict(model,hold,sm):
    model.eval(); out=[]
    with torch.no_grad():
        for b in MA.MacroDeviceTensors(hold,torch.device('cpu'),sm).batches(4096): out.append(model(b).numpy())
    return hold.to_sgd(np.concatenate(out))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    OUT.mkdir(exist_ok=True); torch.set_num_threads(2)
    ck=torch.load(HIST/'checkpoint.pt',map_location='cpu',weights_only=False)
    config=TrainConfig(**{k:v for k,v in ck['config'].items() if k in TrainConfig().__dict__})
    assert ck['selected_epoch']==23 and config.schedule_length()==60 and config.batch_size==512
    cd=MA.CONTRACT_ROOT/MA.CONTRACT_ID; manifest=C.load_manifest(cd); md=MC.latest_v3_contract_dir(); mm=MC.load_manifest(md)
    fp=C.load_population(cd,manifest,'orig','B','fit'); vp=C.load_population(cd,manifest,'orig','B','validation')
    state=fit_stage_state(fp,'B/fit'); assert state.to_dict()==ck['state_b']
    fit=build_tensors(fp,state); val=build_tensors(vp,state); sm=build_stage_macro(mm,'orig','B')
    side=json.loads((HIST/'recovery_state.json').read_text())['stage_macro']['B']
    assert scaler_dict(sm)==side['scalers']
    assert np.array_equal(sm.window_index,side['row_window_index'])
    hist=np.load(HIST/'validation_predictions.npz')
    for key,arr in [('row_id',val.row_id),('horizon',val.horizon),('actual_sgd',val.actual_sgd)]: assert np.array_equal(hist[key],arr),key
    assert len(fit)==88192 and len(val)==35158
    assert set(fit.row_id).isdisjoint(val.row_id)
    for pop in (fp,vp):
        idx=np.array([sm.position[int(i)] for i in pop.arrays.row_id]); assert np.all(sm.macro_cutoff_month[idx]<=pop.arrays.origin)
    set_seed(42)
    base=build_for(state.numeric_names,state.cardinalities(),context_dim=fit.query_context.shape[1],config=config)
    original=MV.build_with_macro(base,lambda m:MA._source_factory(sm,mm,m,48),'both')
    postbuild=rng_get(); initial=copy.deepcopy(original.state_dict())
    models={a:model_for(original,a) for a in ARMS}
    params={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()}; assert set(params.values())=={87297}
    for m in models.values():
        core=m if isinstance(m,MV.MacroOnFeatureTransformer) else m.original
        assert all(torch.equal(v,initial[k]) for k,v in core.state_dict().items())
        b=next(MA.MacroDeviceTensors(fit.subset(np.arange(8)),torch.device('cpu'),sm).batches(8)); m.eval(); y=m(b); assert torch.isfinite(y).all(); y.sum().backward(); m.zero_grad()
    paths=[Path(__file__),ROOT/'experiments/cross_attention_v11/model.py',HIST/'checkpoint.pt',HIST/'recovery_state.json',HIST/'validation_predictions.npz',cd/'manifest.json',md/'manifest.json']
    paths += [p for p in cd.rglob('*') if p.is_file() and ('stageB' in p.name)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}; fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    reg={'fingerprint':fingerprint,'hashes':hashes,'config':config.to_dict(),'parameters':params,'fit_rows':len(fit),'validation_rows':len(val),'fit_signature':fit.parity_signature(),'validation_signature':val.parity_signature(),'numeric_features':state.numeric_names,'categorical_features':state.categorical_names,'cardinalities':state.cardinalities(),'backend':'cpu','torch':torch.__version__,'threads':2,'epochs':23,'steps_per_epoch':173,'inference_batch':4096,'feature_and_macro_state_match_historical':True,'post_build_rng':True}
    registration=OUT/'registration.json'
    if registration.exists():
        if not args.resume: raise RuntimeError('Existing registration: use --resume')
        assert json.loads(registration.read_text())['fingerprint']==fingerprint
    else: dump(registration,reg)
    print('PREFLIGHT',json.dumps({k:reg[k] for k in ('parameters','fit_rows','validation_rows','backend','fingerprint')}),flush=True)
    for arm in ARMS:
        d=OUT/arm; d.mkdir(exist_ok=True); model=models[arm]; opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay); sched=scheduler(opt)
        checkpoint=d/'checkpoint.pt'; done=0; curve=[]; steps=0
        if checkpoint.exists():
            assert args.resume
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False); assert saved['fingerprint']==fingerprint and saved['arm']==arm
            model.load_state_dict(saved['state_dict']); opt.load_state_dict(saved['optimizer']); sched.load_state_dict(saved['scheduler']); rng_set(saved['rng']); done=saved['epoch']; steps=saved['steps']; curve=saved['curve']
        else: rng_set(postbuild)
        for epoch in epoch_range(done):
            start=time.perf_counter(); total=0.; nsteps=0; model.train()
            for features,target in MA.MacroDeviceTensors(fit,torch.device('cpu'),sm).batches(512,shuffle=True,seed=batch_seed(epoch),with_target=True):
                opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model(features),target)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(target); nsteps+=1
            assert nsteps==173; steps+=nsteps; lr=opt.param_groups[0]['lr']; sched.step()
            curve.append({'epoch':epoch,'train_mse':total/len(fit),'lr':lr,'seconds':time.perf_counter()-start,'steps':steps})
            save(checkpoint,{'arm':arm,'fingerprint':fingerprint,'epoch':epoch,'steps':steps,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng_get(),'curve':curve,'config':config.to_dict(),'feature_state':state.to_dict(),'macro_scalers':scaler_dict(sm),'macro_scaler_meta':side['scalers_meta']})
            dump(d/'curve.json',curve); print(arm,json.dumps(curve[-1]),flush=True)
        assert steps==3979
    # Only after all training completes, reconstruct preprocessing and model from saved checkpoint.
    for arm in ARMS:
        d=OUT/arm; saved=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False); assert saved['epoch']==23 and saved['steps']==3979
        model=models[arm]; model.load_state_dict(saved['state_dict']); pred=predict(model,val,sm)
        np.savez_compressed(d/'validation_predictions.npz',row_id=val.row_id,origin=vp.arrays.origin,horizon=val.horizon,actual_sgd=val.actual_sgd,predicted_sgd=pred)
        replay_state=StageState.from_dict(saved['feature_state']); replay_val=build_tensors(vp,replay_state)
        scalers=MacroScalers(**{n:ScalerState.from_dict(saved['macro_scalers'][n]) for n in ('dbs','dbs_scalar','coe','coe_scalar')},**saved['macro_scaler_meta'])
        replay_sm=build_stage_macro(mm,'orig','B',scalers=scalers)
        replay_base=build_for(replay_state.numeric_names,replay_state.cardinalities(),context_dim=replay_val.query_context.shape[1],config=config)
        replay_original=MV.build_with_macro(replay_base,lambda m:MA._source_factory(replay_sm,mm,m,48),'both'); replay_model=model_for(replay_original,arm); replay_model.load_state_dict(saved['state_dict'])
        delta=float(np.abs(predict(replay_model,replay_val,replay_sm)-pred).max()); assert delta<.01
        summary={'arm':arm,'epoch':23,'steps':3979,'overall':metrics(val.actual_sgd,pred),'horizon':{str(h):metrics(val.actual_sgd[val.horizon==h],pred[val.horizon==h]) for h in np.unique(val.horizon)},'replay_max_abs_delta_sgd':delta,'parameters':params[arm]}
        dump(d/'summary.json',summary); print('RESULT',json.dumps(summary),flush=True)
    guard=json.loads((OUT/'root_original_guard.json').read_text()); dump(OUT/'guard_verification.json',{'checked':len(guard),'unchanged':all(sha(ROOT/p)==v for p,v in guard.items())})
    (OUT/'STATUS').write_text('COMPLETED\n')
if __name__=='__main__':
    try: main()
    except Exception:
        (OUT/'STATUS').write_text('FAILED\n'+traceback.format_exc()); raise

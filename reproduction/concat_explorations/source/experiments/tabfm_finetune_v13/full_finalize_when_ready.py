"""Wait for full-development training then independently replay the full holdout."""
import json,time,traceback
import numpy as np
import torch
from experiments.tabfm_finetune_v13 import full_runner as run
from experiments.tabfm_finetune_v13.guarded_resume import check_sources,file_sha

def metrics(p,a):
    e=p-a
    return {'n':len(e),'rmse_sgd':float(np.sqrt(np.mean(e**2))),'mae_sgd':float(np.mean(np.abs(e))),'bias_sgd':float(np.mean(e))}

def finalize():
    check_sources()
    fit,hold,train,valid,model,fingerprint=run.initialize()
    state=torch.load(run.OUT/'resume.pt',map_location='cpu',weights_only=False)
    if state['fingerprint']!=fingerprint: raise ValueError('resume fingerprint mismatch')
    best=state['bestpack']; params={n:p for n,p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n,p in params.items(): p.copy_(best['params'][n])
    cap=run.get_cache(model,valid[0],run.OUT/'cache'/'validation'/'00000.pt',fingerprint); run.raw_parity(model,valid[0],cap)
    p,b,a,ids=run.predictions(model,valid,fingerprint,params,state['initial'])
    if not all((np.array_equal(ids,best['row_id']),np.array_equal(p,best['predicted_sgd']),np.array_equal(a,best['actual_sgd']),np.array_equal(b,best['baseline_predictions']))): raise ValueError('full replay differs from atomic best checkpoint')
    lookup={int(k):i for i,k in enumerate(hold.row_id)}; idx=np.array([lookup[int(v)] for v in ids]);
    if not np.array_equal(a,hold.actual_sgd[idx].astype(float)): raise ValueError('actual labels mismatch')
    if len(ids)!=35882 or len(np.unique(ids))!=35882: raise ValueError('holdout incomplete')
    groups={str(h):metrics(p[hold.horizon[idx]==h],a[hold.horizon[idx]==h]) for h in np.unique(hold.horizon[idx])}
    report={'scope':'complete eligible development fit (49146), complete development holdout (35882); excludes 3164 cold-start rows; not final all-label refit','best_epoch':best['epoch'],'best':metrics(p,a),'pretrained_baseline':metrics(b,a),'horizon_groups':groups,'raw_and_cache_parity':'exact','full_holdout_replay':'exact','source_and_original_guards':'passed','checkpoint_sha256':file_sha(run.ROOT/'reports/tabfm_v10/weights/regression/model.safetensors'),'training_loss_weighting':'per episode, equal weight','final_verification':'passed'}
    run.atomic_json(run.OUT/'final_report.json',report)
    status=json.loads((run.OUT/'status.json').read_text()); status['final_verification']='passed'; run.atomic_json(run.OUT/'status.json',status)

if __name__=='__main__':
    while True:
        path=run.OUT/'status.json'; s=json.loads(path.read_text()) if path.exists() else {}
        if s.get('status')=='FAILED': run.atomic_json(run.OUT/'finalizer_status.json',{'status':'blocked_training_failed','error':s.get('error')}); break
        if s.get('status')=='training_complete':
            try:
                run.atomic_json(run.OUT/'finalizer_status.json',{'status':'running'}); finalize(); run.atomic_json(run.OUT/'finalizer_status.json',{'status':'passed'})
            except Exception as e:
                run.atomic_json(run.OUT/'finalizer_status.json',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}); raise
            break
        time.sleep(30)

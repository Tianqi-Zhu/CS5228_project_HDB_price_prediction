"""Independent sequential post-training replay and immutable artifact verification."""
import json, numpy as np
from experiments.tabfm_finetune_v13 import runner
from experiments.tabfm_finetune_v13.guarded_resume import check_sources, check_best_transaction, file_sha, pre_register_banks
from experiments.tabfm_finetune_v13.replay import replay
from experiments.tabfm_finetune_v13.data import load_banks

def main():
    queue=json.loads((runner.OUT/'queue_status.json').read_text())
    if queue['status']!='training_complete': raise ValueError('training queue incomplete')
    check_sources(); pre_register_banks()
    saved=json.loads((runner.ROOT/'reports/tabfm_finetune_v13/root_base_weight.json').read_text())
    weight=runner.ROOT/'reports/tabfm_v10/weights/regression/model.safetensors'
    if file_sha(weight)!=saved['sha256']: raise ValueError('full original weight hash mismatch')
    _,hold,_=load_banks(); lookup={int(k):i for i,k in enumerate(hold.row_id)}
    summaries={}; reference=None
    for name in runner.CONFIG['banks']:
        out=runner.OUT/name; check_best_transaction(out)
        if json.loads((out/'status.json').read_text())['status']!='training_complete': raise ValueError('bank incomplete')
        selection=np.load(out/'selection.npz'); current={k:selection[k] for k in selection.files}
        if reference is None: reference=current
        elif any(not np.array_equal(reference[k],current[k]) for k in reference): raise ValueError('paired bank selection mismatch')
        metric=replay(name); best=np.load(out/'best_predictions.npz'); baseline=np.load(out/'baseline_predictions.npz')
        if not np.array_equal(best['row_id'],baseline['row_id']): raise ValueError('paired predictions differ')
        idx=np.array([lookup[int(i)] for i in best['row_id']]); actual=hold.actual_sgd[idx].astype(float)
        if not np.array_equal(actual,best['actual_sgd']): raise ValueError('ground truth mismatch')
        groups={}
        for h in np.unique(hold.horizon[idx]):
            mask=hold.horizon[idx]==h; err=best['predicted_sgd'][mask]-actual[mask]
            groups[str(h)]={'n':int(mask.sum()),'rmse_sgd':float(np.sqrt(np.mean(err**2))),'mae_sgd':float(np.mean(np.abs(err))),'bias_sgd':float(np.mean(err))}
        base_rmse=float(np.sqrt(np.mean((baseline['predicted_sgd']-actual)**2)))
        summaries[name]={'best':metric,'baseline_rmse_sgd':base_rmse,'horizons':groups,'raw_replay':'exact','cached_512_replay':'exact'}
    report={'scope':'1024 train subset and fixed 512 validation monitor; not full populations','banks':summaries,'original_guard':'passed','full_checkpoint_sha256':saved['sha256'],'loss_weighting':'equal weight per episode; variable query counts','final_verification':'passed'}
    runner.atomic_json(runner.OUT/'final_report.json',report); runner.atomic_json(runner.OUT/'queue_status.json',{'status':'training_complete','final_verification':'passed'})
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()

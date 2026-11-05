"""Portable float32 CUDA runner; same bank, model and optimization as local capacity v21."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.training_schedule_v15 import WarmupCosine
from experiments.feature_transformer_v4.models import FeatureTransformer
from experiments.feature_transformer_v4.macro_v4 import MacroOnFeatureTransformer

ROOT = Path(__file__).resolve().parents[2]
CAPACITY = {'width':128,'depth':3}
DEVICE=torch.device('cpu')
STOP=False
DEADLINE=float('inf')
OUT = ROOT / 'reports/concat_capacity_v21/w128_d3'
LRS = (1e-3, 5e-4, 3e-4)
SEED = 42
BATCH = 512
MAX_EPOCHS = 40
PATIENCE = 5

def build_model(names, cardinalities, source=None):
    from experiments.concat_capacity_v21.model import make_model
    return make_model(names,cardinalities,CAPACITY['width'],CAPACITY['depth'])

def epoch_order(n, epoch):
    return np.random.default_rng(SEED + epoch).permutation(n)

def rmse_sgd(actual, residual, anchor):
    return float(np.sqrt(np.mean((np.asarray(actual) - 1000*(np.asarray(residual)+np.asarray(anchor)))**2)))

def atomic_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)

def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n'); os.replace(tmp, path)

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)

def make_batch(pop, ids, windows):
    return {'numeric': torch.as_tensor(pop.numeric[ids], dtype=torch.float32,device=DEVICE),
            'categorical': torch.as_tensor(pop.categorical[ids], dtype=torch.long,device=DEVICE),
            'origin': torch.as_tensor(pop.origin[ids], dtype=torch.long,device=DEVICE)}

def predict(model, pop, windows):
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(pop.row_id), BATCH):
            ids = np.arange(start, min(start+BATCH, len(pop.row_id)))
            out.append(model(make_batch(pop, ids, windows)).detach().cpu().numpy())
    return np.concatenate(out)

def fingerprint(*objects):
    h = hashlib.sha256()
    for obj in objects:
        if isinstance(obj, np.ndarray):
            h.update(str(obj.dtype).encode()); h.update(str(obj.shape).encode()); h.update(np.ascontiguousarray(obj).tobytes())
        else: h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()

def source_hashes():
    paths=['cluster/concat_v21/train.py','experiments/concat_capacity_v21/model.py','experiments/feature_transformer_v4/models.py','experiments/feature_transformer_v4/macro_v4.py','experiments/training_schedule_v15.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}

def train_update(model, optimizer, scheduler, batch, target):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = (output-target).square().mean()
    if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    used_lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), float(norm), float(used_lr)

def validate_resume(epoch, offset, n_rows, batch_size, updates_completed):
    if epoch < 1 or not 0 <= offset <= n_rows or (offset != n_rows and offset % batch_size):
        raise RuntimeError('invalid resume position')
    expected = (epoch-1)*math.ceil(n_rows/batch_size)+math.ceil(offset/batch_size)
    if updates_completed != expected:
        raise RuntimeError('scheduler update count does not match resume position')

def train_candidate(lr, fit, val, names, cards, windows, identity):
    path = OUT / ('lr_' + str(lr))
    path.mkdir(parents=True, exist_ok=True)
    progress, best = path/'progress.pt', path/'best.pt'
    seed(); model = build_model(names, cards, copy.deepcopy(windows)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(fit.row_id) / BATCH)
    scheduler = WarmupCosine(optimizer, total_steps=steps_per_epoch * MAX_EPOCHS, warmup_steps=steps_per_epoch, min_lr_ratio=.01, start_ratio=.01)
    epoch, offset, best_rmse, best_epoch, stale, history = 1, 0, float('inf'), 0, 0, []
    if progress.exists():
        s = torch.load(progress, map_location='cpu', weights_only=False)
        if s['identity'] != identity or s['lr'] != lr: raise RuntimeError('resume identity mismatch')
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        torch.set_rng_state(s['torch_rng']); np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
        if DEVICE.type=='cuda': torch.cuda.set_rng_state_all(s['cuda_rng'])
        epoch, offset, best_rmse, best_epoch, stale, history = (s[k] for k in ('epoch','offset','best_rmse','best_epoch','stale','history'))
        validate_resume(epoch, offset, len(fit.row_id), BATCH, scheduler.updates_completed)
        if s.get('complete'): return s['result']
    def save(complete=False, result=None):
        atomic_save(dict(identity=identity,lr=lr,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None,numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                         epoch=epoch,offset=offset,best_rmse=best_rmse,best_epoch=best_epoch,stale=stale,history=history,
                         complete=complete,result=result),progress)
    while epoch <= MAX_EPOCHS and stale < PATIENCE:
        start_time = time.monotonic(); model.train(); order = epoch_order(len(fit.row_id), epoch)
        while offset < len(order):
            if STOP or time.monotonic()>=DEADLINE:
                save(); return None
            ix = order[offset:offset+BATCH]
            target = torch.as_tensor(fit.target[ix],dtype=torch.float32,device=DEVICE)
            loss, grad_norm, used_lr = train_update(model, optimizer, scheduler, make_batch(fit, ix, windows), target)
            offset += len(ix)
            if math.ceil(offset/BATCH)%25==0 or STOP or time.monotonic()>=DEADLINE: save()
            batch_number = math.ceil(offset / BATCH)
            if batch_number % 10 == 0 or offset == len(order):
                record = dict(status='training', identity=identity, candidate_lr=lr, epoch=epoch,
                              batch=batch_number, batches_per_epoch=steps_per_epoch, rows_seen=offset,
                              training_mse=loss, gradient_norm=grad_norm, learning_rate=used_lr,
                              next_learning_rate=optimizer.param_groups[0]['lr'])
                atomic_json(record, path/'progress.json')
                print(json.dumps(record), flush=True)
        pred = predict(model, val, windows); score = rmse_sgd(val.actual_sgd,pred,val.anchor_ksgd)
        elapsed = time.monotonic()-start_time
        history.append(dict(epoch=epoch,rmse_sgd=score,elapsed_seconds=elapsed, supervised_rows=len(order), unique_rows=len(np.unique(fit.row_id[order]))))
        if score < best_rmse:
            best_rmse, best_epoch, stale = score, epoch, 0
            atomic_save(dict(identity=identity,lr=lr,epoch=epoch,model=model.state_dict(),predictions=pred,rmse_sgd=score),best)
        else: stale += 1
        print(json.dumps(dict(lr=lr,**history[-1])),flush=True)
        epoch += 1; offset = 0; save()
    result = dict(lr=lr,best_epoch=best_epoch,best_rmse_sgd=best_rmse,history=history,checkpoint=str(best))
    save(True,result); atomic_json(result,path/'result.json'); return result

def main():
    global OUT,CAPACITY,DEVICE,DEADLINE,MAX_EPOCHS,STOP
    import fcntl,csv,signal,platform
    from types import SimpleNamespace
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=['w128_d3','w128_d6'],required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--smoke',action='store_true');ap.add_argument('--stop-after-seconds',type=int,default=9000);args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable; submit through Slurm GPU partition, not login-node training')
    DEVICE=torch.device(args.device);torch.set_num_threads(2)
    if DEVICE.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    def request_stop(signum,frame):
        global STOP
        STOP=True
    for signum in (signal.SIGTERM,signal.SIGINT,getattr(signal,'SIGUSR1',signal.SIGTERM)):signal.signal(signum,request_stop)
    DEADLINE=time.monotonic()+args.stop_after_seconds;CAPACITY={'width':128,'depth':3 if args.candidate=='w128_d3' else 6}
    OUT=ROOT/'reports/concat_soc_v21'/('smoke_'+args.device if args.smoke else 'selection')/args.candidate;OUT.mkdir(parents=True,exist_ok=True)
    if args.smoke:MAX_EPOCHS=2
    with (OUT/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=ROOT/'reports/concat_aligned_v20';meta=json.loads((parent/'feature_manifest.json').read_text());bank=parent/'bank.npz'
        if meta['status']!='ready' or hashlib.sha256(bank.read_bytes()).hexdigest()!=meta['array_sha256']:raise ValueError('bank checksum mismatch')
        with np.load(bank,allow_pickle=False) as arrays:
            fit=SimpleNamespace(**{k[6:]:arrays[k].copy() for k in arrays.files if k.startswith('train_')});val=SimpleNamespace(**{k[4:]:arrays[k].copy() for k in arrays.files if k.startswith('val_')})
        if (len(fit.row_id),len(val.row_id))!=(114842,35158) or np.intersect1d(fit.row_id,val.row_id).size:raise ValueError('split mismatch')
        if args.smoke:
            fit=SimpleNamespace(**{k:v[:32] for k,v in vars(fit).items()});val=SimpleNamespace(**{k:v[:16] for k,v in vars(val).items()})
        names=meta['numeric_names'];cards=meta['cardinalities']
        config={'width':128,'depth':CAPACITY['depth'],'heads':4,'ff_multiplier':2,'dropout':.05,'lr':3e-4,'batch':512,'seed':42,'max_epochs':MAX_EPOCHS,'patience':5,'warmup_epochs':1,'schedule_epochs':MAX_EPOCHS,'target':'teammate residual/1000, add same anchor once','precision':'float32; TF32 off','device_type':DEVICE.type,'smoke':args.smoke,'checkpoint_every_batches':25,'final_refit':False}
        environment={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if DEVICE.type=='cuda' else None}
        identity=fingerprint(meta,config,source_hashes())
        atomic_json({'identity':identity,'configuration':config,'environment':environment,'metadata':meta,'source_hashes':source_hashes()},OUT/'protocol.json')
        atomic_json({'status':'training','pid':os.getpid(),'candidate':args.candidate,'device':str(DEVICE),'identity':identity},OUT/'status.json')
        print(json.dumps({'status':'training','candidate':args.candidate,'environment':environment,'train_rows':len(fit.row_id),'validation_rows':len(val.row_id)}),flush=True)
        result=train_candidate(3e-4,fit,val,names,cards,None,identity)
        if result is None:
            atomic_json({'status':'paused','candidate':args.candidate,'resume':'submit same candidate and output directory'},OUT/'status.json');print('Checkpoint saved; resubmit this candidate to resume.',flush=True);return 75
        ck=torch.load(result['checkpoint'],map_location='cpu',weights_only=False);seed();model=build_model(names,cards).to(DEVICE);model.load_state_dict(ck['model']);pred=predict(model,val,None)
        if not np.allclose(pred,ck['predictions'],rtol=1e-6,atol=1e-6):raise RuntimeError('reload parity mismatch')
        score=rmse_sgd(val.actual_sgd,pred,val.anchor_ksgd)
        if not np.isclose(score,result['best_rmse_sgd'],rtol=1e-7):raise RuntimeError('metric mismatch')
        with (OUT/'predictions.csv.tmp').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['row_id','actual_sgd','prediction_sgd','anchor_sgd','residual_prediction_sgd','horizon']);w.writerows(zip(val.row_id,val.actual_sgd,1000*(pred+val.anchor_ksgd),1000*val.anchor_ksgd,1000*pred,val.horizon))
        os.replace(OUT/'predictions.csv.tmp',OUT/'predictions.csv')
        atomic_json({**result,'reload_verified':True,'validation_rows':len(val.row_id),'device':str(DEVICE),'environment':environment,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else None,'final_refit_performed':False},OUT/'selection.json');atomic_json({'status':'completed','candidate':args.candidate,'rmse_sgd':score},OUT/'status.json');return 0

if __name__=='__main__':
    raise SystemExit(main())

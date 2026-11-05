"""Raw full-population concat Transformer learning-rate selection; no final refit."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.feature_transformer_v4.models import FeatureTransformer
from experiments.feature_transformer_v4.macro_v4 import MacroOnFeatureTransformer
from experiments.macro_encoders_v3.encoders import build_encoders

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/concat_raw_v14'
LRS = (1e-3, 5e-4, 3e-4)
SEED = 42
BATCH = 512
MAX_EPOCHS = 40
PATIENCE = 5

class NativeSource(nn.Module):
    n_tokens = 2
    d_model = 48
    def __init__(self):
        super().__init__()
        self.encoders = build_encoders(hidden_dim=32, token_dim=48)
    def forward(self, batch):
        d, _ = self.encoders.dbs(batch['dbs_seq'], batch['dbs_mask'])
        c, _ = self.encoders.coe(batch['coe_seq'], batch['coe_mask'])
        index = batch['window_index']
        return torch.stack((d[index], c[index]), 1)

def build_model(names, cardinalities, source=None):
    base = FeatureTransformer(names, cardinalities, d_model=48, layers=3, heads=4,
                              ff_multiplier=2, dropout=.05, pooling='cls')
    return MacroOnFeatureTransformer(base, source if source is not None else NativeSource(), 'both')

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

def make_batch(pop, ids, windows):
    return {'numeric': torch.as_tensor(pop.numeric[ids], dtype=torch.float32),
            'categorical': torch.as_tensor(pop.categorical[ids], dtype=torch.long),
            'origin': torch.as_tensor(pop.origin[ids], dtype=torch.long)}

def predict(model, pop, windows):
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(pop.row_id), BATCH):
            ids = np.arange(start, min(start+BATCH, len(pop.row_id)))
            out.append(model(make_batch(pop, ids, windows)).numpy())
    return np.concatenate(out)

def fingerprint(*objects):
    h = hashlib.sha256()
    for obj in objects:
        if isinstance(obj, np.ndarray):
            h.update(str(obj.dtype).encode()); h.update(str(obj.shape).encode()); h.update(np.ascontiguousarray(obj).tobytes())
        else: h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()

def source_hashes():
    paths = ['experiments/concat_raw_v14/pipeline.py', 'experiments/tabfm_finetune_v13/data_full.py',
             'experiments/feature_transformer_v4/models.py', 'experiments/feature_transformer_v4/macro_v4.py',
             'experiments/macro_encoders_v3/encoders.py', 'experiments/tabfm_finetune_v13/macro_raw.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}

def train_candidate(lr, fit, val, names, cards, windows, identity):
    path = OUT / ('lr_' + str(lr))
    path.mkdir(parents=True, exist_ok=True)
    progress, best = path/'progress.pt', path/'best.pt'
    seed(); model = build_model(names, cards, copy.deepcopy(windows))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    epoch, offset, best_rmse, best_epoch, stale, history = 1, 0, float('inf'), 0, 0, []
    if progress.exists():
        s = torch.load(progress, map_location='cpu', weights_only=False)
        if s['identity'] != identity or s['lr'] != lr: raise RuntimeError('resume identity mismatch')
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        torch.set_rng_state(s['torch_rng']); np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
        epoch, offset, best_rmse, best_epoch, stale, history = (s[k] for k in ('epoch','offset','best_rmse','best_epoch','stale','history'))
        if s.get('complete'): return s['result']
    def save(complete=False, result=None):
        atomic_save(dict(identity=identity,lr=lr,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                         torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                         epoch=epoch,offset=offset,best_rmse=best_rmse,best_epoch=best_epoch,stale=stale,history=history,
                         complete=complete,result=result),progress)
    while epoch <= MAX_EPOCHS and stale < PATIENCE:
        start_time = time.monotonic(); model.train(); order = epoch_order(len(fit.row_id), epoch)
        while offset < len(order):
            ix = order[offset:offset+BATCH]; optimizer.zero_grad(set_to_none=True)
            output = model(make_batch(fit, ix, windows)); target = torch.as_tensor(fit.target[ix],dtype=torch.float32)
            loss = (output-target).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            offset += len(ix); save()
        scheduler.step()
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

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-only', action='store_true', help='run full training/validation selection (default)')
    parser.parse_args(argv)
    torch.set_num_threads(2)
    from experiments.tabfm_finetune_v13.data_full import prepare_phase
    from experiments.tabfm_finetune_v13.macro_raw import make_source
    fit, val, meta, state = prepare_phase('selection', feature_bank='stock')
    if len(fit.row_id) != 114842 or len(val.row_id) != 35158 or len(np.unique(fit.row_id)) != len(fit.row_id) or len(np.unique(val.row_id)) != len(val.row_id) or np.intersect1d(fit.row_id,val.row_id).size:
        raise RuntimeError('population does not match declared full split')
    names = meta['numeric_names']; cards = tuple(int(fit.categorical[:,i].max())+1 for i in range(fit.categorical.shape[1]))
    if len(names) != 184 or len(cards) != 4: raise RuntimeError('unexpected feature bank')
    seed()
    source = make_source(fit, meta, phase='selection', mode='both', d_model=48)
    windows = source  # source supplies phase-local windows through its forward method
    identity = fingerprint(meta, state, source.describe(), source_hashes(), fit.row_id, fit.numeric, fit.categorical, fit.target, fit.origin,
                           fit.actual_sgd, fit.anchor_ksgd, fit.horizon, val.row_id, val.numeric, val.categorical, val.target, val.origin, val.actual_sgd, val.anchor_ksgd, val.horizon)
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(identity=identity,metadata=meta,preprocessing=state,source_hashes=source_hashes(),cardinalities=cards,
                     macro_state=source.describe(), protocol='selection only; no all-label refit; full validation is development data',
                     comparison='End-to-end comparison: shared 188 flat inputs and raw split, plus native macro sequence branches for this model.',
                     min_eligible_epoch=1, batch_size=BATCH, max_epochs=40, patience=5, cosine_tmax=40), OUT/'protocol.json')
    results = [train_candidate(lr,fit,val,names,cards,windows,identity) for lr in LRS]
    winner = min(results,key=lambda x:x['best_rmse_sgd'])
    checkpoint = torch.load(winner['checkpoint'], map_location='cpu', weights_only=False)
    seed(); reloaded = build_model(names, cards, copy.deepcopy(source))
    reloaded.load_state_dict(checkpoint['model'])
    verified = predict(reloaded, val, source)
    if not np.allclose(verified, checkpoint['predictions'], rtol=1e-6, atol=1e-6): raise RuntimeError('reload prediction mismatch')
    if not np.isclose(rmse_sgd(val.actual_sgd, verified, val.anchor_ksgd), winner['best_rmse_sgd'], rtol=1e-8): raise RuntimeError('reload RMSE mismatch')
    atomic_json(dict(identity=identity,selected_lr=winner['lr'],selected_epoch=winner['best_epoch'],
                     selection_rmse_sgd=winner['best_rmse_sgd'],selection_checkpoint=winner['checkpoint'],
                     candidates=results,verified_reload=True,validation_rows_verified=len(verified),final_refit_performed=False), OUT/'selection.json')
    print(json.dumps({'status':'selection-complete','selection':str(OUT/'selection.json')}),flush=True)

if __name__ == '__main__': main()

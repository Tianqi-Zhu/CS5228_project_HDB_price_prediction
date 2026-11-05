"""Export a completed run's last checkpoint; never modifies training artifacts."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from experiments.feature_transformer_v4.models import FeatureTransformer
from experiments.spatiotemporal_v17.models import TemporalFT
from experiments.spatiotemporal_v17.train import predict,metrics,ROOT

def export(path,expected=5000):
    from experiments.spatiotemporal_v17.contract import load_bank
    path=Path(path); cfg=json.loads((path/'config.json').read_text()); ck=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
    if not ck.get('complete') or ck['identity']!=cfg['identity']: raise RuntimeError('checkpoint incomplete or mismatched')
    for source,sha in cfg['source_hashes'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha: raise RuntimeError('model source changed')
    bank=load_bank(cfg['bank'],variant=cfg['variant']); val=bank['val']
    if bank['manifest']!=cfg['manifest'] or len(val['row_id'])!=expected: raise RuntimeError('bank changed')
    if cfg['architecture']=='static': model=FeatureTransformer(cfg['numeric_names'],tuple(cfg['cardinalities']),d_model=48,layers=3,heads=4,ff_multiplier=2,dropout=.05,pooling='cls')
    else: model=TemporalFT(cfg['numeric_names'],cfg['cardinalities'],len(bank['sequence_names']),mode=cfg['architecture'],d_model=48,layers=3,heads=4,ff_multiplier=2,dropout=.05)
    torch.set_num_threads(2); model.load_state_dict(ck['model']); pred=1000*(predict(model,val)+val['anchor_ksgd']); result=metrics(val,pred)
    if not np.isclose(result['rmse_sgd'],ck['history'][-1]['rmse_sgd'],rtol=1e-6,atol=1e-4): raise RuntimeError('last checkpoint metric mismatch')
    tmp=path/'last_predictions.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['row_id','y','pred','month','horizon','town','flat_type','origin']); w.writerows(zip(val['row_id'],val['y_sgd'],pred,val['month'],val['horizon'],val['town'],val['flat_type'],val['origin']))
    os.replace(tmp,path/'last_predictions.csv'); return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);args=parser.parse_args();print(json.dumps(export(args.path)))

"""Prepare full confirmation banks after the active training queue has stopped.

Does not run on import; refuses to overwrite any prior full-stage artifacts.
"""
import argparse, hashlib, json, os
from pathlib import Path
from .contract import OUT


def check_destination(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'refusing to overwrite full-stage artifacts: {path}')


def build_full(split='orig'):
    from experiments.fusion_contract_v2 import export
    from .feature_runner import build
    from .no_mrt import build as strip_mrt
    from .audit import audit
    import numpy as np
    path=OUT/split/'full'; check_destination(path)
    frame=export.load_source(); cutoff={'orig':38,'early':26}[split]
    train=int((frame.MONTH_INDEX<=cutoff).sum()); val=int(((frame.MONTH_INDEX>cutoff)&(frame.MONTH_INDEX<=cutoff+12)).sum())
    if split=='orig' and (train,val)!=(114842,35158): raise ValueError(f'unexpected orig source counts {(train,val)}')
    del frame
    build(stage='full',split=split); strip_mrt(split=split,stage='full'); result=audit(stage='full',split=split,variant='causal_no_mrt')
    if (result['train_rows'],result['validation_rows'])!=(train,val): raise ValueError('full count mismatch')
    path=path/'causal_no_mrt'
    audit_hash=hashlib.sha256((path/'audit.json').read_bytes()).hexdigest()
    for manifest in sorted(path.glob('F*/manifest.json')):
        obj=json.loads(manifest.read_text()); obj['status']='ready'; obj['audit_sha256']=audit_hash
        temp=manifest.with_suffix('.tmp'); temp.write_text(json.dumps(obj,indent=2)); os.replace(temp,manifest)
    gate={'status':'ready','scope':f'{split}/full/causal_no_mrt','train_rows':train,'validation_rows':val,'historical_availability':'MRT snapshot excluded; mall and school snapshots have unverified historical availability'}
    tmp=path/'gate.tmp'; tmp.write_text(json.dumps(gate,indent=2)); os.replace(tmp,path/'gate.json')
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=('orig','early'),default='orig'); a=p.parse_args(); print(json.dumps(build_full(a.split),indent=2))

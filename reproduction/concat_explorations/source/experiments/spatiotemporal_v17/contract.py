"""Frozen-bank loader. Artifacts are written by feature_runner, never at load time."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/spatiotemporal_v17/features'

def load_bank(bank_id, split='orig', stage='screen', variant='snapshot'):
    if variant not in ('snapshot', 'causal_no_mrt'): raise ValueError('unknown variant')
    base=OUT/split/stage
    path=(base if variant == 'snapshot' else base/variant)/bank_id
    manifest=json.loads((path/'manifest.json').read_text())
    with np.load(path/'arrays.npz',allow_pickle=False) as data:
        groups={p:{k[len(p)+1:]:data[k].copy() for k in data.files if k.startswith(p+'_')} for p in ('train','val')}
    return {**groups,**{k:manifest[k] for k in ('numeric_names','categorical_names','cardinalities','sequence_names')},'manifest':manifest}

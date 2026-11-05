"""Immutable per-anchor arrays aligned to frozen v17 screen query populations."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/market_anchor_v18/data'

def load_anchor(anchor_id):
    path=OUT/anchor_id; manifest=json.loads((path/'manifest.json').read_text())
    with np.load(path/'arrays.npz',allow_pickle=False) as a:
        groups={p:{k[len(p)+1:]:a[k].copy() for k in a.files if k.startswith(p+'_')} for p in ('train','val')}
    return {**groups,'manifest':manifest}

"""Recompute screening metrics, reporting incomplete entries without selecting a winner."""
import argparse, csv, json, math
from pathlib import Path
from experiments.concat_warmup_v15.pipeline import atomic_json
from experiments.spatiotemporal_v17.queue import ORDER, OUT

def collect(root, candidates=ORDER):
    completed=[]; incomplete=[]; reference=None
    for name in candidates:
        path=Path(root)/name
        try:
            result=json.loads((path/'result.json').read_text()); config=json.loads((path/'config.json').read_text())
            with (path/'predictions.csv').open(newline='') as f: rows=list(csv.DictReader(f))
            if not result.get('reload_verified') or len(rows)!=5000 or result['identity']!=config['identity']: raise ValueError('incomplete result')
            key=[(r['row_id'],r['y']) for r in rows]
            if len(set(r[0] for r in key))!=len(key): raise ValueError('duplicate row IDs')
            if reference is None: reference=key
            if key!=reference: raise ValueError('validation rows or targets differ')
            score=math.sqrt(sum((float(r['y'])-float(r['pred']))**2 for r in rows)/len(rows))
            if not math.isfinite(score) or not math.isclose(score,result['rmse_sgd'],rel_tol=1e-7,abs_tol=1e-4): raise ValueError('RMSE mismatch')
            completed.append({'bank':name,'rmse_sgd':score,'best_epoch':result['best_epoch'],'training_seconds':result['training_seconds'],'history':result['history']})
        except (OSError,ValueError,KeyError,TypeError) as e: incomplete.append({'bank':name,'reason':str(e)})
    return {'stage':'exploratory 30k/5k screen; not full validation','complete':completed,'incomplete':incomplete,'all_complete':not incomplete,'selection':None}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path); args=parser.parse_args(); result=collect(OUT)
    if args.output: atomic_json(result,args.output)
    print(json.dumps(result,indent=2))

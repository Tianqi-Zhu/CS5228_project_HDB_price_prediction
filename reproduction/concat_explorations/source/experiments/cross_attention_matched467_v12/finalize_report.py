"""Generate a human-readable report after the fixed-budget queue completes."""
import json, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/cross_attention_matched467_v12'
def main():
    while True:
        status=(OUT/'STATUS').read_text().strip() if (OUT/'STATUS').exists() else ''
        if status.startswith('FAILED'): raise RuntimeError(status)
        if status=='COMPLETED': break
        time.sleep(30)
    from experiments.cross_attention_matched467_v12.verify import main as verify
    verify()
    summaries=[json.loads((OUT/a/'summary.json').read_text()) for a in ('concat','summary','sequence')]
    guard=json.loads((OUT/'guard_verification.json').read_text()); assert guard['unchanged']
    rows=['# Matched historical 467 architecture comparison','', 'Historical reference: 467.4027 SGD RMSE on MPS. Fresh arms ran on CPU with two threads, so exact historical reproduction is not expected. The fresh concat arm is the matched baseline.','', '| Arm | RMSE SGD | MAE SGD | Bias SGD | Epochs | Steps | Replay max delta SGD |','|---|---:|---:|---:|---:|---:|---:|', '| Historical MPS reference | 467.4027 | — | — | 23 | 3979 | — |']
    for s in summaries:
        m=s['overall']; rows.append(f"| {s['arm']} | {m['rmse_sgd']:.4f} | {m['mae_sgd']:.4f} | {m['bias_sgd']:.4f} | {s['epoch']} | {s['steps']} | {s['replay_max_abs_delta_sgd']:.6f} |")
    rows += ['',f"Fresh concat minus historical RMSE: {summaries[0]['overall']['rmse_sgd']-467.4027:+.4f} SGD.",'', 'Rank, fresh arms only: '+', '.join(s['arm'] for s in sorted(summaries,key=lambda s:s['overall']['rmse_sgd']))+'.', '', 'All arms used the same fresh initial weights, post-construction training RNG state, fixed 23-epoch budget and 60-epoch cosine schedule. Validation was scored after all training, using batch size 4096. Horizon-level metrics and prediction archives are stored in each arm directory.','',f"Historical guard: {guard['checked']} files unchanged. The benchmark was previously viewed, uses one seed and one validation window, and is not an independent test."]
    (OUT/'REPORT.md').write_text('\n'.join(rows)+'\n'); (OUT/'FINALIZATION_STATUS').write_text('COMPLETED\n')
if __name__=='__main__':
    try: main()
    except Exception:
        (OUT/'FINALIZATION_STATUS').write_text('FAILED\n'+traceback.format_exc()); raise

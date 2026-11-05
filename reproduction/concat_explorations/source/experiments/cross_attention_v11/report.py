"""Assemble final screening report from the three completed run artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/cross_attention_v11'

def main():
    if (OUT/'STATUS').read_text().strip() != 'COMPLETED':
        raise RuntimeError('Training queue is not complete')
    names=('concat','summary','sequence')
    summaries={n:json.loads((OUT/n/'summary.json').read_text()) for n in names}
    replays={n:json.loads((OUT/n/'independent_replay.json').read_text()) for n in names}
    for n in names:
        if summaries[n]['selected_epoch'] != replays[n]['epoch']: raise RuntimeError('Replay is stale: '+n)
    lines=['# Cross-attention v11: orig/A screening','',
      'All results use the same orig/A fit (52,310 rows) and holdout (35,882 rows), seed 42, CPU, and fit-only preprocessing. Stage B was not scored. The historical 467.4027 result uses a different evaluation window and is not comparable to these holdout scores.','',
      '| Arm | Best epoch | RMSE SGD | MAE SGD | Bias SGD | Parameters | Seconds | Replay max delta SGD |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in names:
        s=summaries[n]; m=s['overall']; r=replays[n]
        lines.append(f"| {n} | {s['selected_epoch']} | {m['rmse_sgd']:.4f} | {m['mae_sgd']:.4f} | {m['bias_sgd']:.4f} | {s['parameters']} | {s['seconds']:.1f} | {r['max_abs_delta_sgd']:.6f} |")
    lines += ['', 'Horizon breakdown (months):', '', '| Arm | Horizon | n | RMSE SGD | MAE SGD | Bias SGD |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in names:
        for h,m in sorted(summaries[n]['horizon'].items(),key=lambda x:int(x[0])):
            lines.append(f"| {n} | {h} | {m['n']} | {m['rmse_sgd']:.4f} | {m['mae_sgd']:.4f} | {m['bias_sgd']:.4f} |")
    lines += ['', 'The concatenation arm retains the original 3-layer, 63-token FT plus DBS/COE architecture. Both cross arms replace only the final attention block, use the first two self-attention layers for tabular queries and share all initial weights. Sequence memory uses 63 DBS steps and 12 COE months, with sinusoidal lag and existing source embeddings, and learned missing-window fallback. All arms have 87,249 parameters.', '',
      'The historical outer model has 87,297 parameters because its B/fit flat-model vocabulary has 20 rows versus 19 in A/fit; the other vocabulary sizes are 27, 7, and 2 in both. The additional 48-dimensional embedding row explains the difference.', '',
      'This is one development window and one seed, used for screening. Early-window, multiple-seed, and independent-test confirmation remain outstanding. Independent specification and code-quality reviews found no blocking issues. The root reviewer ran all six tests and checked the 58-file historical guard. Full training curves, preprocessing, source and input hashes, predictions with IDs/origins/horizons and reconstructed-checkpoint replay checks are stored alongside this report. Checkpoints support prediction reconstruction but do not contain optimizer/scheduler state and cannot resume training. Peak memory was not measured: process inspection was denied by the environment.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()

"""Single pre-specified historical 50/50 blend of saved predictions; no training."""
import hashlib,json,os,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
O=Path('reports/best467_ft_lgb_blend_v1'); F=Path('reports/feature_transformer_v4/runs/v4-macro-primary-s42/arm_both/validation_predictions.npz'); L=Path('reports/feature_engineering_v1/predictions/F3_hist_ft_area__main__seed42.csv'); M=L.with_suffix('.metrics.json'); S=F.parent/'summary.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value): (O/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def status(s,**kw): dump('STATUS.json',dict(status=s,pid=os.getpid(),provider='Terra',updated_at=datetime.now(timezone.utc).isoformat(),**kw))
def score(d,p):
 e=d[p].to_numpy()-d.actual_sgd.to_numpy()
 return dict(n=len(e),rmse_sgd=float(np.sqrt(np.mean(e*e))),mae_sgd=float(np.mean(np.abs(e))),bias_sgd=float(np.mean(e)))
def main():
 status('RUNNING')
 manifest=dict(registered_at=datetime.now(timezone.utc).isoformat(),purpose='Historical diagnostic only, repeatedly viewed holdout; no independent test or robustness claim',scheme='0.5 * FT + 0.5 * F3 LightGBM, fixed in advance',no_weight_search=True,no_new_training=True,seed=42,window='2024-04..2025-03',n_expected=35158,lgb_arm='F3_hist_ft_area/main/seed42; source metrics document n_estimators=400',feature_training_caveat='FT and F3 use different training sample counts and feature sets, not a pure architecture comparison',date_provenance='RENT_APPROVAL_DATE from F3 CSV joined by unique ID, not independently present in FT NPZ; FT horizon independently cross-checked against F3 horizon and month relative to 2024-03 cutoff',source_sha256={str(p):sha(p) for p in [F,L,M,S,Path(__file__)]})
 dump('manifest.json',manifest)
 with np.load(F,allow_pickle=False) as z:
  f=pd.DataFrame(dict(row_id=z['row_id'],actual_sgd=z['actual_sgd'],horizon=z['horizon'],ft_prediction=z['predicted_sgd']))
 l=pd.read_csv(L,float_precision='round_trip')
 assert len(f)==len(l)==35158 and f.row_id.is_unique and l.SOURCE_ROW_ID.is_unique
 for col,v in [('model','F3_hist_ft_area'),('split_id','main'),('seed',42),('cutoff_month','2024-03')]: assert l[col].eq(v).all(),col
 metadata=json.loads(M.read_text())['metrics']; assert json.loads(metadata['parameters'])['n_estimators']==400
 d=f.merge(l,left_on='row_id',right_on='SOURCE_ROW_ID',validate='one_to_one',how='outer',indicator=True)
 assert len(d)==35158 and d['_merge'].eq('both').all()
 assert np.array_equal(d.actual_sgd,d.actual_rent_sgd) and np.array_equal(d.horizon,d.horizon_months)
 months=pd.PeriodIndex(d.RENT_APPROVAL_DATE,freq='M'); cutoff=pd.Period('2024-03',freq='M')
 assert np.array_equal(months.asi8-cutoff.ordinal,d.horizon.to_numpy())
 assert set(d.horizon)==set(range(1,13)) and months.min()==pd.Period('2024-04') and months.max()==pd.Period('2025-03')
 d['lgb_prediction']=d.predicted_rent_sgd; d['blend_prediction']=.5*(d.ft_prediction+d.lgb_prediction)
 d=d[['row_id','RENT_APPROVAL_DATE','horizon','actual_sgd','ft_prediction','lgb_prediction','blend_prediction']].rename(columns={'RENT_APPROVAL_DATE':'month'})
 for name,expected in [('ft',467.40269960753227),('lgb',469.0081832787802)]: assert abs(score(d,name+'_prediction')['rmse_sgd']-expected)<1e-9
 d.to_csv(O/'predictions.csv',index=False,float_format='%.17g')
 d=pd.read_csv(O/'predictions.csv',float_precision='round_trip')
 names=['ft','lgb','blend']
 metrics={'overall':{n:score(d,n+'_prediction') for n in names},'horizon_bands':{},'monthly':{},'residual_correlation':float(np.corrcoef(d.ft_prediction-d.actual_sgd,d.lgb_prediction-d.actual_sgd)[0,1]),'training_rows':{'ft':88192,'lgb':114842}}
 for lo in (1,4,7,10):
  sub=d[d.horizon.between(lo,lo+2)]; metrics['horizon_bands'][f'{lo}-{lo+2}']={n:score(sub,n+'_prediction') for n in names}
 for month,sub in d.groupby('month',sort=True): metrics['monthly'][month]={n:score(sub,n+'_prediction') for n in names}
 metrics['blend_minus_ft_rmse_sgd']=metrics['overall']['blend']['rmse_sgd']-metrics['overall']['ft']['rmse_sgd']; metrics['blend_minus_lgb_rmse_sgd']=metrics['overall']['blend']['rmse_sgd']-metrics['overall']['lgb']['rmse_sgd']; dump('metrics.json',metrics)
 lines=['# FT + F3 LightGBM: fixed 50/50 historical diagnostic','','This 2024-04–2025-03 validation window has been repeatedly inspected. It is not an independent test and cannot demonstrate robustness. This is a single fixed-weight diagnostic; no fitting, weight search, or new training was performed.','','| Model | RMSE SGD | MAE SGD | Bias SGD |','|---|---:|---:|---:|']
 for n in names:
  s=metrics['overall'][n]; lines.append(f"| {n} | {s['rmse_sgd']:.6f} | {s['mae_sgd']:.6f} | {s['bias_sgd']:.6f} |")
 lines += ['',f"Blend − FT RMSE: {metrics['blend_minus_ft_rmse_sgd']:+.6f} SGD; blend − LGB: {metrics['blend_minus_lgb_rmse_sgd']:+.6f} SGD. Residual correlation: {metrics['residual_correlation']:.8f}. Bias is prediction − actual.", '', '| FT horizon | FT RMSE / MAE / bias | LGB RMSE / MAE / bias | Blend RMSE / MAE / bias |','|---|---:|---:|---:|']
 for band,items in metrics['horizon_bands'].items():
  cells=[f"{items[n]['rmse_sgd']:.4f} / {items[n]['mae_sgd']:.4f} / {items[n]['bias_sgd']:.4f} (n={items[n]['n']})" for n in names]; lines.append('| '+band+' | '+' | '.join(cells)+' |')
 lines+=['','| Month | FT RMSE | LGB RMSE | Blend RMSE | n |','|---|---:|---:|---:|---:|']
 for month,items in metrics['monthly'].items(): lines.append('| '+month+' | '+' | '.join(f"{items[n]['rmse_sgd']:.4f}" for n in names)+f" | {items['ft']['n']} |")
 lines+=['','FT and F3 each have 35,158 uniquely paired rows with exactly equal actuals and horizons. The F3 source contains one model/split/seed arm with no duplicate IDs; its source metrics specify 400 trees. Month comes from the F3 CSV joined by ID; it is not independently recorded in FT predictions. The FT and F3 horizon fields agree exactly and match month offsets from the F3 cutoff 2024-03.','', 'FT training used 88,192 rows, F3 114,842; their features also differ. This is not a pure architecture comparison. FT is the saved DBS/COE seed42 model. The monthly MAE and bias are in metrics.json; predictions.csv allows independent recomputation. A result on this repeatedly viewed window is exploratory; future unseen data and paired confirmation are needed.']
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n'); status('COMPLETED'); print(json.dumps(metrics['overall'],indent=2)); print('correlation',metrics['residual_correlation'],'delta',metrics['blend_minus_ft_rmse_sgd'])
if __name__=='__main__':
 try: main()
 except Exception as e: status('FAILED',reason=str(e),traceback=traceback.format_exc()); raise

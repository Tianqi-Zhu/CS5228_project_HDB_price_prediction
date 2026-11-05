"""Run with benchmark Python: exact teammate builders, encoder class AST only."""
import ast,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

ROOT=Path(__file__).resolve().parents[1]/'runtime';OUT=ROOT/'reports/concat_aligned_v20';VENDOR=Path(__file__).resolve().parents[3]

def main():
 start=time.monotonic();sys.path.insert(0,str(VENDOR))
 from analysis.experiments.frontier_search import frame_data,DistributionHistory,month_number
 from analysis.experiments.frontier_compact import ADDITIONS
 from analysis.experiments.second_wave_anniversary import AnniversaryHistory
 from analysis.experiments.third_wave_training import inputs
 from analysis.pipeline.features import model_inputs,FEATURE_CANDIDATES
 frame,hashes=frame_data();fit=frame.loc[frame.RENT_APPROVAL_DATE<'2024-04'].copy();val=frame.loc[(frame.RENT_APPROVAL_DATE>='2024-04')&(frame.RENT_APPROVAL_DATE<'2025-04')].copy()
 assert (len(fit),len(val))==(114842,35158)
 x,q=model_inputs(fit,val,FEATURE_CANDIDATES['compact_history_27'])
 dist=DistributionHistory().fit(fit)
 # Only KNN outputs omitted; they never enter this member's select_inputs.
 ef=dist.transform(fit,month_number(fit.RENT_APPROVAL_DATE),neighbors=False)
 ev=dist.transform(val,np.full(len(val),month_number(pd.Series(['2024-04']))[0]),neighbors=False)
 ann=AnniversaryHistory().fit(fit);af,av=ann.transform(fit),ann.transform(val)
 bundle={'x_fit':x,'x_val':q,'extra_fit':ef,'extra_val':ev,'ann_fit':af,'ann_val':av,'y_fit':fit.MONTHLY_RENT.to_numpy(float),'y_val':val.MONTHLY_RENT.to_numpy(float)}
 spec={'adjusted':True,'target':'residual','drop_ann36':True}
 x,q,base,offset=inputs(bundle,spec)
 raw_columns=list(x);columns=[c for c in x if x[c].notna().any() and x[c].nunique(dropna=False)>1];x,q=x[columns],q[columns]
 # Execute original encoder class without importing unused CUDA training module.
 encoder_path=VENDOR/'analysis/pipeline/neural_models.py';tree=ast.parse(encoder_path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeuralFrameEncoder')
 namespace={'np':np,'pd':pd,'QuantileTransformer':QuantileTransformer};exec(compile(ast.Module(body=[node],type_ignores=[]),str(encoder_path),'exec'),namespace)
 cats=list(x.select_dtypes(include=['category','object','string']).columns);encoder=namespace['NeuralFrameEncoder'](cats,True).fit(x)
 cf,nf=encoder.transform(x);cv,nv=encoder.transform(q)
 numeric=list(encoder.numeric_)+[c+'__missing' for c in encoder.missing_]
 # Independently replay builder expressions and poison future validation labels.
 expected=np.column_stack([ef[c] for c in ADDITIONS['relative_history']]);np.testing.assert_allclose(x[ADDITIONS['relative_history']].to_numpy(float),expected,equal_nan=True)
 ann_replay=AnniversaryHistory().fit(fit).transform(val.iloc[:25]);np.testing.assert_allclose(av.iloc[:25].to_numpy(float),ann_replay.to_numpy(float),equal_nan=True)
 poison=val.iloc[:25].copy();poison['MONTHLY_RENT']=1e9;np.testing.assert_allclose(ann.transform(poison).to_numpy(float),ann_replay.to_numpy(float),equal_nan=True)
 assert len(raw_columns)==62 and not any(c.startswith('ANN_BLOCK_36_') for c in columns)
 assert set(fit.index).isdisjoint(val.index) and np.isfinite(nf).all() and np.isfinite(nv).all()
 arrays={}
 for label,raw,n,c,a in [('train',fit,nf,cf,base),('val',val,nv,cv,offset)]:
  target=raw.MONTHLY_RENT.to_numpy(float);target_month=raw.APPROVAL_MONTH_INDEX.to_numpy(int)
  origin=target_month-1 if label=='train' else np.full(len(raw),38)
  for k,v in {'numeric':n,'categorical':c,'target':(target-a)/1000,'actual_sgd':target,'anchor_ksgd':a/1000,'row_id':raw.index.to_numpy(),'origin':origin,'horizon':target_month-origin,'month':target_month}.items():arrays[label+'_'+k]=v
  np.testing.assert_allclose((arrays[label+'_target']+arrays[label+'_anchor_ksgd'])*1000,target,atol=.001)
 path=OUT/'bank.npz';np.savez_compressed(OUT/'bank.tmp.npz',**arrays);os.replace(OUT/'bank.tmp.npz',path)
 src={str(p.relative_to(VENDOR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (VENDOR/'analysis').rglob('*.py')}
 manifest={'status':'ready','teammate_ref':json.loads((Path(__file__).resolve().parents[1]/'teammate_ref.json').read_text()),'matched_member':'ft_plr_128_3; adjusted residual; drop_ann36=True','raw_columns':raw_columns,'selected_columns':columns,'numeric_names':numeric,'categorical_names':cats,'cardinalities':encoder.cardinalities,'raw_feature_count':len(raw_columns),'encoded_numeric_count':len(numeric),'train_rows':len(fit),'validation_rows':len(val),'array_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'data_hashes':hashes,'vendor_source_hashes':src,'encoder_source_sha256':hashlib.sha256(encoder_path.read_bytes()).hexdigest(),'source_code_unmodified':True,'preprocessing':'exact NeuralFrameEncoder class; fit-only constants, vocab, medians, quantiles, missing flags','origin_policy':'teammate: train history strictly before target month; validation history fixed before2024-04','target':'same COHORT24 adjusted baseline and signed SGD residual, /1000 numerical units only','checks':['counts','train-only encoding','array finite','target reconstruction','anniversary replay','query-label poisoning','relative-field identity'],'notes':'Unused neighbors=False excludes only unused KNN fields. Static MRT is retained exactly as teammate; no point-in-time guarantee added. This is development validation, not independent test.','seconds':time.monotonic()-start}
 (OUT/'feature_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['vendor_source_hashes','data_hashes','raw_columns','numeric_names','selected_columns']}),flush=True)
 (OUT/'feature_list.txt').write_text('\n'.join(raw_columns)+'\n')
if __name__=='__main__':main()

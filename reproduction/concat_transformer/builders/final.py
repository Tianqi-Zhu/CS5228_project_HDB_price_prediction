"""Rebuild all-label fit preprocessing and unlabeled-test features locally."""
import ast,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import QuantileTransformer
ROOT=Path(__file__).resolve().parents[1]/'runtime';OUT=ROOT/'reports/concat_final_v22';VENDOR=Path(__file__).resolve().parents[3]
def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();sys.path.insert(0,str(VENDOR))
 from analysis.experiments.best_ensemble_submission import prepare
 from analysis.experiments.third_wave_training import inputs
 from analysis.pipeline.data import sha256
 bundle,hashes=prepare();fit,query=bundle['raw_fit'],bundle['raw_val'];template=pd.read_csv(ROOT/'data/example-submission.csv');rawtest=pd.read_csv(ROOT/'data/test.csv')
 assert (len(fit),len(query),len(template))==(150000,50000,50000)
 assert list(template)==['Id','Predicted'] and template.Id.is_unique
 np.testing.assert_array_equal(query.RENT_APPROVAL_DATE.to_numpy(),rawtest.RENT_APPROVAL_DATE.to_numpy())
 np.testing.assert_allclose(query.FLOOR_AREA_SQM.to_numpy(float),rawtest.FLOOR_AREA_SQM.to_numpy(float),equal_nan=True)
 x,q,base,offset=inputs(bundle,{'adjusted':True,'target':'residual','drop_ann36':True});raw_columns=list(x)
 columns=[c for c in x if x[c].notna().any() and x[c].nunique(dropna=False)>1];x,q=x[columns],q[columns]
 encoder_path=VENDOR/'analysis/pipeline/neural_models.py';tree=ast.parse(encoder_path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeuralFrameEncoder')
 namespace={'np':np,'pd':pd,'QuantileTransformer':QuantileTransformer};exec(compile(ast.Module(body=[node],type_ignores=[]),str(encoder_path),'exec'),namespace)
 cats=list(x.select_dtypes(include=['category','object','string']).columns);encoder=namespace['NeuralFrameEncoder'](cats,True).fit(x)
 cf,nf=encoder.transform(x);cq,nq=encoder.transform(q);names=list(encoder.numeric_)+[c+'__missing' for c in encoder.missing_]
 arrays={}
 for label,raw,n,c,a in [('train',fit,nf,cf,base),('test',query,nq,cq,offset)]:
  month=raw.APPROVAL_MONTH_INDEX.to_numpy(int);origin=month-1 if label=='train' else np.minimum(month,fit.APPROVAL_MONTH_INDEX.max()+1)-1
  for k,v in {'numeric':n,'categorical':c,'anchor_ksgd':a/1000,'row_id':raw.index.to_numpy(),'origin':origin,'horizon':month-origin}.items():arrays[label+'_'+k]=v
 arrays['train_target']=(bundle['y_fit']-base)/1000;arrays['train_actual_sgd']=bundle['y_fit'];arrays['test_id']=template.Id.to_numpy()
 assert len(raw_columns)==62 and np.isfinite(nf).all() and np.isfinite(nq).all() and np.isfinite(base).all() and np.isfinite(offset).all()
 np.testing.assert_allclose(1000*(arrays['train_target']+arrays['train_anchor_ksgd']),arrays['train_actual_sgd'],atol=.001)
 assert 'test_target' not in arrays and np.all(arrays['test_origin']<=fit.APPROVAL_MONTH_INDEX.max())
 np.savez_compressed(OUT/'bank.tmp.npz',**arrays);os.replace(OUT/'bank.tmp.npz',OUT/'bank.npz')
 # Dict state avoids pickling the dynamically loaded class itself. Restore with original encoder class.
 joblib.dump({'encoder_state':vars(encoder),'selected_columns':columns,'encoder_source':encoder_path.read_text(),'sklearn':__import__('sklearn').__version__},OUT/'preprocessing.joblib',compress=1)
 selection=json.loads((Path(__file__).resolve().parents[3]/'experiment/main/M05/concat_transformer/concat128x6/selection.json').read_text())
 assert selection['best_epoch']==8 and selection['reload_verified']
 manifest={'status':'ready','final_refit_authorized':True,'train_rows':150000,'test_rows':50000,'validation_rows':0,'fixed_epochs':8,'schedule_epochs':40,'numeric_names':names,'categorical_names':cats,'cardinalities':encoder.cardinalities,'raw_columns':raw_columns,'selected_columns':columns,'array_sha256':sha256(OUT/'bank.npz'),'preprocessing_sha256':sha256(OUT/'preprocessing.joblib'),'data_hashes':dict(hashes)|{'data/example-submission.csv':sha256(ROOT/'data/example-submission.csv')},'teammate_ref':json.loads((Path(__file__).resolve().parents[1]/'teammate_ref.json').read_text()),'selected_model':{'width':128,'depth':6,'heads':4,'ff_multiplier':2,'dropout':.05,'best_epoch':8,'previous_validation_rmse_sgd':selection['best_rmse_sgd']},'test_months':[query.RENT_APPROVAL_DATE.min(),query.RENT_APPROVAL_DATE.max()],'history_policy':'train histories strictly before own target month; test cut=min(targetmonth,lasttrainmonth+1), strictly earlier sources; March2025 query excludes March2025 labels','preprocessing':'refit from scratch on all150000; exact teammate definitions; same62raw features','test_target_used':False,'test_rmse':None,'seconds':time.monotonic()-start}
 (OUT/'feature_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'feature_list.txt').write_text('\n'.join(raw_columns)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ['data_hashes','raw_columns','selected_columns','numeric_names']}),flush=True)
if __name__=='__main__':main()

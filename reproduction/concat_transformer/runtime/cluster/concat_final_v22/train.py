"""Fresh all-label fixed-epoch refit; no validation targets or early stopping."""
import argparse,csv,fcntl,hashlib,json,math,os,platform,random,signal,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from cluster.concat_v21 import train as core
from experiments.concat_capacity_v21.model import make_model
from experiments.training_schedule_v15 import WarmupCosine
ROOT=Path(__file__).resolve().parents[2]

def validate_submission(ids,pred,expected=50000):
 ids=np.asarray(ids);pred=np.asarray(pred,dtype=float)
 if ids.shape!=(expected,) or pred.shape!=(expected,) or not np.array_equal(ids,np.arange(expected)) or not np.isfinite(pred).all():raise ValueError('submission IDs/count/finite check failed')
 return {'rows':expected,'min':float(pred.min()),'max':float(pred.max()),'mean':float(pred.mean()),'negative_count':int((pred<0).sum())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');ap.add_argument('--smoke',action='store_true');ap.add_argument('--stop-after-seconds',type=int,default=9000);a=ap.parse_args()
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Submit on GPU partition; CUDA unavailable')
 device=torch.device(a.device);core.DEVICE=device;torch.set_num_threads(2)
 if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 stop=[False]
 def request_stop(signum,frame):stop[0]=True
 for sig in [signal.SIGTERM,signal.SIGINT,getattr(signal,'SIGUSR1',signal.SIGTERM)]:signal.signal(sig,request_stop)
 deadline=time.monotonic()+a.stop_after_seconds;parent=ROOT/'reports/concat_final_v22';out=parent/('smoke_'+a.device if a.smoke else 'refit_w128_d6');out.mkdir(parents=True,exist_ok=True)
 with (out/'worker.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);meta=json.loads((parent/'feature_manifest.json').read_text());bank=parent/'bank.npz'
  if meta['status']!='ready' or hashlib.sha256(bank.read_bytes()).hexdigest()!=meta['array_sha256']:raise ValueError('bank checksum mismatch')
  with np.load(bank,allow_pickle=False) as f:
   if 'test_target' in f.files:raise ValueError('test targets forbidden')
   fit=SimpleNamespace(**{k[6:]:f[k].copy() for k in f.files if k.startswith('train_')});query=SimpleNamespace(**{k[5:]:f[k].copy() for k in f.files if k.startswith('test_')})
  if (len(fit.row_id),len(query.row_id))!=(150000,50000):raise ValueError('full counts mismatch')
  if a.smoke:
   fit=SimpleNamespace(**{k:v[:32] for k,v in vars(fit).items()});query=SimpleNamespace(**{k:v[:16] for k,v in vars(query).items()})
  epochs=2 if a.smoke else 8
  config={'width':128,'depth':6,'heads':4,'ff_multiplier':2,'dropout':.05,'batch':512,'lr':3e-4,'weight_decay':1e-4,'seed':42,'fixed_epochs':epochs,'schedule_epochs':40,'warmup_epochs':1,'validation_rows':0,'early_stopping':False,'smoke':a.smoke,'device_type':device.type,'precision':'float32,TF32off','test_target_used':False,'final_refit':True}
  paths=['cluster/concat_final_v22/train.py','cluster/concat_v21/train.py','experiments/concat_capacity_v21/model.py','experiments/feature_transformer_v4/models.py','experiments/feature_transformer_v4/macro_v4.py','experiments/training_schedule_v15.py']
  hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths};identity=core.fingerprint(meta,config,hashes)
  core.atomic_json({'identity':identity,'configuration':config,'metadata':meta,'source_hashes':hashes,'environment':{'torch':str(torch.__version__),'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None}},out/'protocol.json')
  core.seed();model=make_model(meta['numeric_names'],meta['cardinalities'],128,6).to(device)
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);steps=math.ceil(len(fit.row_id)/512);scheduler=WarmupCosine(opt,total_steps=steps*40,warmup_steps=steps,min_lr_ratio=.01,start_ratio=.01)
  epoch,offset,total_loss,seen,history=1,0,0.,0,[];checkpoint=out/'progress.pt'
  if checkpoint.exists():
   s=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if s['identity']!=identity:raise RuntimeError('resume identity mismatch')
   model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
   if device.type=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
   epoch,offset,total_loss,seen,history=(s[k] for k in ['epoch','offset','total_loss','seen','history']);core.validate_resume(epoch,offset,len(fit.row_id),512,scheduler.updates_completed)
  def save(complete=False):
   core.atomic_save({'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'epoch':epoch,'offset':offset,'total_loss':total_loss,'seen':seen,'history':history,'complete':complete},checkpoint)
  core.atomic_json({'status':'training','epoch':epoch,'fixed_epochs':epochs,'pid':os.getpid()},out/'status.json')
  while epoch<=epochs:
   started=time.monotonic();model.train();order=core.epoch_order(len(fit.row_id),epoch)
   while offset<len(order):
    if stop[0] or time.monotonic()>=deadline:
     save();core.atomic_json({'status':'paused','epoch':epoch,'rows_seen':offset,'resume':'resubmit same final-refit job'},out/'status.json');return 75
    ix=order[offset:offset+512];target=torch.as_tensor(fit.target[ix],dtype=torch.float32,device=device)
    loss,norm,lr=core.train_update(model,opt,scheduler,core.make_batch(fit,ix,None),target);total_loss+=loss*len(ix);seen+=len(ix);offset+=len(ix)
    if math.ceil(offset/512)%25==0 or offset==len(order):
     save();record={'status':'training','epoch':epoch,'fixed_epochs':epochs,'batch':math.ceil(offset/512),'batches_per_epoch':steps,'rows_seen':offset,'training_mse_ksgd':loss,'learning_rate':lr,'gradient_norm':norm};core.atomic_json(record,out/'progress.json');print(json.dumps(record),flush=True)
   history.append({'epoch':epoch,'training_mse_ksgd':total_loss/seen,'seconds':time.monotonic()-started,'supervised_rows':seen});print(json.dumps(history[-1]),flush=True)
   epoch+=1;offset=0;total_loss=0.;seen=0;save()
  if scheduler.updates_completed!=epochs*steps:raise RuntimeError('fixed epoch update count mismatch')
  core.atomic_json({'status':'predicting','train_epochs':epochs,'test_rows':len(query.row_id)},out/'status.json')
  residual=core.predict(model,query,None);pred=1000*(residual+query.anchor_ksgd);summary=validate_submission(query.id,pred,len(query.row_id))
  core.atomic_save({'identity':identity,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'configuration':config,'numeric_names':meta['numeric_names'],'cardinalities':meta['cardinalities'],'array_sha256':meta['array_sha256'],'preprocessing_sha256':meta['preprocessing_sha256']},out/'final_model.pt')
  reloaded=make_model(meta['numeric_names'],meta['cardinalities'],128,6).to(device);reloaded.load_state_dict(torch.load(out/'final_model.pt',map_location='cpu',weights_only=False)['model']);np.testing.assert_allclose(core.predict(reloaded,query,None),residual,rtol=1e-6,atol=1e-6)
  with (out/'submission.csv.tmp').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['Id','Predicted']);w.writerows(zip(query.id,pred))
  os.replace(out/'submission.csv.tmp',out/'submission.csv')
  with (out/'submission.csv').open() as f:rows=list(csv.DictReader(f))
  validate_submission([int(r['Id']) for r in rows],[float(r['Predicted']) for r in rows],len(query.id));np.testing.assert_allclose([float(r['Predicted']) for r in rows],pred,rtol=1e-12,atol=1e-8)
  report={'status':'completed','identity':identity,'train_rows':len(fit.row_id),'fixed_epochs':epochs,'history':history,'test_rows':len(query.id),'submission_columns':['Id','Predicted'],'submission_sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(),'reload_verified':True,'test_rmse':None,'previous_selection_rmse':meta['selected_model']['previous_validation_rmse_sgd'],'previous_selection_is_not_test_score':True,'summary':summary,'final_refit_performed':not a.smoke,'smoke':a.smoke}
  save(True);core.atomic_json(report,out/'final_report.json');core.atomic_json({'status':'completed','test_rows':len(query.id),'fixed_epochs':epochs},out/'status.json');print(json.dumps(report),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())

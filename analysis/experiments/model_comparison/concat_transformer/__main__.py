"""Explicit prepare/train/predict modes; help and verify never train."""
import argparse,sys,subprocess,os,json,csv,hashlib
from pathlib import Path
from . import ROOT,runtime_path
BASE=ROOT/'reproduction/concat_transformer'
def verify():
 manifest=json.loads((BASE/'manifest.json').read_text());bad=[]
 for name,digest in manifest['files'].items():
  p=ROOT/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(name)
 if bad:raise ValueError('Checksum mismatch: '+str(bad))
 with (BASE/'artifacts/submission.csv').open() as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['Id','Predicted'];rows=list(reader)
 import math
 assert len(rows)==50000 and [int(r['Id']) for r in rows]==list(range(50000)) and all(math.isfinite(float(r['Predicted'])) for r in rows)
 print(json.dumps({'status':'passed','files_checked':len(manifest['files']),'submission_rows':50000,'training_performed':False}))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','predict','prepare','select','refit']);p.add_argument('--stage',choices=['selection','final'],default='final');p.add_argument('--candidate',choices=['w128_d3','w128_d6'],default='w128_d6');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--python',default=sys.executable,help='Python for feature preparation');p.add_argument('--output',type=Path,default=ROOT/'output/concat_transformer/replayed_submission.csv');a=p.parse_args()
 if a.mode=='verify':return verify()
 runtime=runtime_path()
 if a.mode=='predict':
  verify();import numpy as np,torch
  sys.path.insert(0,str(runtime));from experiments.concat_capacity_v21.model import make_model
  if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose --device cpu')
  device=torch.device(a.device);torch.set_num_threads(2)
  if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  ck=torch.load(BASE/'artifacts/final_model.pt',map_location='cpu',weights_only=False);model=make_model(ck['numeric_names'],ck['cardinalities'],128,6).to(device);model.load_state_dict(ck['model']);model.eval();output=[]
  with np.load(BASE/'artifacts/bank.npz') as bank:
   with torch.inference_mode():
    for i in range(0,50000,512):
     batch={'numeric':torch.as_tensor(bank['test_numeric'][i:i+512],device=device),'categorical':torch.as_tensor(bank['test_categorical'][i:i+512],device=device)}
     output.extend((1000*(model(batch).cpu().numpy()+bank['test_anchor_ksgd'][i:i+512])).tolist())
  with (BASE/'artifacts/submission.csv').open() as f:original=np.array([float(r['Predicted']) for r in csv.DictReader(f)])
  delta=float(np.max(np.abs(np.asarray(output)-original)));np.testing.assert_allclose(output,original,rtol=1e-4,atol=.10)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  if a.output.resolve()==(BASE/'artifacts/submission.csv').resolve():raise ValueError('Do not overwrite frozen CSV')
  with a.output.open('w',newline='') as f:w=csv.writer(f);w.writerow(['Id','Predicted']);w.writerows(enumerate(output))
  report={'rows':50000,'max_abs_diff_sgd':delta,'training_performed':False,'output':str(a.output),'device':str(device)};a.output.with_suffix('.verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
 env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 if a.mode=='prepare':
  data=runtime/'data'
  if not data.exists():data.symlink_to(ROOT/'data',target_is_directory=True)
  subprocess.run([a.python,str(BASE/'builders'/f'{a.stage}.py')],cwd=ROOT,env=env,check=True);return
 module='cluster.concat_v21.train' if a.mode=='select' else 'cluster.concat_final_v22.train'
 args=[sys.executable,'-u','-m',module,'--device',a.device]
 if a.mode=='select':args+=['--candidate',a.candidate]
 subprocess.run(args,cwd=runtime,env=env,check=True)
if __name__=='__main__':main()

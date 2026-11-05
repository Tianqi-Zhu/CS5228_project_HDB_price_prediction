"""Bounded v18 matched controls, waits for v17 low-LR queue to finish."""
import fcntl,hashlib,itertools,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/market_anchor_v18/runs';V17=ROOT/'reports/spatiotemporal_v17/models/causal_no_mrt'
ANCHORS=('original','refreshed_12m','market_level')
def task_list(): return list(itertools.product(ANCHORS,('direct','residual'),('ridge','lightgbm')))
def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def worker(anchor,mode,learner):
 import numpy as np
 from experiments.spatiotemporal_v17.contract import load_bank
 from experiments.market_anchor_v18.contract import load_anchor
 from experiments.market_anchor_v18.train import run
 bank=load_bank('F0',variant='causal_no_mrt'); anchors=load_anchor(anchor)
 f0=ROOT/'reports/spatiotemporal_v17/features/orig/screen/causal_no_mrt/F0/arrays.npz'; ap=ROOT/'reports/market_anchor_v18/data'/anchor/'arrays.npz'
 if digest(f0)!=bank['manifest']['array_sha256'] or digest(ap)!=anchors['manifest']['array_sha256'] or anchors['manifest']['v17_bank_sha256']!=bank['manifest']['array_sha256']: raise RuntimeError('bank or anchor hash mismatch')
 if [len(bank[x]['row_id']) for x in ('train','val')]!=[30000,5000]: raise RuntimeError('unexpected split sizes')
 b3=load_bank('F3',variant='causal_no_mrt'); idx=b3['categorical_names'].index('BLOCK_ID')
 for split in ('train','val'):
  if not np.array_equal(bank[split]['row_id'],b3[split]['row_id']): raise RuntimeError('block diagnostic row mismatch')
  bank[split]['block_known']=b3[split]['x_cat'][:,idx]!=0
 run(bank,anchors,learner,mode,OUT/f'{anchor}__{mode}__{learner}')

def supervise():
 OUT.mkdir(parents=True,exist_ok=True);status={'pid':os.getpid(),'status':'waiting_for_v17_lowlr','child_pid':None,'tasks':task_list()}; lockfile=(V17/'queue.lock').open('a+')
 while True:
  atomic_json(OUT/'queue_status.json',status)
  try: fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: time.sleep(30);continue
  try:
   prev=json.loads((V17/'lowlr/queue_status.json').read_text())
   if prev.get('status')!='completed': status['status']='blocked_v17_incomplete'; fcntl.flock(lockfile,fcntl.LOCK_UN);time.sleep(30);continue
   break
  except Exception:
   fcntl.flock(lockfile,fcntl.LOCK_UN); raise
 schema=json.loads((ROOT/'reports/market_anchor_v18/data/schema.json').read_text())
 if schema.get('status')!='ready': raise RuntimeError('anchor audit not ready')
 paths=[Path(__file__),Path(__file__).with_name('train.py'),Path(__file__).with_name('residual.py'),Path(__file__).with_name('evaluate.py')]
 hashes={str(p):digest(p) for p in paths};status.update(status='starting',source_hashes=hashes)
 for anchor,mode,learner in task_list():
  if any(digest(Path(p))!=h for p,h in hashes.items()): raise RuntimeError('source changed')
  name=f'{anchor}__{mode}__{learner}'; env=os.environ.copy();env.update(DYLD_FALLBACK_LIBRARY_PATH='/opt/anaconda3/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
  with (OUT/(name+'.log')).open('ab',buffering=0) as log:
   proc=subprocess.Popen([sys.executable,'-u','-m','experiments.market_anchor_v18.queue','--worker',anchor,mode,learner],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   status.update(status='training',candidate=name,child_pid=proc.pid);atomic_json(OUT/'queue_status.json',status);code=proc.wait()
  if code: raise RuntimeError(f'{name} exited {code}')
  if not json.loads((OUT/name/'result.json').read_text()).get('complete'): raise RuntimeError('missing complete result')
 status.update(status='completed',candidate=None,child_pid=None);atomic_json(OUT/'queue_status.json',status)

if __name__=='__main__':
 if len(sys.argv)==5 and sys.argv[1]=='--worker':
  task=tuple(sys.argv[2:]);
  if task not in task_list():raise ValueError('unregistered task')
  worker(*task)
 else:
  try:supervise()
  except Exception as e:atomic_json(OUT/'queue_status.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()});raise

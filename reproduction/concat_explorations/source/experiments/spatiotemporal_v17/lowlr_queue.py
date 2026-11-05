"""F0 learning-rate diagnostics, sequential and isolated; no final refit."""
import fcntl,hashlib,json,os,subprocess,sys,traceback
from experiments.concat_warmup_v15.pipeline import atomic_json
from experiments.spatiotemporal_v17.queue import ROOT,OUT as ORIGINAL,lock_is_held
from experiments.spatiotemporal_v17.followup_queue import check_bank
from experiments.spatiotemporal_v17.contract import load_bank
OUT=ORIGINAL/'lowlr'
TASKS={'static_1e-4':('static',1e-4),'cross_1e-4':('cross',1e-4),'static_3e-5':('static',3e-5),'cross_3e-5':('cross',3e-5)}

def task_command(name):
    if name not in TASKS: raise ValueError('unregistered task')
    arch,lr=TASKS[name]
    return [sys.executable,'-u','-m','experiments.spatiotemporal_v17.train','F0','--architecture',arch,'--lr',str(lr),'--diagnostics','--output',str(OUT/name)]

def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    with (ORIGINAL/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=json.loads((ORIGINAL/'followup/queue_status.json').read_text())
        if prior.get('status')!='completed': raise RuntimeError('prior followup incomplete')
        if any(lock_is_held(ROOT/f'reports/overnight_hdb_v{x}/queue.lock') for x in (14,15)) or lock_is_held(ROOT/'reports/concat_depth_v16/supervisor.lock'): raise RuntimeError('prior queue active')
        bank=load_bank('F0',variant='causal_no_mrt'); check_bank('F0',bank)
        for path in (ORIGINAL/'F0/config.json',ORIGINAL/'followup/F0_cross/config.json'):
            if json.loads(path.read_text())['manifest']['array_sha256']!=bank['manifest']['array_sha256']: raise RuntimeError('baseline bank differs')
        paths=[__file__,str(ROOT/'experiments/spatiotemporal_v17/train.py'),str(ROOT/'experiments/spatiotemporal_v17/models.py')]
        hashes={p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in paths}
        status={'status':'starting','pid':os.getpid(),'tasks':list(TASKS),'source_hashes':hashes,'child_pid':None}
        for task in TASKS:
            if any(hashlib.sha256(open(p,'rb').read()).hexdigest()!=h for p,h in hashes.items()): raise RuntimeError('runner source changed')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (OUT/(task+'.log')).open('ab',buffering=0) as log:
                proc=subprocess.Popen(task_command(task),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status.update(status='training',candidate=task,child_pid=proc.pid); atomic_json(status,OUT/'queue_status.json'); code=proc.wait()
            if code: raise RuntimeError(f'{task} exited {code}')
            if not json.loads((OUT/task/'result.json').read_text()).get('reload_verified'): raise RuntimeError('incomplete result')
        status.update(status='completed',candidate=None,child_pid=None); atomic_json(status,OUT/'queue_status.json')

if __name__=='__main__':
    try: supervise()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True); atomic_json({'status':'failed','error':repr(exc),'traceback':traceback.format_exc()},OUT/'queue_status.json'); raise

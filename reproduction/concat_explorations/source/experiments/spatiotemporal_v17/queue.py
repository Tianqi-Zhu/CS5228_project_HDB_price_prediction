"""Single local screen supervisor, run under caffeinate -i; waits for v16."""
import fcntl, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
from experiments.concat_warmup_v15.pipeline import atomic_json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/spatiotemporal_v17/models/causal_no_mrt'
PRIOR=ROOT/'reports/concat_depth_v16'
ORDER=('F1','F0','F3','F4','F5','F2')

def pid_state(pid):
    if not isinstance(pid,int) or pid<=0: return 'unknown'
    try: os.kill(pid,0)
    except ProcessLookupError: return 'absent'
    except PermissionError: return 'unknown'
    return 'present'

def prior_state(status, locked, processes=None):
    if locked: return 'waiting'
    if status=='completed': return 'ready'
    if status=='running' and processes and len(processes)==2 and all(p=='absent' for p in processes): return 'ready_stale'
    return 'blocked'

def lock_is_held(path):
    with path.open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(f,fcntl.LOCK_UN); return False

def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={'status':'waiting_for_v16','pid':os.getpid(),'order':ORDER,'child_pid':None}
        while True:
            previous=json.loads((PRIOR/'status.json').read_text()); probes=[pid_state(previous.get('pid')),pid_state(previous.get('child_pid'))]; state=prior_state(previous.get('status'),lock_is_held(PRIOR/'supervisor.lock'),probes)
            status['prior_process_probes']=probes
            status['prior_status']=previous; atomic_json(status,OUT/'queue_status.json')
            if state in ('ready','ready_stale') and not lock_is_held(ROOT/'reports/overnight_hdb_v15/queue.lock') and not lock_is_held(ROOT/'reports/overnight_hdb_v14/queue.lock'):
                status['prior_resolution']=state; break
            if state=='blocked':
                status['status']='blocked_prior_inconsistent'
                atomic_json(status,OUT/'queue_status.json')
            time.sleep(30)
        schema=ROOT/'reports/spatiotemporal_v17/features/schema.json'
        while json.loads(schema.read_text()).get('causal_no_mrt_status') != 'ready':
            status.update(status='blocked_feature_audit',child_pid=None)
            atomic_json(status,OUT/'queue_status.json')
            time.sleep(30)
        from experiments.spatiotemporal_v17.contract import load_bank
        from experiments.spatiotemporal_v17.train import validate_bank
        import numpy as np
        reference=load_bank(ORDER[0],variant='causal_no_mrt'); validate_bank(reference)
        for candidate in ORDER[1:]:
            other=load_bank(candidate,variant='causal_no_mrt'); validate_bank(other)
            if any(not np.array_equal(reference[split]['row_id'],other[split]['row_id']) for split in ('train','val')):
                raise RuntimeError('candidate row IDs differ')
        paths=[Path(__file__),ROOT/'experiments/spatiotemporal_v17/train.py',ROOT/'experiments/spatiotemporal_v17/features.py',ROOT/'experiments/spatiotemporal_v17/contract.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in ORDER:
            if any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)] for p in paths): raise RuntimeError('runner or feature source changed during queue')
            status.update(status='starting',candidate=name,child_pid=None); atomic_json(status,OUT/'queue_status.json')
            env=os.environ.copy(); env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (OUT/(name+'.log')).open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','experiments.spatiotemporal_v17.train',name],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status.update(status='training',child_pid=proc.pid); atomic_json(status,OUT/'queue_status.json'); code=proc.wait()
            if code: raise RuntimeError(f'{name} exited {code}')
            result=json.loads((OUT/name/'result.json').read_text())
            if not result.get('reload_verified'): raise RuntimeError(f'{name} result incomplete')
        status.update(status='completed',candidate=None,child_pid=None); atomic_json(status,OUT/'queue_status.json')

if __name__=='__main__':
    try: supervise()
    except Exception as exc:
        atomic_json({'status':'failed','error':repr(exc),'traceback':traceback.format_exc()},OUT/'queue_status.json'); raise

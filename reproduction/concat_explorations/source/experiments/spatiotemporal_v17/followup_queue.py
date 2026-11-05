"""Sequential matched-input exploratory followups; run beneath caffeinate."""
import fcntl, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from experiments.concat_warmup_v15.pipeline import atomic_json
from experiments.spatiotemporal_v17.queue import ROOT, OUT as ORIGINAL, lock_is_held
from experiments.spatiotemporal_v17.contract import load_bank
from experiments.spatiotemporal_v17.train import validate_bank
OUT=ORIGINAL/'followup'
ORDER=('F0_concat','F0_cross','F4_support','F3_no_block')

def task_command(name):
    if name not in ORDER: raise ValueError('unregistered task')
    bank,arch=('F0',name.split('_')[1]) if name.startswith('F0_') else (name,'static')
    return [sys.executable,'-u','-m','experiments.spatiotemporal_v17.train',bank,'--architecture',arch,'--output',str(OUT/name)]

def check_bank(name, reference):
    bank=load_bank(name,variant='causal_no_mrt'); validate_bank(bank)
    base=ROOT/'reports/spatiotemporal_v17/features/orig/screen/causal_no_mrt'/name
    digest=hashlib.sha256((base/'arrays.npz').read_bytes()).hexdigest()
    if digest!=bank['manifest']['array_sha256']: raise RuntimeError('bank hash mismatch')
    for split in ('train','val'):
        for key in ('row_id','y_sgd','origin','horizon','anchor_ksgd'):
            if not np.array_equal(reference[split][key],bank[split][key]): raise RuntimeError('bank row contract mismatch')
    return bank

def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    with (ORIGINAL/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads((ORIGINAL/'queue_status.json').read_text())
        if old.get('status')!='completed': raise RuntimeError('initial queue incomplete')
        if any(lock_is_held(ROOT/f'reports/overnight_hdb_v{x}/queue.lock') for x in (14,15)) or lock_is_held(ROOT/'reports/concat_depth_v16/supervisor.lock'): raise RuntimeError('prior queue active')
        reference=load_bank('F0',variant='causal_no_mrt'); check_bank('F0',reference)
        baseline=json.loads((ORIGINAL/'F0/config.json').read_text())
        if baseline['manifest']['array_sha256']!=reference['manifest']['array_sha256']: raise RuntimeError('F0 differs from static baseline')
        paths=[Path(__file__),ROOT/'experiments/spatiotemporal_v17/train.py',ROOT/'experiments/spatiotemporal_v17/models.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        status={'status':'starting','pid':os.getpid(),'tasks':ORDER,'source_hashes':hashes,'child_pid':None}
        for task in ORDER:
            if task in ('F4_support','F3_no_block'):
                schema=ROOT/'reports/spatiotemporal_v17/features/schema.json'
                while json.loads(schema.read_text()).get('followup_fixes_status')!='ready':
                    status.update(status='waiting_feature_audit',candidate=task,child_pid=None); atomic_json(status,OUT/'queue_status.json'); time.sleep(30)
                check_bank(task,reference)
            if any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)] for p in paths): raise RuntimeError('runner sources changed')
            env=os.environ.copy(); env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
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

"""Sequential full-data depth/dropout selection, no full-label refit."""
import argparse, fcntl, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/concat_depth_v16'
PREVIOUS=ROOT/'reports/concat_warmup_v15/selection.json'
QUEUE=ROOT/'reports/overnight_hdb_v15'
CANDIDATES=[(3,.15),(4,.05),(4,.15),(6,.05),(6,.15)]

def candidate_name(layers,dropout): return f'layers_{layers}_dropout_{dropout:.2f}'

def validated_lr(selection):
    lr=selection.get('selected_lr')
    if selection.get('verified_reload') is not True or selection.get('final_refit_performed') is not False or lr not in (1e-3,5e-4,3e-4):
        raise ValueError('previous selection not verified or has invalid learning rate')
    return lr

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(tmp,path)

def configure(layers,dropout,lr):
    if (layers,dropout) not in CANDIDATES or lr not in (1e-3,5e-4,3e-4): raise ValueError('candidate not registered')
    from experiments.concat_warmup_v15 import pipeline as p
    p.OUT=OUT/candidate_name(layers,dropout); p.LRS=(lr,)
    old_hashes=p.source_hashes
    config={'layers':layers,'dropout':dropout,'selected_lr':lr,'seed':p.SEED}
    def hashes():
        return {**old_hashes(),'experiments/concat_depth_v16/depth.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'candidate_config':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}
    def build(names,cardinalities,source=None):
        base=p.FeatureTransformer(names,cardinalities,d_model=48,layers=layers,heads=4,ff_multiplier=2,dropout=dropout,pooling='cls')
        return p.MacroOnFeatureTransformer(base,source if source is not None else p.NativeSource(),'both')
    p.source_hashes=hashes; p.build_model=build
    return p

def prior_queue_active():
    with (QUEUE/'queue.lock').open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(f,fcntl.LOCK_UN); return False

def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={'status':'waiting_for_learning_rate_search','pid':os.getpid(),'final_refit_performed':False}
        atomic_json(OUT/'status.json',status)
        while True:
            previous=json.loads((QUEUE/'status.json').read_text())
            if previous.get('status')=='selection_jobs_finished': break
            if not prior_queue_active(): raise RuntimeError('previous supervisor stopped before verified selection completed')
            time.sleep(10)
        baseline=json.loads(PREVIOUS.read_text()); lr=validated_lr(baseline)
        original_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        summaries=[]
        for layers,dropout in CANDIDATES:
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=original_sha: raise RuntimeError('depth runner source changed')
            name=candidate_name(layers,dropout); path=OUT/name/'selection.json'
            status.update(status='running',candidate=name,selected_lr=lr,completed=len(summaries)); atomic_json(OUT/'status.json',status)
            env=os.environ.copy(); env.pop('PYTHONPATH',None); env.pop('PYTHONSTARTUP',None)
            env.update(DYLD_FALLBACK_LIBRARY_PATH='/opt/anaconda3/lib',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (OUT/(name+'.log')).open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','experiments.concat_depth_v16.depth','--candidate',str(layers),str(dropout),'--lr',str(lr),'--selection-only'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status['child_pid']=proc.pid; atomic_json(OUT/'status.json',status)
                code=proc.wait()
            status['child_pid']=None
            if code: raise RuntimeError(f'{name} exited {code}')
            result=json.loads(path.read_text()); validated_lr(result)
            summaries.append({'layers':layers,'dropout':dropout,'rmse_sgd':result['selection_rmse_sgd'],'epoch':result['selected_epoch'],'report':str(path.relative_to(ROOT))})
        all_results=[{'layers':3,'dropout':.05,'rmse_sgd':baseline['selection_rmse_sgd'],'epoch':baseline['selected_epoch'],'report':str(PREVIOUS.relative_to(ROOT))}]+summaries
        atomic_json(OUT/'comparison.json',{'selected_lr':lr,'results':all_results,'winner':min(all_results,key=lambda x:x['rmse_sgd']),'final_refit_performed':False,'metric':'development validation RMSE SGD'})
        status.update(status='completed',candidate=None,completed=len(summaries),child_pid=None); atomic_json(OUT/'status.json',status)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--candidate',nargs=2); parser.add_argument('--lr',type=float); parser.add_argument('--selection-only',action='store_true'); args=parser.parse_args()
    if args.candidate:
        if not args.selection_only or args.lr is None: parser.error('candidate requires --selection-only and --lr')
        configure(int(args.candidate[0]),float(args.candidate[1]),args.lr).main(['--selection-only'])
    else:
        try: supervise()
        except Exception as e:
            atomic_json(OUT/'error.json',{'error':repr(e),'traceback':traceback.format_exc()})
            atomic_json(OUT/'status.json',{'status':'failed','error':repr(e),'final_refit_performed':False}); raise

if __name__=='__main__': main()

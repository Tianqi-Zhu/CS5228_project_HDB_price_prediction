"""Device, checkpoint, stop, and lock primitives for the isolated SoC run."""
import contextlib,fcntl,json,os,signal,time,importlib.metadata,platform
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]

def resolve_device(name):
    if name=='cpu': return torch.device('cpu')
    if name!='cuda': raise ValueError('device must be cpu or cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but no CUDA device is available')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA device does not support BF16')
    return torch.device('cuda',torch.cuda.current_device())

def move_tree(value,device):
    if torch.is_tensor(value): return value.detach().to(device)
    if isinstance(value,dict): return {k:move_tree(v,device) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(move_tree(v,device) for v in value)
    if isinstance(value,list): return [move_tree(v,device) for v in value]
    return value

def capture_rng(device):
    return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}

def restore_rng(state,device):
    torch.set_rng_state(state['cpu'].cpu())
    if device.type=='cuda':
        if len(state['cuda'])!=torch.cuda.device_count(): raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,default=str)+'\n'); os.replace(tmp,path)

def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); torch.save(move_tree(value,torch.device('cpu')),tmp); os.replace(tmp,path)

@contextlib.contextmanager
def run_lock(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('output directory already in use') from exc
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

class StopController:
    def __init__(self,stop_after_seconds=None):
        self.deadline=None if stop_after_seconds is None else time.monotonic()+stop_after_seconds
        self.signaled=False
    def install(self):
        def stop(signum,frame): self.signaled=True
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGUSR1,stop)
    def requested(self): return self.signaled or (self.deadline is not None and time.monotonic()>=self.deadline)

def device_identity(device):
    versions={}
    for name in ('numpy','pandas','scipy','scikit-learn','safetensors','huggingface-hub'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    result={'type':device.type,'torch':str(torch.__version__),'python':platform.python_version(),'libraries':versions,'cudnn_version':torch.backends.cudnn.version(),
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),
            'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark':bool(torch.backends.cudnn.benchmark)}
    if device.type=='cuda': result.update(name=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),cuda_runtime=torch.version.cuda,bf16=torch.cuda.is_bf16_supported())
    return result

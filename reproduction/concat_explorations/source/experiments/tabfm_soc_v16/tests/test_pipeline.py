import numpy as np
import torch
from experiments.tabfm_soc_v16 import pipeline as p
from experiments.tabfm_soc_v16.runtime import StopController


def test_cpu_toy_pause_resume_restores_schedule_rng_and_optimizer(tmp_path,monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.))
    monkeypatch.setattr(p,'OUT',tmp_path); monkeypatch.setattr(p,'DEVICE',torch.device('cpu')); monkeypatch.setattr(p,'STOP',StopController())
    monkeypatch.setattr(p,'model_init',Tiny); monkeypatch.setattr(p,'cache',lambda *a:None); monkeypatch.setattr(p,'cast_cap',lambda x:x)
    monkeypatch.setattr(p,'raw_parity_own',lambda *a:None)
    monkeypatch.setattr(p,'cached_forward',lambda model,cap:model.weight.reshape(1,1,1).expand(1,2,1))
    ep={'train_size':1,'query_targets':np.array([0.]),'target_mean':0.,'target_std':1.,'query_row_id':[1]}; train=[ep,ep]
    a=p.train_run('selection','test',1e-4,train,[],'fp',40,True,max_new_steps=1)
    first=torch.load(tmp_path/'selection/test/resume.pt',weights_only=False)
    assert a['status']=='paused_for_queue' and first['scheduler']['updates_completed']==1 and first['next_step']==1
    assert first['params']['weight'].item()!=1. and 'cpu' in first['rng'] and first['rng']['cuda']==[]
    b=p.train_run('selection','test',1e-4,train,[],'fp',40,True,max_new_steps=1)
    second=torch.load(tmp_path/'selection/test/resume.pt',weights_only=False)
    assert b['status']=='paused_for_queue' and second['scheduler']['updates_completed']==2 and second['next_step']==2
    assert second['optimizer']['state'] and second['params']['weight'].item()!=first['params']['weight'].item()


def test_stop_before_first_step_writes_resumable_checkpoint(tmp_path,monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.))
    monkeypatch.setattr(p,'OUT',tmp_path); monkeypatch.setattr(p,'DEVICE',torch.device('cpu')); monkeypatch.setattr(p,'STOP',StopController(0)); monkeypatch.setattr(p,'model_init',Tiny)
    result=p.train_run('selection','test',1e-4,[{},{}],[],'fp',40,True)
    state=torch.load(tmp_path/'selection/test/resume.pt',weights_only=False)
    assert result['status']=='paused_for_queue' and state['next_step']==0 and state['scheduler']['updates_completed']==0


def test_validation_interruption_keeps_completed_epoch_unadvanced(tmp_path,monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.))
    class DuringValidation:
        def __init__(self): self.calls=0
        def requested(self): self.calls+=1; return self.calls>=3
    monkeypatch.setattr(p,'OUT',tmp_path); monkeypatch.setattr(p,'DEVICE',torch.device('cpu')); monkeypatch.setattr(p,'STOP',DuringValidation())
    monkeypatch.setattr(p,'model_init',Tiny); monkeypatch.setattr(p,'cache',lambda *args:None); monkeypatch.setattr(p,'cast_cap',lambda x:x); monkeypatch.setattr(p,'raw_parity_own',lambda *args:None)
    monkeypatch.setattr(p,'cached_forward',lambda model,cap:model.weight.reshape(1,1,1).expand(1,2,1))
    ep={'train_size':1,'query_targets':np.array([0.]),'target_mean':0.,'target_std':1.,'query_row_id':[1]}
    result=p.train_run('selection','test',1e-4,[ep],[ep],'fp',40,True)
    state=torch.load(tmp_path/'selection/test/resume.pt',weights_only=False)
    assert result['reason']=='validation_interrupted' and state['epoch']==1 and state['next_step']==1
    assert state['scheduler']['updates_completed']==1 and state['curve']==[]


def test_cli_pause_code_and_lock_does_not_damage_status(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.tabfm_soc_v16.runtime import run_lock
    import json,pytest
    args=SimpleNamespace(output_dir=str(tmp_path),checkpoint_dir=str(tmp_path),preflight_report=str(tmp_path/'preflight.json'),device='cpu',stop_after_seconds=None,max_new_steps=1)
    def paused(_): p.runner.atomic_json(tmp_path/'status.json',{'status':'paused_for_queue'})
    monkeypatch.setattr(p,'main',paused)
    assert p.run_cli(args)==75
    assert json.loads((tmp_path/'status.json').read_text())['status']=='paused_for_queue'
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError): p.run_cli(args)
    assert json.loads((tmp_path/'status.json').read_text())['status']=='paused_for_queue'

import torch
from experiments.tabfm_warmup_v15.pipeline import make_schedule


def test_schedule_resume_preserves_next_learning_rate():
    x=torch.nn.Parameter(torch.tensor(1.0)); opt=torch.optim.AdamW([x],lr=1e-4)
    sched=make_schedule(opt,steps_per_epoch=4)
    first=opt.param_groups[0]['lr']
    for _ in range(3):
        opt.zero_grad(); x.square().backward(); opt.step(); sched.step()
    state=sched.state_dict(); optstate=opt.state_dict(); expected=opt.param_groups[0]['lr']
    y=torch.nn.Parameter(torch.tensor(1.0)); other=torch.optim.AdamW([y],lr=1e-4); resumed=make_schedule(other,steps_per_epoch=4); other.load_state_dict(optstate); resumed.load_state_dict(state)
    assert first < expected <= 1e-4
    assert other.param_groups[0]['lr']==expected
    opt.step(); sched.step(); other.step(); resumed.step()
    assert other.param_groups[0]['lr']==opt.param_groups[0]['lr']


def test_reject_inconsistent_resume_step_count():
    import pytest
    from experiments.tabfm_warmup_v15.pipeline import validate_schedule_position
    validate_schedule_position({'updates_completed':9},epoch=3,next_step=1,steps_per_epoch=4)
    with pytest.raises(ValueError):
        validate_schedule_position({'updates_completed':8},epoch=3,next_step=1,steps_per_epoch=4)


def test_training_pause_resume_persists_gradient_and_schedule(tmp_path,monkeypatch):
    from experiments.tabfm_warmup_v15 import pipeline as p
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.))
    active=[]
    def create():
        model=Tiny(); active.append(model); return model
    monkeypatch.setattr(p,'OUT',tmp_path)
    monkeypatch.setattr(p,'model_init',create)
    monkeypatch.setattr(p,'cache',lambda *args:None)
    monkeypatch.setattr(p,'cast_cap',lambda x:x)
    monkeypatch.setattr(p,'raw_parity_own',lambda *args:None)
    monkeypatch.setattr(p,'cached_forward',lambda model,cap:model.weight.reshape(1,1,1).expand(1,2,1))
    ep={'train_size':1,'query_targets':__import__('numpy').array([0.]),'target_mean':0.,'target_std':1.,'query_row_id':[1]}
    train=[ep,ep]
    first=p.train_run('selection','test',1e-4,train,[],'fingerprint',40,True,max_new_steps=1)
    saved=torch.load(tmp_path/'selection/test/resume.pt',weights_only=False)
    assert first['status']=='paused_for_queue' and saved['next_step']==1
    assert saved['scheduler']['updates_completed']==1 and saved['params']['weight'].item()!=1.
    second=p.train_run('selection','test',1e-4,train,[],'fingerprint',40,True,max_new_steps=1)
    saved=torch.load(tmp_path/'selection/test/resume.pt',weights_only=False)
    assert second['status']=='paused_for_queue' and saved['next_step']==2
    assert saved['scheduler']['updates_completed']==2

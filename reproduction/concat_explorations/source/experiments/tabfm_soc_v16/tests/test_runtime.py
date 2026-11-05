import torch
import pytest
from experiments.tabfm_soc_v16.runtime import resolve_device,move_tree,StopController,run_lock,capture_rng,restore_rng


def test_nested_tensors_and_rng_round_trip():
    value={'args':(torch.tensor([2.]),),'kwargs':{'mask':torch.tensor([True])}}
    moved=move_tree(value,torch.device('cpu'))
    assert moved['kwargs']['mask'].dtype==torch.bool and moved['args'][0].device.type=='cpu'
    torch.manual_seed(34); state=capture_rng(torch.device('cpu')); a=torch.randn(4); restore_rng(state,torch.device('cpu'))
    assert torch.equal(a,torch.randn(4))


def test_deadline_and_exclusive_lock(tmp_path):
    assert StopController(stop_after_seconds=0).requested()
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path): pass


def test_cuda_fails_closed_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError): resolve_device('cuda')
    assert resolve_device('cpu').type=='cpu'


def test_model_parameter_placement_and_nested_cache_lists(monkeypatch):
    from experiments.tabfm_soc_v16 import pipeline
    module=torch.nn.Linear(2,1)
    placed=pipeline.place_model(module,torch.device('meta'))
    assert all(p.device.type=='meta' for p in placed.parameters())
    monkeypatch.setattr(pipeline,'DEVICE',torch.device('cpu'))
    result=pipeline.cast_cap({'items':[torch.ones(2,dtype=torch.bfloat16)]})
    assert result['items'][0].dtype==torch.float32

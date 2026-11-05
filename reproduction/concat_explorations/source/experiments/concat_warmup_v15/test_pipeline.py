import copy
import torch
from experiments.concat_warmup_v15.pipeline import train_update
from experiments.training_schedule_v15 import WarmupCosine


def test_interrupted_training_matches_uninterrupted_across_epoch_boundary():
    torch.manual_seed(42)
    initial = torch.nn.Linear(2, 1).state_dict()
    batches = [(torch.tensor([[float(i), 1.]]), torch.tensor([[float(i % 3)]])) for i in range(8)]
    def new():
        model = torch.nn.Linear(2, 1); model.load_state_dict(initial)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        schedule = WarmupCosine(opt, total_steps=8, warmup_steps=3, min_lr_ratio=.01, start_ratio=.01)
        return model, opt, schedule
    model, opt, schedule = new()
    rates = []
    for x, y in batches:
        rates.append(opt.param_groups[0]['lr'])
        train_update(model, opt, schedule, x, y)
    expected = copy.deepcopy(model.state_dict())
    a, b, c = new()
    for x, y in batches[:4]: train_update(a, b, c, x, y)
    checkpoint = copy.deepcopy((a.state_dict(), b.state_dict(), c.state_dict()))
    a, b, c = new(); a.load_state_dict(checkpoint[0]); b.load_state_dict(checkpoint[1]); c.load_state_dict(checkpoint[2])
    for x, y in batches[4:]: train_update(a, b, c, x, y)
    assert all(torch.equal(expected[k], a.state_dict()[k]) for k in expected)
    assert rates[0] < rates[1] < rates[2] and rates[-1] < rates[3]


def test_resume_rejects_mismatched_update_count():
    import pytest
    from experiments.concat_warmup_v15.pipeline import validate_resume
    validate_resume(2, 8, 10, 4, 5)
    with pytest.raises(RuntimeError, match='scheduler'):
        validate_resume(2, 8, 10, 4, 4)


def test_nonfinite_gradient_does_not_update_optimizer_or_schedule():
    import pytest
    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): return x.clone()
        @staticmethod
        def backward(ctx, grad): return grad * float('nan')
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Identity())
    class Wrapper(torch.nn.Module):
        def __init__(self): super().__init__(); self.inner = model
        def forward(self, x): return BadGradient.apply(self.inner(x))
    wrapper = Wrapper(); opt = torch.optim.AdamW(wrapper.parameters(), lr=.01)
    schedule = WarmupCosine(opt, total_steps=4, warmup_steps=2)
    before = copy.deepcopy(wrapper.state_dict())
    with pytest.raises(RuntimeError): train_update(wrapper, opt, schedule, torch.ones(1,1), torch.zeros(1,1))
    assert schedule.updates_completed == 0
    assert all(torch.equal(v, wrapper.state_dict()[k]) for k,v in before.items())

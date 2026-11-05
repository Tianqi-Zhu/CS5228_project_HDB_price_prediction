import copy
import pytest
import torch


def api():
    from experiments import training_schedule_v15
    assert hasattr(training_schedule_v15, 'WarmupCosine'), 'warmup scheduler is missing'
    return training_schedule_v15.WarmupCosine


def test_warmup_and_cosine_rates_used_by_actual_optimizer_updates():
    scheduler_type = api()
    weight = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([weight], lr=1.)
    scheduler = scheduler_type(optimizer, total_steps=9, warmup_steps=3)
    used = []
    for _ in range(9):
        used.append(optimizer.param_groups[0]['lr'])
        weight.grad = torch.ones_like(weight)
        optimizer.step()
        scheduler.step()
    assert used[:3] == pytest.approx([.01, .505, 1.])
    assert all(a > b for a, b in zip(used[2:], used[3:]))
    assert used[-1] == pytest.approx(.01)
    assert scheduler.state_dict()['updates_completed'] == 9
    with pytest.raises(ValueError):
        scheduler.step()


def test_optimizer_and_schedule_resume_exactly_without_restarting_warmup():
    scheduler_type = api()
    def setup():
        w = torch.nn.Parameter(torch.tensor([.3, -.7]))
        optimizer = torch.optim.AdamW([w], lr=.05, weight_decay=.01)
        return w, optimizer, scheduler_type(optimizer, total_steps=13, warmup_steps=4)
    def train(w, optimizer, scheduler, start, end):
        used = []
        for i in range(start, end):
            optimizer.zero_grad()
            (w * torch.tensor([float(i + 1), .25])).square().mean().backward()
            used.append(optimizer.param_groups[0]['lr'])
            optimizer.step(); scheduler.step()
        return used
    expected, opt, sch = setup()
    rates = train(expected, opt, sch, 0, 13)
    for cut in (1, 4, 8, 12):
        w, opt, sch = setup()
        first = train(w, opt, sch, 0, cut)
        saved = copy.deepcopy((w.detach(), opt.state_dict(), sch.state_dict()))
        resumed, opt2, sch2 = setup()
        with torch.no_grad(): resumed.copy_(saved[0])
        opt2.load_state_dict(saved[1]); sch2.load_state_dict(saved[2])
        remaining = train(resumed, opt2, sch2, cut, 13)
        assert first + remaining == rates
        assert torch.equal(expected, resumed)


def test_invalid_or_changed_schedule_fails_before_resuming():
    scheduler_type = api()
    def optimizer(): return torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.))], lr=.1)
    for total, warmup in ((0, 0), (3, 0), (3, 3), (3, 4)):
        with pytest.raises(ValueError): scheduler_type(optimizer(), total, warmup)
    with pytest.raises(ValueError): scheduler_type(optimizer(), 9, 2, min_lr_ratio=-.1)
    saved = scheduler_type(optimizer(), 9, 2).state_dict()
    with pytest.raises(ValueError): scheduler_type(optimizer(), 10, 2).load_state_dict(saved)
    with pytest.raises(ValueError): scheduler_type(optimizer(), 9, 3).load_state_dict(saved)

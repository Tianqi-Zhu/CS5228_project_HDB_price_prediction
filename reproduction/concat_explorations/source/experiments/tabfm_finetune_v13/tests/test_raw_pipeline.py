import numpy as np
from experiments.tabfm_finetune_v13.data import Population
from experiments.tabfm_finetune_v13.data_full import episode
from experiments.tabfm_finetune_v13.pipeline_raw import build_episodes, preprocess


def test_full_query_coverage_cold_context_and_masking():
    pop=Population(numeric=np.ones((4,2)),categorical=np.zeros((4,1)),target=np.array([1.,2.,3.,4.]),anchor_ksgd=np.zeros(4),actual_sgd=np.array([1000.,2000.,3000.,4000.]),row_id=np.arange(4),origin=np.array([-1,-1,0,1]),horizon=np.ones(4))
    eps=build_episodes(pop,pop,np.arange(4),43,episode)
    assert sum(len(e['query_row_id']) for e in eps)==4
    assert eps[0]['train_size']==0 and eps[0]['target_mean']==0 and eps[0]['target_std']==1
    assert np.isfinite(preprocess(eps[0])).all()
    for e in eps:
        assert np.all(e['y'][e['train_size']:]==0)
        assert not np.intersect1d(e['context_row_id'],e['query_row_id']).size


def test_budget_saves_next_step_only_when_exhausted():
    from experiments.tabfm_finetune_v13.pipeline_raw import pause_after_step
    checkpoints=[]
    save=lambda epoch,step:checkpoints.append((epoch,step))
    assert not pause_after_step(2,1,save,1,1)
    assert pause_after_step(2,2,save,1,2)
    assert checkpoints==[(1,2)]
    resume_epoch,resume_next_step=checkpoints[-1]
    assert list(range(resume_next_step,4))==[2,3] and resume_epoch==1

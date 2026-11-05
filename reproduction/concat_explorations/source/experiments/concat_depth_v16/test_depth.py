import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('experiments.concat_depth_v16.depth'), 'depth runner not implemented'
    from experiments.concat_depth_v16 import depth
    return depth


def test_predeclared_candidates_include_controls_and_distinct_outputs():
    d=api()
    assert d.CANDIDATES==[(3,.15),(4,.05),(4,.15),(6,.05),(6,.15)]
    assert len({d.candidate_name(*c) for c in d.CANDIDATES})==5


def test_only_verified_prior_selection_can_supply_learning_rate():
    d=api()
    assert d.validated_lr({'verified_reload':True,'selected_lr':.0003,'final_refit_performed':False})==.0003
    with pytest.raises(ValueError): d.validated_lr({'verified_reload':False,'selected_lr':.0003,'final_refit_performed':False})
    with pytest.raises(ValueError): d.validated_lr({'verified_reload':True,'selected_lr':.9,'final_refit_performed':False})
    with pytest.raises(ValueError): d.validated_lr({'verified_reload':True,'selected_lr':.0003,'final_refit_performed':True})

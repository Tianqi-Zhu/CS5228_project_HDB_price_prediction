import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('experiments.overnight_hdb_v15.queue') is not None, 'warmup queue is missing'
    from experiments.overnight_hdb_v15 import queue
    return queue


def test_only_new_selection_and_verifier_modules_allowed():
    queue = api()
    for module in ('experiments.concat_warmup_v15.pipeline', 'experiments.tabfm_warmup_v15.pipeline', 'experiments.tabfm_warmup_v15.verify'):
        cmd = ['python', '-u', '-m', module, '--selection-only']
        queue.validate_command(cmd)
        with pytest.raises(ValueError): queue.validate_command(cmd[:-1])
        with pytest.raises(ValueError): queue.validate_command(cmd + ['--final-refit'])
    with pytest.raises(ValueError):
        queue.validate_command(['python', '-m', 'experiments.tabfm_finetune_v13.pipeline_raw', '--selection-only'])


def test_source_hashes_must_still_match_after_readiness(tmp_path):
    queue = api()
    file = tmp_path/'source.py'; file.write_text('registered\n')
    marker = {'approved': True, 'sources': {str(file): queue.sha256(file)}}
    queue.verify_readiness(marker)
    file.write_text('changed\n')
    with pytest.raises(ValueError): queue.verify_readiness(marker)

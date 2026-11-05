"""Sequential local selection jobs, intended to run beneath caffeinate -i."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/overnight_hdb_v14'
MODULES = {'experiments.tabfm_finetune_v13.pipeline_raw', 'experiments.concat_raw_v14.pipeline'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_command(command):
    if '-m' not in command or command[command.index('-m') + 1] not in MODULES:
        raise ValueError('unapproved training module')
    if '--selection-only' not in command:
        raise ValueError('selection-only flag is required')
    if any(flag in command for flag in ('--final-refit', '--refit', '--all-labels', '--all-label-refit')):
        raise ValueError('all-label refit requires a later user decision')


def verify_readiness(marker):
    if marker.get('approved') is not True or not marker.get('sources'):
        raise ValueError('missing source review approval')
    for file, digest in marker['sources'].items():
        path = Path(file)
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != digest:
            raise ValueError('reviewed source changed: ' + file)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest.get('allow_final_refit') is not False:
        raise ValueError('final refit must be disabled')
    state_path = OUT / 'status.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    child = old.get('child_pid')
    if child:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('previous child may still be running; refusing duplicate training')
    state = {'status': 'starting', 'started_at': now(), 'queue_pid': os.getpid(),
             'power_guard_parent_pid': os.getppid(), 'allow_final_refit': False,
             'jobs': old.get('jobs', {})}
    environment = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONSTARTUP'):
        environment.pop(name, None)
    environment.update(DYLD_FALLBACK_LIBRARY_PATH='/opt/anaconda3/lib',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for job in manifest['jobs']:
        key = job['id']
        command = job['command']
        validate_command(command)
        if state['jobs'].get(key, {}).get('status') == 'completed':
            continue
        marker_path = ROOT / job['ready_marker']
        state.update(status='waiting_for_preparation', current_job=key, child_pid=None)
        atomic_json(state_path, state)
        while not marker_path.exists():
            time.sleep(5)
        marker = json.loads(marker_path.read_text())
        verify_readiness(marker)
        if marker.get('allow_final_refit') is not False:
            raise ValueError('readiness must explicitly prohibit final refit')
        if command not in marker.get('approved_commands', []):
            raise ValueError('job differs from reviewed command')
        log_path = ROOT / job['log']
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {'status': 'running', 'started_at': now(), 'log': job['log'], 'attempts': 0}
        state['jobs'][key] = record
        for attempt in range(2):
            verify_readiness(marker)
            record['attempts'] = attempt + 1
            with log_path.open('ab', buffering=0) as logfile:
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=logfile,
                                           stderr=subprocess.STDOUT, close_fds=True)
                state.update(status='running', child_pid=process.pid)
                atomic_json(state_path, state)
                print(now(), key, 'started', process.pid, flush=True)
                code = process.wait()
            state['child_pid'] = None
            record['return_code'] = code
            if code == 0:
                record.update(status='completed', finished_at=now())
                break
            record.update(status='retry_pending' if attempt == 0 else 'failed', finished_at=now())
            atomic_json(state_path, state)
            if attempt == 0:
                time.sleep(5)
        print(now(), key, record['status'], record['return_code'], flush=True)
        atomic_json(state_path, state)
    failed = [key for key, value in state['jobs'].items() if value['status'] != 'completed']
    state.update(status='finished_with_errors' if failed else 'selection_jobs_finished',
                 finished_at=now(), failed_jobs=failed, current_job=None,
                 next_action='Review validation performance with user; final 150000-label refit is not authorized yet.')
    atomic_json(state_path, state)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / 'queue_error.json', {'status': 'failed', 'at': now(), 'traceback': traceback.format_exc()})
        raise

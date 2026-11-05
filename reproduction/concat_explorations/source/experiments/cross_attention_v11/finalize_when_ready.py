"""Wait for the registered queue, then replay each selected checkpoint and report."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/cross_attention_v11'
STATUS = OUT / 'FINALIZATION_STATUS'

def main():
    STATUS.write_text('WAITING\n')
    print(f'watching {OUT / "STATUS"}', flush=True)
    while True:
        status = (OUT / 'STATUS').read_text().splitlines()[0].strip()
        if status == 'FAILED':
            STATUS.write_text('FAILED: training queue failed\n')
            print('Training queue failed; finalization stopped.', flush=True)
            return 1
        if status == 'COMPLETED': break
        if status != 'RUNNING': raise RuntimeError(f'unexpected training status {status!r}')
        time.sleep(30)
    STATUS.write_text('RUNNING\n')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None); env.pop('PYTHONSTARTUP', None)
    env['DYLD_FALLBACK_LIBRARY_PATH'] = '/opt/anaconda3/lib'
    for arm in ('concat', 'summary', 'sequence'):
        print(f'Replaying {arm}', flush=True)
        subprocess.run([sys.executable, '-m', 'experiments.cross_attention_v11.replay', arm], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'experiments.cross_attention_v11.verify_artifacts'], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'experiments.cross_attention_v11.report'], cwd=ROOT, env=env, check=True)
    STATUS.write_text('COMPLETED\n')
    print('Finalization completed.', flush=True)
    return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except Exception:
        message = traceback.format_exc()
        STATUS.write_text('FAILED\n' + message)
        print(message, flush=True)
        sys.exit(1)

"""Watch the sequential training queue and verify it once, with no busy polling."""
import json, time, traceback
from experiments.tabfm_finetune_v13.runner import OUT, atomic_json
from experiments.tabfm_finetune_v13.finalize import main as finalize

if __name__=='__main__':
    while True:
        path=OUT/'queue_status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('status')=='FAILED':
            atomic_json(OUT/'finalizer_status.json',{'status':'blocked_training_failed','error':state.get('error')}); break
        if state.get('status')=='training_complete':
            try:
                atomic_json(OUT/'finalizer_status.json',{'status':'running'})
                finalize()
                atomic_json(OUT/'finalizer_status.json',{'status':'passed'})
            except Exception as exc:
                atomic_json(OUT/'finalizer_status.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc()})
                raise
            break
        time.sleep(30)

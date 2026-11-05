"""Stop the old CPU batch once its in-flight trial has landed.

Two batches were running at once: the CPU one (`v4-search-s42`, trials 2-8) and the MPS one
(`v4-search-mps-s42`, all 8). They contend for the same cores and the CPU batch would
duplicate work the MPS run is superseding, so the CPU batch's *queue* must end — but not by
killing a trial that has not yet written its summary, which would leave a half-written
`trial2/` with no way to tell how far it got.

This waits for `trial2/summary.json` to appear, then terminates the batch. Completed trials
keep their checkpoints and summaries; nothing is deleted.

    .venv/bin/python experiments/feature_transformer_v4/stop_cpu_queue.py --pid 2415
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-s42"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--trial", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--reason", default=(
        "two batches ran concurrently; the CPU queue is superseded by the unified MPS run "
        "v4-search-mps-s42. The in-flight trial was allowed to finish so its artefacts are "
        "complete."))
    args = parser.parse_args(argv)

    marker = RUN / f"trial{args.trial}" / "summary.json"
    log_path = RUN / "queue_stop.json"
    started = time.time()

    def alive() -> bool:
        try:
            os.kill(args.pid, 0)
            return True
        except ProcessLookupError:
            return False

    if not alive():
        print(json.dumps({"already_stopped": True, "pid": args.pid}))
        return 0

    while not marker.exists():
        if not alive():
            print(json.dumps({"batch_exited_on_its_own": True,
                              "marker_present": marker.exists()}))
            return 0
        if time.time() - started > 3600:
            print(json.dumps({"timeout": True, "marker_present": marker.exists()}))
            return 1
        time.sleep(args.poll_seconds)

    # the trial is on disk; stop the queue before trial 3 begins
    summary = json.loads(marker.read_text())
    os.kill(args.pid, signal.SIGTERM)
    time.sleep(2.0)
    stopped = not alive()
    record = {
        "stopped_pid": args.pid,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "waited_seconds": round(time.time() - started, 1),
        "completed_trial": args.trial,
        "completed_trial_summary": {
            "selected_epoch": summary.get("selected_epoch"),
            "selected_rmse_sgd": summary.get("selected_rmse_sgd"),
            "epochs_run": summary.get("epochs_run"),
        },
        "terminated": stopped,
        "reason": args.reason,
        "preserved": "trial1 and trial2 keep their checkpoints, predictions and summaries",
        "deleted_anything": False,
    }
    log_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0 if stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())

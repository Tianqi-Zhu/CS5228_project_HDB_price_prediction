"""Wait for measured GPU capacity without starting a CUDA context."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import time


def gpu_snapshot(index=0):
    command = ["nvidia-smi", f"--id={index}",
               "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    row = next(csv.reader(io.StringIO(result.stdout)))
    if len(row) != 5:
        raise RuntimeError("Unexpected nvidia-smi response")
    try:
        return {"index": int(row[0]), "name": row[1].strip(),
                "total_gib": float(row[2]) / 1024, "free_gib": float(row[3]) / 1024,
                "utilization_pct": float(row[4])}
    except ValueError as error:
        raise OSError("GPU telemetry is not numeric; keep waiting") from error


def has_capacity(snapshot, min_free_gib, max_utilization):
    return (snapshot["free_gib"] >= min_free_gib
            and snapshot["utilization_pct"] <= max_utilization)


def wait_for_gpu(*, index=0, min_free_gib=8., max_utilization=35.,
                 stable_samples=2, poll_seconds=15., status_path=None,
                 timeout_seconds=None, snapshot_fn=None, sleep_fn=None):
    """Fail closed on telemetry errors; never kill another GPU process.

    No timeout is used by default. Optional timeouts fail rather than bypassing
    the gate. Injectable readers/clocks make the queue policy testable on CPU.
    """
    if min_free_gib <= 0 or stable_samples < 1 or poll_seconds <= 0:
        raise ValueError("Invalid GPU waiting policy")
    read = snapshot_fn or (lambda: gpu_snapshot(index))
    sleep = sleep_fn or time.sleep
    start, stable = time.monotonic(), 0
    while True:
        try:
            sample = read()
            if min_free_gib > sample["total_gib"]:
                raise ValueError("Requested free VRAM exceeds physical GPU memory")
            ready = has_capacity(sample, min_free_gib, max_utilization)
            stable = stable + 1 if ready else 0
            state = {"state": "ready" if stable >= stable_samples else "waiting",
                     "gpu": sample, "stable_samples": stable,
                     "required_free_gib": min_free_gib,
                     "max_utilization_pct": max_utilization}
        except (subprocess.SubprocessError, OSError, StopIteration, IndexError) as error:
            stable = 0
            state = {"state": "waiting_for_telemetry", "error": str(error)}
        state["wait_seconds"] = round(time.monotonic() - start, 2)
        if status_path is not None:
            path = Path(status_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(path)
        print("GPU_GATE " + json.dumps(state), flush=True)
        if state["state"] == "ready":
            return state
        if timeout_seconds is not None and time.monotonic() - start >= timeout_seconds:
            raise TimeoutError("GPU capacity wait timed out; training was not started")
        sleep(poll_seconds)

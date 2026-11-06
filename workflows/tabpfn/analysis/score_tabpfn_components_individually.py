"""Run the already selected outer ablations in sequential fresh GPU processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    import torch
    import numpy as np
    from analysis.compare_tabpfn_components_80_20 import (
        OUT, CACHE, CANDIDATES, DIAG_ADAPTER, audit, prediction_frame,
        predict_resumable, summarize,
    )
    from analysis.evaluation import metrics
    torch.set_num_threads(1)
    selected = json.loads((OUT / "component_selection.json").read_text())["selected"]
    if not (OUT / "view_selection.json").exists():
        raise ValueError("Inference views must be selected before outer scoring")
    jobs = {name: (CANDIDATES[name], CACHE / "candidates" / name / "adapter.pt", 1)
            for name in selected.values()}
    jobs.update({f"original_diag_{v}view": (None, DIAG_ADAPTER, v) for v in (2, 4)})
    if args.label:
        if args.label not in jobs:
            raise ValueError("Only previously selected/fixed ablations can be scored")
        base, _ = audit()
        config, adapter, views = jobs[args.label]
        x, q = base["X_train"], base["X_query"]
        y = np.asarray(base["y_train"], float)-base["train_baseline"]
        prediction = predict_resumable(args.label, config, adapter, x, y, q, base["query_baseline"],
                                       views=views, stage="outer")
        frame = prediction_frame(q, base["query_month"], base["y_query"], prediction)
        frame.to_csv(OUT / "outer" / args.label / "predictions.csv", index=False, float_format="%.10f")
        print(json.dumps({"model": args.label, **metrics(frame.actual,frame.predicted)}),flush=True)
        return
    for label in jobs:
        destination = OUT / "outer" / label / "predictions.csv"
        if destination.exists():
            print(f"REUSE completed {label}",flush=True)
            continue
        log = OUT / f"outer_{label}.log"
        print(f"START isolated inference {label}; log={log}",flush=True)
        with log.open("w",encoding="utf-8") as handle:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--label",label],
                           stdout=handle,stderr=subprocess.STDOUT,check=True)
        print(f"COMPLETE {label}",flush=True)
    summarize()


if __name__ == "__main__":
    main()

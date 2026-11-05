# Matched historical-467 comparison

Run the registered queue once with the repository `.venv/bin/python -m experiments.cross_attention_matched467_v12.run`. The current queue has already started; do not launch a second copy.

For recovery after a confirmed interruption, the supported entry point is:

```sh
env -u PYTHONPATH -u PYTHONSTARTUP DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib .venv/bin/python -m experiments.cross_attention_matched467_v12.resume
```

This verifies historical guards, registered and supplemental source/input hashes, Torch version and CPU/thread configuration before restoring the last completed epoch. Do not call the internal `run --resume` flag directly: it checks a narrower fingerprint and bypasses the supplemental guards. An unfinished epoch is rerun from its preceding saved boundary.

The runner trains all three arms before scoring validation. `finalize_report` waits for training completion, then invokes independent artifact verification and checkpoint replay before producing `REPORT.md`. Inspect both `STATUS` and `FINALIZATION_STATUS`, not the presence of a partial checkpoint, to determine completion. The local process can progress only while its host/runtime continues executing.

Fresh initialization is supported by the runner's common-state equality assertions and source review. `supplemental_provenance.json` independently reconstructs initialization/RNG digests; the live runner did not save a separate initial digest to compare directly. Do not describe the reconstructed digest as a comparison against a live-recorded digest.

The historical MPS 467.4027 score is a reference. Rank the fresh CPU arms against fresh CPU concat, all on the same 88,192-row fit and 35,158-row historical validation population, fixed 23 epochs and 3,979 updates. Validation was previously viewed and is not an independent test.

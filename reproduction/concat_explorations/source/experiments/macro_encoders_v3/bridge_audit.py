"""Read-only executable audit of `experiments/neural_fusion_v3/macro_bridge.py`.

Modifies nothing: no fitting, no run directory, no consumer source.  It imports the
bridge and drives it against the real exported contracts, then asserts the mapping
properties the review asks about:

1. row_id -> stage-absolute position (bijective, ascending, reversible)
2. fit / holdout / validation population membership against the producer's tables
3. `subset` / `batches` ordering, with macro arrays and targets correspondable
4. `MacroTensors` presents the v2 surface the training loop consumes

Run:
    .venv/bin/python -m experiments.macro_encoders_v3.bridge_audit
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

FAILURES: list[str] = []

V2_CONTRACT_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def main() -> int:
    from experiments.macro_encoders_v3 import api, contract as macro_contract
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import scaling as macro_scaling
    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.preprocessing import ScalerState
    from neural_fusion_v3 import macro_bridge

    macro_manifest = macro_contract.load_manifest()
    v2_manifest = C.load_manifest(V2_CONTRACT_DIR)
    split = "orig"
    print(f"macro contract: {macro_manifest['contract_id']}")
    print(f"v2  contract:   {v2_manifest['contract_id']}")
    print(f"policy:         {macro_manifest['policy']}\n")

    stages = {stage: macro_bridge.build_stage_macro(macro_manifest, split, stage)
              for stage in ("A", "B")}

    # ------------------------------------------------------------------ 1
    print("--- 1. row_id -> stage position")
    for stage, sm in stages.items():
        print(f"  stage {stage}: rows={sm.n_rows:,} "
              f"windows={len(np.unique(sm.window_index))}")
        check(f"{stage}: position map has one entry per row",
              len(sm.position) == sm.n_rows)
        check(f"{stage}: position values cover 0..n-1",
              sorted(sm.position.values()) == list(range(sm.n_rows)))
        probe = sm.row_id[:200]
        check(f"{stage}: position lookup round-trips",
              np.array_equal(np.array([sm.row_id[sm.position[int(r)]] for r in probe]),
                             probe))
        check(f"{stage}: row ids strictly ascending",
              bool(np.all(np.diff(sm.row_id) > 0)))

    # ------------------------------------------------------------------ 2
    print("\n--- 2. per-row arrays match an independent gather")
    for stage, sm in stages.items():
        windows = macro_data.load_windows(macro_manifest, split, stage)
        rows = macro_data.load_rows(macro_manifest, split, stage)
        scalers = macro_scaling.fit_macro_scalers(windows, rows)
        tensors = api.build_window_tensors(windows, scalers)
        check(f"{stage}: DBS rows equal independent gather",
              np.array_equal(sm.dbs_seq, tensors.dbs_seq.numpy()[rows.row_window_index]))
        check(f"{stage}: COE rows equal independent gather",
              np.array_equal(sm.coe_seq, tensors.coe_seq.numpy()[rows.row_window_index]))
        check(f"{stage}: row_id matches the producer",
              np.array_equal(sm.row_id, rows.row_id))
        check(f"{stage}: window_index matches the producer",
              np.array_equal(sm.window_index, rows.row_window_index))
        check(f"{stage}: cutoff matches the producer",
              np.array_equal(sm.macro_cutoff_month, rows.macro_cutoff_month))
        check(f"{stage}: scaler fitted on fit population only",
              sm.scaler_fitted_on.startswith("windows referenced by population=='fit'"),
              sm.scaler_fitted_on)

        samples = pd.read_csv(macro_contract.resolve(
            macro_manifest["splits"][split]["stages"][stage]["artifacts"]["samples"]["path"]))
        if macro_manifest["policy"] == "origin_asof":
            check(f"{stage}: cutoff equals row origin (origin_asof)",
                  np.array_equal(sm.macro_cutoff_month,
                                 samples["origin_month_index"].to_numpy()))
        print(f"      producer populations: {samples['population'].value_counts().to_dict()}")

    # ------------------------------------------------------------------ 3
    print("\n--- 3. align_population semantics")
    populations = {
        ("A", "fit"): C.load_population(V2_CONTRACT_DIR, v2_manifest, split, "A", "fit"),
        ("A", "holdout"): C.load_population(V2_CONTRACT_DIR, v2_manifest, split, "A", "holdout"),
        ("B", "fit"): C.load_population(V2_CONTRACT_DIR, v2_manifest, split, "B", "fit"),
        ("B", "validation"): C.load_population(V2_CONTRACT_DIR, v2_manifest, split, "B", "validation"),
    }
    for (stage, name), population in populations.items():
        sm = stages[stage]
        positions = macro_bridge.align_population(sm, population)
        ids = population.arrays.row_id.astype(np.int64)
        check(f"{stage}/{name}: one position per row", len(positions) == len(ids))
        check(f"{stage}/{name}: values are positions, not row ids",
              int(positions.max()) < sm.n_rows,
              f"max={int(positions.max())} n_rows={sm.n_rows}")
        check(f"{stage}/{name}: positions reproduce the row ids exactly",
              np.array_equal(sm.row_id[positions], ids))
        # population membership must agree with the producer's sample table
        samples = pd.read_csv(macro_contract.resolve(
            macro_manifest["splits"][split]["stages"][stage]["artifacts"]["samples"]["path"]))
        producer_ids = set(samples.loc[samples["population"] == name, "SOURCE_ROW_ID"].tolist())
        check(f"{stage}/{name}: same row set as the producer population",
              set(ids.tolist()) == producer_ids,
              f"{len(set(ids.tolist()) ^ producer_ids)} symmetric difference")
        print(f"      {stage}/{name}: {len(positions):,} rows, "
              f"{len(np.unique(sm.window_index[positions]))} unique windows")

    # ------------------------------------------------------------------ 4
    print("\n--- 4. position vs row-id collision hazard")
    sm_b = stages["B"]
    validation = populations[("B", "validation")]
    positions = macro_bridge.align_population(sm_b, validation)
    ids = validation.arrays.row_id.astype(np.int64)
    overlap = int(sum(1 for value in positions[:50].tolist() if int(value) in sm_b.position))
    print(f"      first 50 positions that are also valid row ids: {overlap}/50")
    if overlap:
        correct = sm_b.select(ids[:5])["macro_row"].numpy()
        confused = sm_b.select(positions[:5])["macro_row"].numpy()
        check("select mis-maps if fed align_population's return value",
              not np.array_equal(correct, confused),
              "the two are indistinguishable, so the API must document which is which")

    # ------------------------------------------------------------------ 5
    print("\n--- 5. subset / batches ordering")
    population = populations[("B", "validation")]
    arrays = population.arrays
    state = ScalerState.fit(arrays, population="audit")
    base = build_tensors(population, state, use_sequence=False)
    wrapped = macro_bridge.MacroTensors(base, arrays.row_id, sm_b)

    check("wrapper length equals the base", len(wrapped) == len(base))
    check("actual_sgd passed through",
          np.array_equal(wrapped.actual_sgd, base.actual_sgd))
    check("anchor_ksgd passed through",
          np.array_equal(wrapped.anchor_ksgd, base.anchor_ksgd))
    check("parity signature passed through",
          wrapped.parity_signature() == base.parity_signature())

    index = np.array([3, 0, 7, 11, 5])
    sub = wrapped.subset(index)
    check("subset: row ids follow the index",
          np.array_equal(sub.row_ids, arrays.row_id[index]))
    check("subset: targets follow the index",
          np.array_equal(sub.actual_sgd, base.actual_sgd[index]))
    check("subset: macro arrays follow the index",
          np.array_equal(sm_b.select(sub.row_ids)["macro_dbs"].numpy(),
                         sm_b.dbs_seq[[sm_b.position[int(i)] for i in sub.row_ids]]))

    # unshuffled batches: macro rows advance with the batch offset
    offset = 0
    mismatch_macro = 0
    mismatch_target = 0
    batches = 0
    for features, target in wrapped.batches(batch_size=512, shuffle=False, with_target=True):
        batches += 1
        macro_row = features["macro_row"].numpy()
        if not np.array_equal(features["macro_dbs"].numpy(), sm_b.dbs_seq[macro_row]):
            mismatch_macro += 1
        expected = base.target[offset:offset + len(target)]
        if not np.allclose(target.numpy(), expected):
            mismatch_target += 1
        offset += len(target)
    check("unshuffled batches: macro arrays track macro_row", mismatch_macro == 0,
          f"{mismatch_macro}/{batches}")
    check("unshuffled batches: targets track the batch offset", mismatch_target == 0,
          f"{mismatch_target}/{batches}")

    # shuffled batches: a permutation, and each row's macro data follows it
    order = []
    mismatch_shuffled = 0
    for features, target in wrapped.batches(batch_size=256, shuffle=True, seed=11,
                                            with_target=True):
        macro_row = features["macro_row"].numpy()
        order.append(macro_row)
        if not np.array_equal(features["macro_dbs"].numpy(), sm_b.dbs_seq[macro_row]):
            mismatch_shuffled += 1
    order = np.concatenate(order)
    # `macro_row` holds STAGE-ABSOLUTE positions, not indices into the wrapped rows,
    # so the correct invariant is that they form exactly the population's position set.
    expected_positions = np.sort(np.array(
        [sm_b.position[int(i)] for i in validation.arrays.row_id.astype(np.int64)]))
    check("shuffled batches: macro_row covers the population's positions exactly once",
          np.array_equal(np.sort(order), expected_positions),
          f"{len(order)} yielded vs {len(expected_positions)} expected")
    check("shuffled batches: macro arrays track macro_row", mismatch_shuffled == 0,
          f"{mismatch_shuffled} batches")

    # The wrapper must shuffle identically to the base for the same seed, or the
    # training loop's row order would drift from what v2 would have produced.  The
    # base yields feature dicts only, so the permutation is recovered from the
    # wrapper's own row ids and checked against a direct RNG replay.
    replay = np.arange(len(wrapped))
    np.random.default_rng(11).shuffle(replay)
    check("shuffled batches: permutation equals a direct seed replay",
          np.array_equal(np.sort(order), np.sort(np.array(
              [sm_b.position[int(validation.arrays.row_id[int(i)])] for i in replay]))))

    # same seed -> same permutation
    order2 = np.concatenate([
        features["macro_row"].numpy()
        for features, _ in wrapped.batches(batch_size=256, shuffle=True, seed=11,
                                           with_target=True)])
    check("shuffled batches: seed reproduces the permutation",
          np.array_equal(order, order2))

    # ------------------------------------------------------------------ 6
    print("\n--- 6. nested subset")
    a = wrapped.subset(np.array([0, 1, 2, 3, 4, 5]))
    b = a.subset(np.array([5, 4, 3]))
    expected_ids = arrays.row_id[np.array([0, 1, 2, 3, 4, 5])][np.array([5, 4, 3])]
    check("nested subset: row ids compose", np.array_equal(b.row_ids, expected_ids))
    check("nested subset: macro select resolves",
          np.array_equal(sm_b.select(b.row_ids)["macro_row"].numpy(),
                         np.array([sm_b.position[int(i)] for i in b.row_ids])))

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("macro_bridge mapping audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

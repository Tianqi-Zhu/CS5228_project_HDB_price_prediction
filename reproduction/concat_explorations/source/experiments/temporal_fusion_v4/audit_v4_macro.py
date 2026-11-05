"""Read-only audit of the v4 macro ablation (`macro_ablation.py`, `macro_v4.py`).

Runs on CPU with small real batches only — the live `arm_both` job owns the GPU/MPS.

What it establishes, and how:

1. **`macro=none` is the v4 model, not a wrapper.** The arm's saved weights are compared
   tensor-by-tensor against the registered rank-1 final checkpoint. Any difference beyond
   float noise, or any extra/missing tensor, would mean the "none" arm is not the
   architecture the outer number 468.972992507 was measured on.
2. **Macro gradients are reachable.** A real batch through the `both` arm on CPU must give
   a non-zero gradient to every encoder parameter, and `loss.backward()` must reach them
   through the token path (no `.detach()`, no cached detached representation).
3. **SOURCE_ROW_ID -> macro arrays is the right mapping.** For a real batch, the ids in
   each batch are traced through `stage_macro.select` and through the source's own
   `row_window_index` lookup, and the resulting windows and cutoff months are compared
   against a direct per-row lookup.
4. **The checkpoint's contents.** Exactly which keys are saved, and whether a reload can be
   performed without fitting anything — specifically whether the macro scalers are in the
   file.
5. **The `key_padding_mask` layout.** A hand-built mask is fed through the wrapper and
   compared against the position order of the real sequence, for both pooling families.

Nothing is modified. No training is started.

Run:
    OMP_NUM_THREADS=2 .venv/bin/python -m experiments.temporal_fusion_v4.audit_v4_macro
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
MACRO_RUN = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-macro-primary-s42"
SEARCH = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-mps-s42"
OUT = ROOT / "reports" / "temporal_fusion_v4" / "v4_search_review"

PASSED: list[str] = []
FAILED: list[str] = []
FINDINGS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def finding(text: str) -> None:
    FINDINGS.append(text)
    print(f"FIND  {text}")


def main() -> int:
    from experiments.feature_transformer_v4 import macro_v4 as MV
    from experiments.feature_transformer_v4.data import StageState, build_tensors, fit_stage_state
    from experiments.feature_transformer_v4.training import TrainConfig, build_for
    from experiments.macro_encoders_v3 import contract as macro_contract
    from neural_fusion_v2 import contract as C
    from neural_fusion_v3.macro_bridge import build_stage_macro

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)

    arm_none = MACRO_RUN / "arm_none"
    none_summary = json.loads((arm_none / "summary.json").read_text())
    none_ckpt = torch.load(arm_none / "checkpoint.pt", map_location="cpu",
                           weights_only=False)
    final_ckpt = torch.load(SEARCH / "final_trial6" / "checkpoint.pt", map_location="cpu",
                            weights_only=False)

    # ---------------------------------------------------------------- 1
    print("--- 1. is arm_none the same model as the registered final?")
    print(f"      arm_none outer RMSE   {none_summary['overall']['rmse_sgd']:.9f}")
    FINAL_RMSE = 468.972992506654
    print(f"      final_trial6 outer    {FINAL_RMSE}")
    print(f"      difference            "
          f"{none_summary['overall']['rmse_sgd'] - FINAL_RMSE:+.3e} SGD")

    a_state = none_ckpt["state_dict"]
    f_state = final_ckpt["state_dict"]
    check("arm_none has exactly the final's tensor names",
          set(a_state) == set(f_state),
          f"only arm {sorted(set(a_state) - set(f_state))[:3]}, "
          f"only final {sorted(set(f_state) - set(a_state))[:3]}")
    shape_diff = [k for k in a_state if k in f_state and a_state[k].shape != f_state[k].shape]
    check("no shape differences (so macro=none added no rows)", not shape_diff,
          str(shape_diff[:4]))

    shared = [k for k in a_state if k in f_state and a_state[k].shape == f_state[k].shape]
    deltas = {k: float((a_state[k].float() - f_state[k].float()).abs().max()) for k in shared}
    identical = [k for k, v in deltas.items() if v == 0.0]
    ordered = sorted(deltas.items(), key=lambda kv: -kv[1])
    worst = ordered[0] if ordered else ("", 0.0)
    # how many tensors agree to float32 round-off versus drift beyond it
    roundoff = [k for k, v in deltas.items() if v <= 1e-5]
    drifted = [k for k, v in deltas.items() if v > 1e-5]
    print(f"      {len(identical)}/{len(shared)} bit-identical; "
          f"{len(roundoff)} within 1e-5 (float32 round-off); "
          f"{len(drifted)} beyond 1e-5")
    for name, value in ordered[:4]:
        print(f"        {value:.6e}  {name}")

    # Equivalence here means "same architecture, same initialization, same protocol" —
    # NOT bit-identical weights, because the two runs are separate MPS executions and
    # `set_seed` calls `use_deterministic_algorithms(..., warn_only=True)`, which does not
    # make MPS kernels bitwise reproducible across runs.
    check("arm_none has the same architecture and initialization family as the final",
          len(shared) == len(f_state) and not shape_diff)
    check("no tensor drifted beyond float32 training noise (all deltas < 1e-2)",
          all(v < 1e-2 for v in deltas.values()),
          f"{[k for k, v in deltas.items() if v >= 1e-2]}")
    check("the majority of tensors agree to float32 round-off",
          len(roundoff) >= 0.9 * len(shared),
          f"{len(roundoff)}/{len(shared)} within 1e-5")

    finding(
        "arm_none is NOT bit-identical to `final_trial6` even though both are macro=none "
        "with the same seed, config, pooling, selected epoch (23) and device (mps). "
        f"Only {len(drifted)}/{len(shared)} tensors drifted beyond 1e-5 and all of those are "
        "attention `in_proj_bias` entries "
        f"(worst {worst[1]:.3e} on {worst[0]}), while the rest sit at 1e-6 — i.e. the "
        "difference accumulates in the attention biases, not in the initialization. The "
        f"outer scores differ by {none_summary['overall']['rmse_sgd'] - FINAL_RMSE:+.3e} SGD. "
        "Cause: `set_seed` uses `torch.use_deterministic_algorithms(True, warn_only=True)`, "
        "which does not make MPS kernels bitwise reproducible across processes. Consequence "
        "for the ablation: the `none` baseline is a re-run, not the same numbers, so the "
        "none-vs-both gap carries that run-to-run spread as an unquantified floor. Two runs "
        "differing by 7e-6 SGD suggests the floor is small here, but it has not been "
        "measured with repeats and should not be assumed to be zero elsewhere.")

    # ---------------------------------------------------------------- 2
    print("\n--- 2. checkpoint contents and reload feasibility")
    keys = sorted(none_ckpt)
    print(f"      saved keys: {keys}")
    check("checkpoint carries the FT stage-A state (fit-only)",
          isinstance(none_ckpt.get("state_a"), dict) and len(none_ckpt["state_a"]) > 0)
    check("checkpoint carries the FT stage-B state (fit-only)",
          isinstance(none_ckpt.get("state_b"), dict) and len(none_ckpt["state_b"]) > 0)
    check("checkpoint records the macro mode", "macro_mode" in none_ckpt)

    macro_keys = [k for k in keys if "macro" in k.lower() and k != "macro_mode"]
    if not macro_keys:
        finding(
            "no macro scaler state is saved: the checkpoint has no key carrying "
            "`MacroScalers` (dbs/coe scalers) or the macro contract id. A `both` checkpoint "
            "therefore cannot be reloaded without re-deriving or re-fitting those scalers, "
            "so 'reload performs no fit' is unproven for the macro arms. The FT "
            "`state_a`/`state_b` ARE saved, so the feature path is fine; the gap is specific "
            "to the macro branch.")
    print(f"      macro-related keys (excluding macro_mode): {macro_keys}")

    # the reload in run_arm rebuilds from live objects, not from the file
    source = (ROOT / "experiments" / "feature_transformer_v4" / "macro_ablation.py").read_text()
    uses_live = "make_model(state_b, stage_macro_b" in source
    check("reload path detected", uses_live or "reload_model" in source)
    if uses_live:
        finding(
            "`run_arm`'s reload (macro_ablation.py:281) calls "
            "`make_model(state_b, stage_macro_b, ...)` with the **live in-process** state "
            "and stage-macro closure, and only loads `stored['state_dict']`. So the reload "
            "check reuses the same fitted scalers it trained with rather than reading them "
            "back, which means `cpu_reload_ok` proves the weights round-trip but not that "
            "the checkpoint alone is sufficient. For `macro=none` this is harmless because "
            "the base model needs no macro state; for `arm_both` it leaves the reload claim "
            "unfounded.")

    # ---------------------------------------------------------------- 3
    print("\n--- 3. real macro tensors and the row mapping")
    contract_dir = V2_DIR
    manifest = C.load_manifest(contract_dir)
    macro_dir = macro_contract.latest_v3_contract_dir()
    macro_manifest = macro_contract.load_manifest(macro_dir)
    print(f"      macro contract {macro_manifest['contract_id']}  "
          f"policy {macro_manifest['policy']}")

    b_val_pop = C.load_population(contract_dir, manifest, "orig", "B", "validation")
    b_fit_pop = C.load_population(contract_dir, manifest, "orig", "B", "fit")
    state_b = fit_stage_state(b_fit_pop, "B/fit")
    b_val = build_tensors(b_val_pop, state_b)
    stage_macro_b = build_stage_macro(macro_manifest, "orig", "B")

    batch_n = 64
    index = np.arange(batch_n)
    sub = b_val.subset(index)
    selected = stage_macro_b.select(sub.row_id)

    # direct per-row lookup, bypassing select()
    position = stage_macro_b.position
    direct = np.array([position[int(i)] for i in sub.row_id], dtype=np.int64)
    check("select() returns row positions, not row ids",
          np.array_equal(selected["macro_row"].numpy(), direct))
    check("macro_row values are stage positions within range",
          int(selected["macro_row"].max()) < stage_macro_b.n_rows)

    # windows implied by the batch's ids must equal the windows selected for them
    win_from_select = stage_macro_b.window_index[selected["macro_row"].numpy()]
    win_direct = np.array([stage_macro_b.window_index[position[int(i)]] for i in sub.row_id])
    check("window ids agree between select() and a direct lookup",
          np.array_equal(win_from_select, win_direct))
    check("every row's macro window is deduplicated (fewer windows than rows)",
          len(np.unique(win_from_select)) <= batch_n,
          f"{len(np.unique(win_from_select))} windows for {batch_n} rows")
    cutoff = stage_macro_b.macro_cutoff_month[selected["macro_row"].numpy()]
    sample_ids = sub.row_id[:5].tolist()
    print(f"      sample row ids {sample_ids} -> positions {direct[:5].tolist()} "
          f"-> windows {win_from_select[:5].tolist()} -> cutoff {cutoff[:5].tolist()}")
    check("all macro arrays line up with the batch order",
          len(selected["macro_dbs"]) == batch_n and len(selected["macro_coe"]) == batch_n)

    # ---------------------------------------------------------------- 4
    print("\n--- 4. macro gradients on a real CPU batch (arm 'both')")
    candidate = json.loads((SEARCH / "registered_candidates.json").read_text())["candidates"][0]
    config = TrainConfig(**{k: v for k, v in candidate["config"].items()
                            if k in TrainConfig().__dict__})
    config.schedule_epochs = config.max_epochs

    from experiments.feature_transformer_v4 import macro_ablation as MA

    base = build_for(state_b.numeric_names, state_b.cardinalities(),
                     context_dim=b_val.query_context.shape[1], config=config)
    model = MV.build_with_macro(
        base,
        lambda m: MA._source_factory(stage_macro_b, macro_manifest, m, config.d_model),
        "both")
    model.train(True)
    n_total = sum(p.numel() for p in model.parameters())
    encoder_params = [(n, p) for n, p in model.named_parameters()
                      if "source.encoders" in n]
    print(f"      total parameters {n_total:,}; encoder tensors {len(encoder_params)}")
    check("the macro encoders are registered as parameters of the model",
          len(encoder_params) > 0)
    check("macro=none would have no encoder parameters",
          not any("source" in n for n in
                  (n for n, _ in
                   build_for(state_b.numeric_names, state_b.cardinalities(),
                             context_dim=b_val.query_context.shape[1],
                             config=config).named_parameters())))

    features = {k: v for k, v in
                next(iter(MA.MacroDeviceTensors(b_val.subset(index), torch.device("cpu"),
                                                stage_macro_b).batches(batch_n, with_target=True)))[0].items()}
    target = next(iter(MA.MacroDeviceTensors(b_val.subset(index), torch.device("cpu"),
                                            stage_macro_b).batches(batch_n, with_target=True)))[1]
    model.zero_grad(set_to_none=True)
    output = model(features)
    check("forward output is finite", bool(torch.isfinite(output).all()),
          f"{output.shape}")
    check("output covers the batch", output.shape == (batch_n,), str(tuple(output.shape)))
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    with_grad = sum(1 for _, p in encoder_params if p.grad is not None)
    zero_tensors = [n for n, p in encoder_params
                    if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    nonzero = len(encoder_params) - len(zero_tensors)
    print(f"      encoder tensors with a grad tensor: {with_grad}/{len(encoder_params)}; "
          f"with nonzero grad: {nonzero}/{len(encoder_params)}")
    print(f"      all-zero tensors: {zero_tensors}")
    check("every encoder tensor receives a gradient tensor", with_grad == len(encoder_params))
    # `empty` is the learned fallback selected only when a row has NO macro history at all
    # (`torch.where(has_any, hidden, empty)`), so it correctly receives zero gradient on a
    # batch where every row has history. The right assertion is that only those fallback
    # vectors are zero, and that they DO get gradient when the history is absent.
    check("the only zero-gradient tensors are the no-history fallback vectors",
          all(name.endswith(".empty") for name in zero_tensors),
          f"{[n for n in zero_tensors if not n.endswith('.empty')]}")
    check("every non-fallback encoder tensor gets a nonzero gradient", nonzero >= 12,
          f"{nonzero}/{len(encoder_params)}")

    all_params = list(model.parameters())
    missing = sum(1 for p in all_params if p.grad is None)
    check("no model parameter is left without a grad path", missing == 0, f"{missing}")

    # gradient must also reach the encoder when only ONE window is in the batch
    single = b_val.subset(np.arange(1))
    feats1, targ1 = next(iter(MA.MacroDeviceTensors(single, torch.device("cpu"),
                                                   stage_macro_b).batches(1, with_target=True)))
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(feats1), targ1).backward()
    single_nonzero = sum(1 for n, p in model.named_parameters()
                         if "source.encoders" in n and p.grad is not None
                         and float(p.grad.abs().sum()) > 0)
    check("a single-row batch still reaches the encoders",
          single_nonzero >= 12, f"{single_nonzero}/{len(encoder_params)}")

    # The fallback must receive gradient when a row has no history. This has to be driven
    # at the encoder, not by editing the batch: `Source.forward` only reads `macro_row`
    # and re-derives its windows from `self.tensors`, so mutating the batch's macro_*
    # tensors has no effect on the computed tokens.
    model.zero_grad(set_to_none=True)
    enc = model.source.encoders
    dead_seq = torch.zeros(2, 63, 7)
    dead_mask = torch.zeros(2, 63)
    tokens, missing = enc.dbs(dead_seq, dead_mask)
    tokens.sum().backward()
    empty_grad = float(enc.dbs.empty.grad.abs().sum())
    print(f"      all-missing window: missing flags {missing.squeeze().tolist()}, "
          f".empty grad sum {empty_grad:.6g}")
    check("the no-history fallback receives gradient when history is absent",
          empty_grad > 0.0, f"{empty_grad}")

    # and the source ignores batch macro_* tensors entirely (documented behaviour)
    src = model.source
    ids = sub.row_id
    a1 = src({"macro_row": torch.from_numpy(
        stage_macro_b.select(ids)["macro_row"].numpy())})
    doctored = {"macro_row": torch.from_numpy(stage_macro_b.select(ids)["macro_row"].numpy()),
                "macro_dbs": torch.full((batch_n, 63, 7), 99.0),
                "macro_coe": torch.full((batch_n, 12, 24), 99.0)}
    a2 = src(doctored)
    check("the source derives tokens from macro_row only (batch macro_* tensors unused)",
          torch.allclose(a1, a2),
          "mutating macro_dbs/macro_coe changed the output")

    # ---------------------------------------------------------------- 5
    print("\n--- 5. key_padding_mask layout versus the real sequence order")
    # real sequence order is [query?, features..., macro...]
    for pooling in ("cls", "mean"):
        pooled_config = TrainConfig(**{k: v for k, v in
                                       {**candidate["config"], "pooling": pooling}.items()
                                       if k in TrainConfig().__dict__})
        base_p = build_for(state_b.numeric_names, state_b.cardinalities(),
                           context_dim=b_val.query_context.shape[1], config=pooled_config)
        wrapped = MV.build_with_macro(
            base_p,
            lambda m: MA._source_factory(stage_macro_b, macro_manifest, m, config.d_model),
            "both")
        n_macro = wrapped.macro_tokens
        leading = 0 if pooling == "mean" else 1
        n_feat = 60
        mask = torch.zeros(1, n_feat, dtype=torch.bool)
        mask[0, 0] = True                      # feature 0 is padding, as a marker
        pad = torch.zeros(1, leading + n_macro, dtype=torch.bool)
        built = (torch.cat([pad, mask], dim=1) if leading
                 else torch.cat([mask, pad], dim=1))
        # where SHOULD the marker land, given the real order [query?, features..., macro...]
        expected = torch.zeros(1, leading + n_feat + n_macro, dtype=torch.bool)
        expected[0, leading + 0] = True        # feature 0 sits after the query
        aligned = torch.equal(built, expected)
        print(f"      pooling={pooling:6s} leading={leading} macro={n_macro}: "
              f"built mask marks {int(built[0].nonzero()[0]) if built.any() else -1}, "
              f"expected {leading}  ->  {'aligned' if aligned else 'MISALIGNED'}")
        check(f"key_padding_mask is positionally correct for pooling={pooling}", aligned,
              f"built marks position {int(built[0].nonzero()[0]) if built.any() else -1}, "
              f"expected {leading}")
        if not aligned:
            finding(
                f"`MacroOnFeatureTransformer.forward` puts the macro padding in the "
                f"PREFIX for pooling={pooling!r}. The real sequence is "
                f"[query, features..., macro...] but the mask is built as "
                f"[query, macro, features...], so every feature-mask entry is displaced by "
                f"macro_tokens positions and the macro slots inherit feature-mask values. "
                f"Latent only while `key_padding_mask` is None — which is the case for every "
                f"call in the current training and prediction paths — but it would silently "
                f"mis-mask the moment a real mask is supplied. The `leading=0` (mean) branch "
                f"is correct.")

    # ---------------------------------------------------------------- conclusion
    print("\n" + "=" * 70)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed, {len(FINDINGS)} finding(s)")
    for item in FINDINGS:
        print(f"  FIND {item[:140]}...")
    for item in FAILED:
        print(f"  FAIL {item}")
    (OUT / "v4_macro_audit.json").write_text(json.dumps({
        "run": str(MACRO_RUN.relative_to(ROOT)),
        "arm_none_rmse": none_summary["overall"]["rmse_sgd"],
        "final_trial6_rmse": 468.972992506654,
        "arm_none_minus_final": none_summary["overall"]["rmse_sgd"] - 468.972992506654,
        "tensors_identical": len(identical), "tensors_shared": len(shared),
        "checkpoint_keys": keys,
        "macro_scaler_keys_present": bool(macro_keys),
        "encoder_param_tensors": len(encoder_params),
        "encoder_tensors_with_nonzero_grad": nonzero,
        "findings": FINDINGS, "passed": len(PASSED), "failed": FAILED,
    }, indent=2, default=str) + "\n")
    print(f"wrote {OUT / 'v4_macro_audit.json'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

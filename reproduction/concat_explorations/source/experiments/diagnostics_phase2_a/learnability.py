"""Phase-2 diagnostic A: is the v4 architecture able to fit a small real subset at all?

The question is *learnability*, not accuracy. If the model cannot drive RMSE near zero on
64 rows it has memorised, then no amount of feature or architecture work on the full
53k rows is the next thing to do. If it can, the architecture and the optimiser are fine
and the remaining error is information, not a defect.

Everything is inner-only: A/fit and A/holdout, frozen contract v2.1, no B ever loaded.

Three pieces:

1. **Collision census.** Count how many rows share an identical model input (56 numeric +
   4 categorical + anchor). Where inputs repeat but labels differ, no function of those
   inputs can predict better than the group mean, so the empirical conditional-mean SSE
   gives a hard floor for that subset. Without this, "RMSE did not reach zero" is
   uninterpretable — the subset may be unable to reach zero.
2. **Unique-input subset.** A deterministic 2,000-row subset with no repeated input, saved
   by SOURCE_ROW_ID so a reviewer can reconstruct exactly which rows were used.
3. **Positive controls at 64 / 256 / 2000.** The unchanged v4 CLS / 3-layer / 48-width
   forward, dropout 0, weight decay 0, full-population evaluation at a fixed interval.

Interpretation is declared before the numbers are seen:
   pass    = RMSE <= 5 SGD **on the same subset the floor was computed on**
   plateau = evaluation RMSE still falling slowly at the end of the declared budget

**What a pass does and does not mean.** A pass proves only that *this subset* is
memorisable by this architecture and optimiser. It does NOT establish that the optimiser is
fine for the full 52,310-row problem, and it does NOT establish that the remaining error on
the full population is information rather than a defect. Those are different claims and
this diagnostic does not test them. Likewise a plateau is NOT by itself a bug: it reports
bounded capacity or optimiser reach, and the next step is gradients/capacity/optimiser in
isolation, not a bigger model.

**The floor must be the subset's own floor.** The uniqueness filter removes every repeated
input, so a unique subset has a conditional-mean floor of exactly 0.0 by construction.
Quoting the full population's floor (49.17 SGD) against a unique subset would silently
widen the 5 SGD target to 54.17 and turn a failure into a pass.

Run:
    OMP_NUM_THREADS=2 .venv/bin/python -m experiments.diagnostics_phase2_a.learnability \
        --budgets 64,256,2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
OUT = ROOT / "reports" / "diagnostics_phase2_a"
ARCHIVE = OUT / "archive"
SEED = 42


def run_dir_name(budget: int, epochs: int, lr: float, seed: int) -> str:
    """One directory per (budget, epochs, lr, seed) so no run can overwrite another."""
    return f"budget{budget}_epochs{epochs}_lr{lr:g}_seed{seed}"


# ---------------------------------------------------------------- collisions
def input_key(numeric: np.ndarray, categorical: np.ndarray, anchor_ksgd: np.ndarray) -> np.ndarray:
    """A stable byte key per row over the *complete* model input.

    The anchor is included because the model predicts a residual on top of it, so two rows
    with identical features but different anchors are not the same prediction problem.
    """
    payload = np.concatenate([numeric.astype(np.float32),
                              categorical.astype(np.float32),
                              anchor_ksgd.astype(np.float64)[:, None]], axis=1)
    contiguous = np.ascontiguousarray(payload)
    return np.array([hashlib.blake2b(row.tobytes(), digest_size=16).digest()
                     for row in contiguous], dtype="S16")


def census(numeric, categorical, anchor, actual) -> dict:
    """Group identical inputs; report inconsistent-label groups and their SSE floor."""
    keys = input_key(numeric, categorical, anchor)
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    # per-group label spread
    order = np.argsort(inverse, kind="stable")
    sorted_group = inverse[order]
    sorted_actual = actual[order]
    bounds = np.flatnonzero(np.diff(sorted_group)) + 1
    groups = np.split(sorted_actual, bounds)

    inconsistent = [g for g in groups if len(g) > 1 and float(g.max() - g.min()) > 0]
    # SSE floor of predicting each group's own mean
    floor_sse = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    n = len(actual)
    floor_rmse_sgd = float(np.sqrt(floor_sse / n)) if n else 0.0

    sizes = counts
    return {
        "n_rows": int(n),
        "n_unique_inputs": int(len(unique)),
        "duplicate_rows": int(n - len(unique)),
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "groups_of_2_plus": int((sizes >= 2).sum()),
        "rows_in_groups_2_plus": int(sizes[sizes >= 2].sum()),
        "inconsistent_label_groups": len(inconsistent),
        "rows_in_inconsistent_groups": int(sum(len(g) for g in inconsistent)),
        "max_label_spread_in_group_sgd": float(max((g.max() - g.min() for g in groups
                                                    if len(g) > 1), default=0.0)),
        "conditional_mean_floor_rmse_sgd": floor_rmse_sgd,
        "note": ("the floor is the RMSE of predicting each identical-input group's own "
                 "mean, i.e. the best any function of these inputs could do on this "
                 "population; 0.0 means inputs are unique and the floor is uninformative"),
    }


def choose_unique_subset(numeric, categorical, anchor, row_id, target: int) -> np.ndarray:
    """First `target` rows (by ascending SOURCE_ROW_ID) with no repeated input.

    Deterministic and reconstructible: a reviewer can re-derive the same list from the
    contract without trusting a saved file.
    """
    keys = input_key(numeric, categorical, anchor)
    order = np.argsort(row_id, kind="stable")
    seen: set[bytes] = set()
    chosen: list[int] = []
    for index in order:
        key = keys[index].tobytes()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(int(index))
        if len(chosen) >= target:
            break
    return np.asarray(chosen, dtype=np.int64)


# ---------------------------------------------------------------- training
def masked_forward(model, batch):
    return model(batch)


def run_budget(model_factory, tensors, rows: np.ndarray, budget: dict,
               curve_path: Path | None = None) -> dict:
    """Train on `rows` with full-subset evaluation every `eval_every` epochs.

    ``curve_path`` is rewritten after **every** epoch so an interrupted run still leaves
    its real curve on disk instead of dying with the process.
    """
    from experiments.feature_transformer_v4.training import metrics

    subset = tensors.subset(rows)
    n = len(subset)
    set_seed = budget["set_seed"]

    set_seed(budget["seed"])
    model = model_factory()
    optimiser = torch.optim.AdamW(model.parameters(), lr=budget["lr"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                          T_max=budget["epochs"])

    curve = []
    started = time.perf_counter()
    batch_size = min(budget["batch_size"], n)
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = 0
    hit_cap = False
    for epoch in range(1, budget["epochs"] + 1):
        model.train(True)
        total, seen = 0.0, 0
        for features, target in subset.batches(batch_size, shuffle=True,
                                               seed=budget["seed"] * 1000 + epoch,
                                               with_target=True):
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(masked_forward(model, features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
            total_steps += 1
            # the cap is enforced per optimiser step, so a partial epoch cannot overshoot
            # it and leave the recorded metrics describing older weights
            if budget.get("max_steps") and total_steps >= budget["max_steps"]:
                hit_cap = True
                break
        scheduler.step()

        record = {"epoch": epoch, "steps": total_steps,
                  "train_mse_ksgd2": total / max(seen, 1)}
        stop_now = False
        if epoch % budget["eval_every"] == 0 or epoch == budget["epochs"]:
            model.eval()
            with torch.no_grad():
                raw = []
                for features in subset.batches(4096, shuffle=False):
                    raw.append(model(features).cpu().numpy())
                predicted = subset.to_sgd(np.concatenate(raw))
            stats = metrics(subset.actual_sgd, predicted)
            record.update({"train_rmse_sgd": float(np.sqrt(record["train_mse_ksgd2"])
                                                   * 1000.0),
                           "eval_rmse_sgd": stats["rmse_sgd"],
                           "eval_mae_sgd": stats["mae_sgd"],
                           "eval_bias_sgd": stats["bias_sgd"],
                           "seconds": round(time.perf_counter() - started, 2)})
            print(f"      epoch {epoch:4d} steps {total_steps:5d}  "
                  f"train_rmse {record['train_rmse_sgd']:9.3f}  "
                  f"eval_rmse {stats['rmse_sgd']:9.3f}  "
                  f"bias {stats['bias_sgd']:+8.3f}  {record['seconds']:.1f}s", flush=True)
            # stop as soon as the subset is memorised: no point spending the full budget
            if budget.get("stop_at_rmse") is not None \
                    and stats["rmse_sgd"] <= budget["stop_at_rmse"]:
                stop_now = True
        curve.append(record)
        if curve_path is not None:
            curve_path.write_text(json.dumps(curve, indent=2) + "\n")
        if stop_now:
            print(f"      STOPPED early: eval RMSE {stats['rmse_sgd']:.6f} <= "
                  f"{budget['stop_at_rmse']} at {total_steps} steps", flush=True)
            break
        if hit_cap:
            print(f"      STOPPED at the declared step cap {budget['max_steps']} "
                  f"(mid-epoch, {total_steps} steps)", flush=True)
            break

    # Whatever ended the loop, the saved model must be the one the metrics describe.
    # A mid-epoch cap break leaves the last recorded eval stale, so re-evaluate once on
    # the final weights and overwrite the record.
    model.eval()
    with torch.no_grad():
        raw = [model(features).cpu().numpy()
               for features in subset.batches(4096, shuffle=False)]
    final_predicted = subset.to_sgd(np.concatenate(raw))
    final_stats = metrics(subset.actual_sgd, final_predicted)
    if curve:
        curve[-1].update({"final_eval_rmse_sgd": final_stats["rmse_sgd"],
                          "final_eval_mae_sgd": final_stats["mae_sgd"],
                          "final_eval_bias_sgd": final_stats["bias_sgd"],
                          "final_steps": total_steps,
                          "final_epoch_exact": round(total_steps / steps_per_epoch, 4),
                          "final_eval_is_posthoc_final": True})
        if curve_path is not None:
            curve_path.write_text(json.dumps(curve, indent=2) + "\n")
    print(f"      final full-subset eval on saved weights: RMSE "
          f"{final_stats['rmse_sgd']:.6f} at {total_steps} steps", flush=True)

    elapsed = time.perf_counter() - started
    evaluations = [c for c in curve if "eval_rmse_sgd" in c]
    first_eval = evaluations[0]["eval_rmse_sgd"] if evaluations else float("nan")
    # prefer the forced post-hoc final evaluation: it always describes the saved weights
    last_eval = (curve[-1].get("final_eval_rmse_sgd", float("nan")) if curve
                 else float("nan"))
    # slope over the second half of the evaluated epochs: a plateau shows here
    half = evaluations[len(evaluations) // 2:]
    slope = float("nan")
    if len(half) >= 2:
        xs = np.array([c["epoch"] for c in half], dtype=float)
        ys = np.array([c["eval_rmse_sgd"] for c in half], dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])

    return {"model": model, "curve": curve, "seconds": elapsed,
            "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
            "first_eval_rmse_sgd": float(first_eval),
            "final_eval_rmse_sgd": float(last_eval),
            "final_eval_mae_sgd": (float(curve[-1]["final_eval_mae_sgd"])
                                   if curve and "final_eval_mae_sgd" in curve[-1] else None),
            "final_eval_bias_sgd": (float(curve[-1]["final_eval_bias_sgd"])
                                    if curve and "final_eval_bias_sgd" in curve[-1]
                                    else None),
            "second_half_slope_sgd_per_epoch": slope,
            "n_rows": int(n)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="64,256,2000")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--target-rmse", type=float, default=5.0)
    parser.add_argument("--stop-at-rmse", type=float, default=5.0,
                        help="stop a budget as soon as eval RMSE reaches this")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="hard optimisation-step cap per budget")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    from experiments.feature_transformer_v4.data import build_tensors, fit_stage_state
    from experiments.feature_transformer_v4.models import build_model
    from experiments.feature_transformer_v4.training import (TrainConfig, build_for,
                                                             metrics, set_seed)
    from neural_fusion_v2 import contract as C

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    manifest = C.load_manifest(V2_DIR)
    a_fit_pop = C.load_population(V2_DIR, manifest, "orig", "A", "fit")
    state = fit_stage_state(a_fit_pop, "A/fit")
    tensors = build_tensors(a_fit_pop, state)
    print(f"A/fit {len(tensors):,} rows  numeric {tensors.numeric.shape[1]}  "
          f"categorical {tensors.categorical.shape[1]}")

    # ------------------------------------------------------------ 1. census
    print("\n--- 1. input-collision census (complete model input, anchor included)")
    summary = census(tensors.numeric, tensors.categorical, tensors.anchor_ksgd,
                     tensors.actual_sgd)
    for key, value in summary.items():
        if key != "note":
            print(f"      {key}: {value}")
    floor = summary["conditional_mean_floor_rmse_sgd"]
    if summary["n_unique_inputs"] == summary["n_rows"]:
        print("      ALL inputs are unique -> the conditional-mean floor is uninformative; "
              "positive control must reach the absolute target unaided")
    (OUT / "census.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ------------------------------------------------------------ 2. subset
    print("\n--- 2. deterministic unique-input subset")
    requested = max(int(b) for b in args.budgets.split(","))
    rows = choose_unique_subset(tensors.numeric, tensors.categorical, tensors.anchor_ksgd,
                                tensors.row_id, requested)
    print(f"      selected {len(rows)} unique-input rows "
          f"(requested {requested}); first ids {tensors.row_id[rows[:5]].tolist()}")
    keys = input_key(tensors.numeric[rows], tensors.categorical[rows],
                     tensors.anchor_ksgd[rows])
    check_unique = len(np.unique(keys)) == len(rows)
    print(f"      uniqueness re-checked: {check_unique}")
    np.save(OUT / "selected_row_ids.npy", tensors.row_id[rows])
    np.save(OUT / "selected_positions.npy", rows)
    (OUT / "subset.json").write_text(json.dumps({
        "requested": requested, "selected": int(len(rows)),
        "unique_inputs": bool(check_unique),
        "selection_rule": ("ascending SOURCE_ROW_ID over A/fit, keep the first row of each "
                           "unique complete input (56 numeric + 4 categorical + anchor)"),
        "row_ids": tensors.row_id[rows].tolist(),
    }, indent=2) + "\n")

    # ------------------------------------------------------------ 3. controls
    print("\n--- 3. positive controls (unchanged v4 CLS/3-layer/48, dropout 0, wd 0)")
    config = TrainConfig(pooling="cls", d_model=48, layers=3, heads=4, ff_multiplier=2,
                         dropout=0.0, learning_rate=args.lr, weight_decay=0.0,
                         batch_size=args.batch_size, max_epochs=args.epochs,
                         eligible_after=1, seed=args.seed, schedule_epochs=args.epochs)

    def factory():
        model = build_for(state.numeric_names, state.cardinalities(),
                          context_dim=tensors.query_context.shape[1], config=config)
        model.train(True)
        return model

    results = []
    for budget_size in [int(b) for b in args.budgets.split(",")]:
        if budget_size > len(rows):
            print(f"      skipping {budget_size}: only {len(rows)} unique rows available")
            continue
        print(f"\n    budget {budget_size} rows, {args.epochs} epochs, lr {args.lr}, "
              f"batch {min(args.batch_size, budget_size)}", flush=True)

        directory = OUT / run_dir_name(budget_size, args.epochs, args.lr, args.seed)
        if directory.exists():
            raise SystemExit(f"refusing to overwrite {directory}; use a different budget/"
                             f"epochs/lr or move it to {ARCHIVE}")
        directory.mkdir(parents=True, exist_ok=True)

        # The floor must be computed on THIS subset. After the uniqueness filter the subset
        # has no repeated inputs, so its floor is 0.0 by construction; the full-population
        # 49.17 would wrongly widen the target to 54.17.
        subset_rows = rows[:budget_size]
        local = census(tensors.numeric[subset_rows], tensors.categorical[subset_rows],
                       tensors.anchor_ksgd[subset_rows], tensors.actual_sgd[subset_rows])
        local_floor = local["conditional_mean_floor_rmse_sgd"]
        print(f"      subset floor RMSE {local_floor:.6f} SGD "
              f"(unique inputs: {local['n_unique_inputs']}/{local['n_rows']})")

        outcome = run_budget(factory, tensors, subset_rows, {
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "eval_every": args.eval_every, "seed": args.seed, "set_seed": set_seed,
            "stop_at_rmse": args.stop_at_rmse, "max_steps": args.max_steps},
            curve_path=directory / "curve.json")

        target = args.target_rmse
        adjusted = target + local_floor
        verdict = ("pass" if outcome["final_eval_rmse_sgd"] <= adjusted
                   else ("plateau" if abs(outcome["second_half_slope_sgd_per_epoch"]) < 0.01
                         else "still-improving"))
        print(f"      -> first {outcome['first_eval_rmse_sgd']:.3f}  "
              f"final {outcome['final_eval_rmse_sgd']:.3f}  "
              f"slope {outcome['second_half_slope_sgd_per_epoch']:+.5f} SGD/epoch  "
              f"verdict {verdict}")

        torch.save({"state_dict": outcome["model"].state_dict(),
                    "config": config.to_dict(),
                    "pooling": "cls", "n_rows": budget_size,
                    "row_ids": tensors.row_id[rows[:budget_size]].tolist(),
                    "state": state.to_dict()}, directory / "model.pt")
        (directory / "curve.json").write_text(
            json.dumps(outcome["curve"], indent=2) + "\n")

        results.append({"run_id": directory.name, "budget": budget_size,
                        "n_rows": outcome["n_rows"],
                        "epochs": args.epochs,
                        "total_steps": outcome["total_steps"],
                        "steps_per_epoch": outcome["steps_per_epoch"],
                        "lr": args.lr, "batch_size": min(args.batch_size, budget_size),
                        "subset_floor_rmse_sgd": local_floor,
                        "subset_unique_inputs": local["n_unique_inputs"],
                        "first_eval_rmse_sgd": outcome["first_eval_rmse_sgd"],
                        "final_eval_rmse_sgd": outcome["final_eval_rmse_sgd"],
                        "final_eval_mae_sgd": outcome["final_eval_mae_sgd"],
                        "final_eval_bias_sgd": outcome["final_eval_bias_sgd"],
                        "second_half_slope_sgd_per_epoch":
                            outcome["second_half_slope_sgd_per_epoch"],
                        "seconds": round(outcome["seconds"], 2),
                        "verdict": verdict,
                        "target_rmse_sgd": target,
                        "floor_adjusted_target_sgd": adjusted})

    # ------------------------------------------------------------ 4. reload
    print("\n--- 4. load-only reload of each budget's final model")
    for entry in results:
        directory = OUT / entry["run_id"]
        stored = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        rebuilt = build_for(state.numeric_names, state.cardinalities(),
                            context_dim=tensors.query_context.shape[1], config=config)
        rebuilt.load_state_dict(stored["state_dict"])
        rebuilt.eval()
        subset = tensors.subset(rows[:entry["budget"]])
        with torch.no_grad():
            raw = [rebuilt(features).cpu().numpy()
                   for features in subset.batches(4096, shuffle=False)]
        predicted = subset.to_sgd(np.concatenate(raw))
        stats = metrics(subset.actual_sgd, predicted)
        np.savez_compressed(directory / "predictions.npz",
                            predicted_sgd=predicted,
                            actual_sgd=subset.actual_sgd,
                            row_id=subset.row_id,
                            horizon=subset.horizon)
        entry["reload_rmse_sgd"] = stats["rmse_sgd"]
        entry["reload_matches"] = bool(abs(stats["rmse_sgd"]
                                           - entry["final_eval_rmse_sgd"]) < 1e-4)
        print(f"      budget {entry['budget']:5d}: reload RMSE {stats['rmse_sgd']:.6f} "
              f"(recorded {entry['final_eval_rmse_sgd']:.6f}) "
              f"match={entry['reload_matches']}")

    payload = {
        "contract_id": manifest["contract_id"],
        "populations": {"A/fit": len(tensors), "A/holdout": "not used in A"},
        "census": summary,
        "config": config.to_dict(),
        "budgets": results,
        "total_seconds": round(time.perf_counter() - started, 1),
        "interpretation": {
            "pass": "final eval RMSE <= target + conditional-mean floor",
            "plateau": "second-half slope magnitude < 0.01 SGD/epoch without passing",
            "plateau_is_not_a_bug": ("a finite-budget plateau means bounded capacity or "
                                     "optimiser reach, not proof of a defect; the next step "
                                     "is gradients/capacity/optimiser in isolation"),
        },
    }
    (OUT / "learnability.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    print("\n" + "=" * 78)
    print(f"{'budget':>8} {'rows':>6} {'steps':>7} {'first':>10} {'final':>10} "
          f"{'MAE':>10} {'bias':>10} {'slope':>12} verdict")
    for entry in results:
        print(f"{entry['budget']:>8} {entry['n_rows']:>6} {entry['total_steps']:>7} "
              f"{entry['first_eval_rmse_sgd']:>10.3f} {entry['final_eval_rmse_sgd']:>10.3f} "
              f"{entry['final_eval_mae_sgd']:>10.3f} {entry['final_eval_bias_sgd']:>+10.3f} "
              f"{entry['second_half_slope_sgd_per_epoch']:>+12.5f} {entry['verdict']}")
    print(f"\ntotal {payload['total_seconds']}s; wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

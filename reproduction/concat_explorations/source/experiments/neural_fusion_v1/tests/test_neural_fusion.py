"""Leakage, alignment, split-boundary and shape tests (neural_fusion_v1).

Run: env -u PYTHONPATH .venv/bin/python -m pytest experiments/neural_fusion_v1/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from data import Dataset, assemble, history_stats, split_frames  # noqa: E402
from models import build_model, group_index_map, run_lstm  # noqa: E402
from panel import Panel, training_samples, validation_samples  # noqa: E402
from preprocessing import StaticPreprocessor, context_matrix  # noqa: E402
from training import (TrainingConfig, band_report, build_and_count, fit_epochs, predict,  # noqa: E402
                      predictions_frame)

TARGET = "MONTHLY_RENT"


def groups(prep):
    """(indices, names) for the numeric token layout, resolved by column name."""
    return group_index_map(prep.numeric_columns)


def synthetic(rows: int = 400, towns=("alpha", "beta"), months=(0, 40), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = rng.integers(months[0], months[1] + 1, rows)
    frame = pd.DataFrame({
        "RENT_APPROVAL_DATE": [config.month_label(int(m)) for m in index],
        "MONTH_INDEX": index.astype(np.int32),
        "TOWN_CLEAN": rng.choice(towns, rows),
        "FLAT_TYPE_CLEAN": rng.choice(["3 room", "4 room"], rows),
        "FLAT_MODEL_CLEAN": rng.choice(["model a", "model b"], rows),
        "FURNISHED_CLEAN": "yes",
        "FLOOR_AREA_SQM": rng.uniform(60, 120, rows),
        "LATITUDE": rng.uniform(1.2, 1.5, rows),
        "LONGITUDE": rng.uniform(103.6, 104.0, rows),
        "MAX_FLOOR": rng.integers(8, 30, rows).astype(float),
        "BUILDING_AGE_YEARS": rng.uniform(0, 40, rows),
        "YEARS_SINCE_LEASE_COMMENCEMENT": rng.uniform(0, 40, rows),
        "APPROVAL_MONTH_INDEX": index.astype(float),
        "APPROVAL_MONTH_SIN": np.sin(2 * np.pi * index / 12),
        "APPROVAL_MONTH_COS": np.cos(2 * np.pi * index / 12),
        "NEAREST_OPEN_MRT_KM": rng.uniform(0.1, 2.0, rows),
        "OPEN_MRT_COUNT_500M": rng.integers(0, 3, rows),
        "OPEN_MRT_COUNT_1000M": rng.integers(0, 5, rows),
        "OPEN_MRT_COUNT_2000M": rng.integers(0, 9, rows),
        "NEAREST_MALL_KM": rng.uniform(0.1, 2.5, rows),
        "MALL_COUNT_500M": rng.integers(0, 2, rows),
        "MALL_COUNT_1000M": rng.integers(0, 3, rows),
        "MALL_COUNT_2000M": rng.integers(0, 6, rows),
        "NEAREST_PRIMARY_SCHOOL_KM": rng.uniform(0.1, 2.0, rows),
        "PRIMARY_SCHOOL_COUNT_500M": rng.integers(0, 2, rows),
        "PRIMARY_SCHOOL_COUNT_1000M": rng.integers(0, 3, rows),
        "PRIMARY_SCHOOL_COUNT_2000M": rng.integers(0, 5, rows),
        "DISTANCE_TO_CBD_KM": rng.uniform(1, 20, rows),
        TARGET: rng.uniform(1200, 4200, rows),
    })
    frame["SOURCE_ROW_ID"] = np.arange(rows)
    return frame


# ---------------------------------------------------------------- split bounds
def test_split_respects_both_cutoff_and_val_end():
    frame = synthetic(600, months=(0, 50))
    cutoff, val_end = 26, 38
    fit, valid = split_frames(frame, cutoff, val_end)
    assert fit["MONTH_INDEX"].max() == cutoff
    assert valid["MONTH_INDEX"].min() == cutoff + 1
    assert valid["MONTH_INDEX"].max() <= val_end, "validation must stop at val_end, not run to the last month"
    assert set(fit["MONTH_INDEX"]) & set(valid["MONTH_INDEX"]) == set()


def test_split_rejects_inverted_bounds():
    frame = synthetic(50)
    with pytest.raises(ValueError):
        split_frames(frame, 30, 30)
    with pytest.raises(ValueError):
        split_frames(frame, 30, 20)


def test_real_split_row_and_month_counts():
    """The two 12-month splits must match the contract's declared counts."""
    path = config.TRAIN_SPATIAL
    frame = pd.read_csv(path)
    frame["MONTH_INDEX"] = ((frame["RENT_APPROVAL_DATE"].str[:4].astype(int) - 2021) * 12
                            + frame["RENT_APPROVAL_DATE"].str[5:7].astype(int) - 1).astype(np.int32)
    for split_id, months in (("orig", 12), ("early", 12), ("long", 16)):
        spec = config.SPLITS[split_id]
        cutoff = config.month_index(spec["cutoff"])
        val_end = config.month_index(spec["val_end"])
        valid = validation_samples(frame, cutoff, val_end)
        assert valid["MONTH_INDEX"].nunique() == months, split_id
        assert valid["MONTH_INDEX"].max() == val_end, split_id
        assert valid["HORIZON"].max() == months, split_id
        if spec["val_rows"] is not None:
            assert len(valid) == spec["val_rows"], f"{split_id}: {len(valid)} != {spec['val_rows']}"
    # the earlier split must NOT be evaluated on orig's 2024-04..2025-03 window
    early = validation_samples(frame, config.month_index("2023-03"), config.month_index("2024-03"))
    assert early["MONTH_INDEX"].max() == config.month_index("2024-03")
    assert len(early) == 35882


# ------------------------------------------------------------- causal panel
def test_panel_excludes_labels_after_cutoff():
    frame = synthetic(600, months=(0, 40))
    cutoff = 20
    base = Panel.build(frame, cutoff)
    later = synthetic(200, months=(cutoff + 1, 40), seed=9)
    later[TARGET] = 99_999.0
    later["SOURCE_ROW_ID"] = np.arange(len(frame), len(frame) + len(later))
    extended = Panel.build(pd.concat([frame, later], ignore_index=True), cutoff)
    assert extended.n_months == base.n_months == cutoff + 1
    for a, b in zip(base.n, extended.n):
        np.testing.assert_array_equal(a, b[:, : a.shape[1]])
    for a, b in zip(base.rent, extended.rent):
        np.testing.assert_allclose(a, b[:, : a.shape[1]], equal_nan=True)


def test_history_window_ends_at_origin_and_padding_is_trailing():
    frame = synthetic(400, months=(0, 40))
    cutoff = 30
    panel = Panel.build(frame, cutoff)
    train = training_samples(frame, cutoff, np.random.default_rng(3), max_horizon=12)
    assert (train["ORIGIN"] >= 0).all()
    assert (train["ORIGIN"] + train["HORIZON"] == train["MONTH_INDEX"]).all()
    assert (train["MONTH_INDEX"] <= cutoff).all()

    codes = panel.codes_for(train)
    channels, mask, summary = panel.windows(codes, train["ORIGIN"].to_numpy())
    assert channels.shape == (len(train), config.HISTORY_LEN, len(config.PANEL_CHANNELS))
    real = mask.astype(bool)
    assert not (real[:, 1:] & ~real[:, :-1]).any(), "padding must be trailing"
    assert (summary["history_months"] == mask.sum(axis=1)).all()

    start = np.maximum(0, train["ORIGIN"].to_numpy() - (config.HISTORY_LEN - 1))
    months = start[:, None] + np.arange(config.HISTORY_LEN)[None, :]
    origin = np.broadcast_to(train["ORIGIN"].to_numpy()[:, None], months.shape)
    assert (months[real] <= origin[real]).all(), "no window slot may exceed the origin"
    # textbook: no slot may equal the target month when h >= 1
    target_month = np.broadcast_to(train["MONTH_INDEX"].to_numpy()[:, None], months.shape)
    assert (months[real] < target_month[real]).all()


def test_validation_windows_frozen_at_cutoff():
    frame = synthetic(500, months=(0, 40))
    cutoff, val_end = 24, 36
    panel = Panel.build(frame, cutoff)
    valid = validation_samples(frame, cutoff, val_end)
    assert (valid["ORIGIN"] == cutoff).all()
    assert (valid["HORIZON"] == valid["MONTH_INDEX"] - cutoff).all()
    codes = panel.codes_for(valid)
    _, mask, summary = panel.windows(codes, valid["ORIGIN"].to_numpy())
    expected = min(config.HISTORY_LEN, cutoff + 1)
    assert np.array_equal(mask.sum(axis=1), np.full(len(valid), expected))


def test_padding_is_zero_and_never_a_real_aggregate():
    frame = synthetic(300, months=(0, 20))
    panel = Panel.build(frame, 5)              # only 6 months of history exist
    valid = validation_samples(frame, 5, 12)
    codes = panel.codes_for(valid)
    channels, mask, summary = panel.windows(codes, valid["ORIGIN"].to_numpy())
    assert (mask.sum(axis=1) == 6).all()
    assert np.all(channels[~mask.astype(bool)] == 0.0)
    assert (summary["anchor_missing"] == 0).all()


# ------------------------------------------------------- unseen category fallback
def test_unseen_town_flat_type_falls_back_instead_of_raising():
    frame = synthetic(400, towns=("alpha", "beta"), months=(0, 40))
    panel = Panel.build(frame, 30)
    unseen = frame.head(4).copy()
    unseen["TOWN_CLEAN"] = "gamma"          # never seen at fit time
    codes = panel.codes_for(unseen)         # must not raise
    assert (codes["exact_group"] < 0).all()
    assert (codes["town"] < 0).all()
    assert (codes["flat_type"] >= 0).all()
    unseen["ORIGIN"] = 30
    unseen["HORIZON"] = 1
    channels, mask, summary = panel.windows(codes, unseen["ORIGIN"].to_numpy())
    assert np.isfinite(channels).all()
    # every slot falls back to the coarse flat-type level (2), never the exact group
    assert (summary["exact_frac"] == 0).all()
    assert mask.sum(axis=1).min() > 0
    assert summary["anchor_missing"].max() == 0
    assert panel.unseen_report(unseen)["unseen_exact_group"] == 4


def test_unseen_flat_type_uses_town_level():
    frame = synthetic(400, towns=("alpha", "beta"), months=(0, 40))
    panel = Panel.build(frame, 30)
    unseen = frame.head(3).copy()
    unseen["FLAT_TYPE_CLEAN"] = "6 room"
    codes = panel.codes_for(unseen)
    assert (codes["flat_type"] < 0).all() and (codes["exact_group"] < 0).all()
    assert (codes["town"] >= 0).all()
    _, mask, summary = panel.windows(codes, np.full(len(unseen), 30))
    assert mask.sum(axis=1).min() > 0


def test_static_categorical_unseen_maps_to_unknown_slot():
    frame = synthetic(200)
    prep = StaticPreprocessor.fit(frame)
    unseen = frame.head(3).copy()
    unseen["TOWN_CLEAN"] = "gamma"
    assert (prep.categorical_matrix(unseen)[:, 0] == 0).all()


# -------------------------------------------------- stage A / stage B isolation
def test_inner_validation_does_not_touch_stage_a_statistics():
    """Changing inner-holdout X must not move stage-A scaling or vocabularies."""
    frame = synthetic(800, months=(0, 40))
    cutoff, inner_cutoff, val_end = 36, 24, 40
    fit_rows = frame.loc[frame.MONTH_INDEX <= inner_cutoff]
    before = StaticPreprocessor.fit(fit_rows)
    panel_before = Panel.build(fit_rows, inner_cutoff)

    tampered = frame.copy()
    holdout = tampered.MONTH_INDEX > inner_cutoff
    tampered.loc[holdout, "FLOOR_AREA_SQM"] *= 3.0
    tampered.loc[holdout, "TOWN_CLEAN"] = "zzz_new_town"
    after = StaticPreprocessor.fit(tampered.loc[tampered.MONTH_INDEX <= inner_cutoff])
    panel_after = Panel.build(tampered.loc[tampered.MONTH_INDEX <= inner_cutoff], inner_cutoff)

    assert before.median == after.median
    assert before.categories == after.categories
    assert before.fitted_rows == after.fitted_rows
    np.testing.assert_allclose(panel_before.rent[0], panel_after.rent[0], equal_nan=True)
    np.testing.assert_allclose(panel_before.channel_mean, panel_after.channel_mean, rtol=0, atol=0)


def test_stage_a_uses_only_months_up_to_inner_cutoff():
    frame = synthetic(700, months=(0, 40))
    inner_cutoff = 24
    rows = frame.loc[frame.MONTH_INDEX <= inner_cutoff]
    panel = Panel.build(rows, inner_cutoff)
    assert panel.n_months == inner_cutoff + 1
    assert panel.cutoff_month == inner_cutoff
    assert panel.scale_until_month <= inner_cutoff - config.HISTORY_LEN


def test_context_is_zero_for_history_columns_in_static_models():
    frame = synthetic(200, months=(0, 30))
    panel = Panel.build(frame, 24)
    valid = validation_samples(frame, 24, 30)
    codes = panel.codes_for(valid)
    _, _, summary = panel.windows(codes, valid["ORIGIN"].to_numpy())
    static = context_matrix(valid["HORIZON"].to_numpy(), valid["MONTH_INDEX"].to_numpy(), summary, False)
    temporal = context_matrix(valid["HORIZON"].to_numpy(), valid["MONTH_INDEX"].to_numpy(), summary, True)
    assert static.shape == temporal.shape == (len(valid), config.CONTEXT_DIM)
    assert np.all(static[:, 9:11] == 0.0)
    assert (temporal[:, 9] > 0).any()
    assert np.allclose(static[:, :9], temporal[:, :9])


# --------------------------------------------------------------- model plumbing
def test_row_alignment_and_ids():
    frame = synthetic(120, months=(0, 30))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 12)
    valid = validation_samples(frame, 12, 20).head(30)
    ds = assemble(valid, prep, panel, with_history=True)
    out = predictions_frame(valid, np.full(len(ds), 2500.0), split_id="orig",
                            cutoff_month="2022-01", model_name="mlp", seed=1)
    assert out["SOURCE_ROW_ID"].is_monotonic_increasing
    assert len(out) == len(valid)
    assert set(out["SOURCE_ROW_ID"]) == set(valid["SOURCE_ROW_ID"])
    horizons = out.set_index("SOURCE_ROW_ID")["horizon_months"]
    assert (horizons.loc[valid["SOURCE_ROW_ID"]] == valid["HORIZON"]).all()
    assert (out["model"] == "mlp").all() and (out["seed"] == 1).all()
    ordered = valid.sort_values("SOURCE_ROW_ID", kind="stable")
    assert np.allclose(out["actual_rent_sgd"], ordered[TARGET].to_numpy())


@pytest.mark.parametrize("name", ["mlp", "static_transformer", "lstm_concat", "lstm_transformer"])
def test_models_forward_backward_and_gradients(name):
    frame = synthetic(160, months=(0, 30))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 20)
    valid = validation_samples(frame, 20, 30).head(24)
    uses_history = name in ("lstm_concat", "lstm_transformer")
    ds = assemble(valid, prep, panel, with_history=uses_history)
    model = build_model(name, prep.cardinalities(), groups(prep)[0], config.CONTEXT_DIM, group_names=groups(prep)[1])
    batch = {k: torch.from_numpy(v) for k, v in ds.arrays.items() if k != "target"}
    prediction = model(batch)
    assert prediction.shape == (len(valid),)
    assert torch.isfinite(prediction).all()
    prediction.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_lstm_ignores_trailing_padding():
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(3, 8, batch_first=True)
    real = torch.randn(2, 5, 3)
    padded = torch.cat([real, torch.randn(2, 3, 3) * 5], dim=1)
    a = run_lstm(lstm, real, torch.ones(2, 5))
    b = run_lstm(lstm, padded, torch.cat([torch.ones(2, 5), torch.zeros(2, 3)], dim=1))
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_models_are_invariant_to_padding_content():
    """Zeroing vs scrambling padded steps must not change the prediction."""
    frame = synthetic(200, months=(0, 10))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 5)           # 6 months of history -> real trailing padding
    valid = validation_samples(frame, 5, 10).head(16)
    ds = assemble(valid, prep, panel, with_history=True)
    assert (ds.arrays["panel_mask"].sum(axis=1) < config.HISTORY_LEN).any()

    for name in ("lstm_concat", "lstm_transformer"):
        torch.manual_seed(0)
        model = build_model(name, prep.cardinalities(), groups(prep)[0], config.CONTEXT_DIM, group_names=groups(prep)[1])
        model.eval()
        clean = {k: torch.from_numpy(v) for k, v in ds.arrays.items() if k != "target"}
        scrambled = dict(clean)
        mask = clean["panel_mask"].unsqueeze(-1)
        scrambled["panel"] = clean["panel"] + (1 - mask) * torch.randn_like(clean["panel"]) * 7.0
        with torch.no_grad():
            a, b = model(clean), model(scrambled)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_empty_horizon_band_is_reported():
    frame = pd.DataFrame({"horizon_months": [1, 2, 5],
                          "actual_rent_sgd": [2000.0, 2100.0, 2200.0],
                          "predicted_rent_sgd": [2050.0, 2050.0, 2050.0]})
    bands = {row["horizon_band"]: row for row in band_report(frame)}
    assert bands["13-16"]["n"] == 0 and bands["13-16"]["rmse_sgd"] is None
    assert bands["1-3"]["n"] == 2


def test_dataset_rejects_ragged_arrays():
    with pytest.raises(ValueError):
        Dataset({"a": np.zeros(3), "b": np.zeros(4)})


# ------------------------------------------------- deterministic training + reload
def test_seed_makes_initialisation_and_training_reproducible():
    frame = synthetic(300, months=(0, 30))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 24)
    rows = training_samples(frame.loc[frame.MONTH_INDEX <= 24], 24, np.random.default_rng(0), 6)
    ds = assemble(rows.head(256), prep, panel, with_history=False)
    cfg = TrainingConfig(model="mlp", epochs=2, batch_size=128, seed=7)

    from training import set_seed

    trajectories = []
    for _ in range(2):
        set_seed(cfg.seed)
        model, _ = build_and_count("mlp", prep.cardinalities(), groups(prep)[0], config.CONTEXT_DIM, cfg,
                                      group_names=groups(prep)[1])
        fit_epochs(model, ds, cfg, torch.device("cpu"), 2)
        trajectories.append(predict(model, ds, torch.device("cpu")))
    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0, atol=1e-6)


def test_fit_epochs_does_not_select_a_checkpoint():
    """Stage B trains exactly N epochs and keeps the LAST weights."""
    frame = synthetic(300, months=(0, 30))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 24)
    rows = training_samples(frame.loc[frame.MONTH_INDEX <= 24], 24, np.random.default_rng(0), 6)
    ds = assemble(rows.head(256), prep, panel, with_history=False)
    cfg = TrainingConfig(model="mlp", epochs=5, batch_size=128, seed=3)
    from training import set_seed

    set_seed(3)
    model, _ = build_and_count("mlp", prep.cardinalities(), groups(prep)[0], config.CONTEXT_DIM, cfg,
                                      group_names=groups(prep)[1])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit_epochs(model, ds, cfg, torch.device("cpu"), 3)
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), "weights must change"
    # no early-stopping artefact is returned: the function is void apart from the loss
    assert isinstance(fit_epochs(model, ds, cfg, torch.device("cpu"), 1), float)


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    frame = synthetic(300, months=(0, 30))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 24)
    valid = validation_samples(frame, 24, 30).head(64)
    ds = assemble(valid, prep, panel, with_history=False)
    cfg = TrainingConfig(model="mlp", epochs=1, batch_size=64, seed=11)
    from training import set_seed

    set_seed(11)
    model, _ = build_and_count("mlp", prep.cardinalities(), groups(prep)[0], config.CONTEXT_DIM, cfg,
                                      group_names=groups(prep)[1])
    fit_epochs(model, ds, cfg, torch.device("cpu"), 1)
    reference = predict(model, ds, torch.device("cpu"))

    ckpt = tmp_path / "model.pt"
    torch.save({"model": "mlp", "state_dict": model.state_dict(),
                "model_kwargs": cfg.model_kwargs(), "cardinalities": prep.cardinalities(),
                "group_indices": groups(prep)[0], "group_names": groups(prep)[1],
                "context_dim": config.CONTEXT_DIM,
                "preprocessing": prep.to_dict()}, ckpt)
    stored = torch.load(ckpt, map_location="cpu", weights_only=False)
    rebuilt = build_model(stored["model"], stored["cardinalities"], stored["group_indices"],
                          stored["context_dim"], group_names=stored["group_names"],
                          **stored["model_kwargs"])
    rebuilt.load_state_dict(stored["state_dict"])
    reloaded_prep = StaticPreprocessor.from_dict(stored["preprocessing"])
    ds2 = assemble(valid, reloaded_prep, panel, with_history=False)
    np.testing.assert_allclose(predict(rebuilt, ds2, torch.device("cpu")), reference, rtol=0, atol=1e-6)


def test_history_stats_counts_short_histories():
    frame = synthetic(200, months=(0, 12))
    prep = StaticPreprocessor.fit(frame)
    panel = Panel.build(frame, 10)
    valid = validation_samples(frame, 10, 12)
    ds = assemble(valid, prep, panel, with_history=True)
    stats = history_stats(ds)
    assert stats["rows"] == len(valid)
    assert 0 <= stats["rows_lt_6_months"] <= stats["rows"]


# --------------------------------------------------- real CLI entry-point wiring
def test_split_state_constructs_on_real_data():
    """Guards the NameError class of bug: SplitState must be constructible from the CLI path."""
    import main as cli

    frame = cli.load_frame(config.TRAIN_SPATIAL)
    state = cli.SplitState(frame, "early", seed=42, mode="probe", inner_months=12)
    assert state.split_id == "early"
    assert len(state.outer_val) == config.SPLITS["early"]["val_rows"]
    assert state.outer_val["MONTH_INDEX"].nunique() == 12
    assert [len(index) for index in state.group_indices] == [2, 3, 4, 13, 2]
    assert sorted(i for index in state.group_indices for i in index) == list(
        range(len(state.prep.numeric_columns)))
    assert state.inner_cutoff == state.cutoff - 12


def test_main_exposes_every_helper_it_calls():
    """A stale function-local import let main.py reference an undefined module global."""
    import main as cli

    for helper in ("group_index_map", "build_model", "set_seed", "SplitState", "run_one",
                   "verify_checkpoint", "prepare_mode", "SplitState"):
        assert hasattr(cli, helper), f"main.py must expose {helper} at module scope"


def test_cli_smoke_end_to_end_with_checkpoint(tmp_path, monkeypatch):
    """Actually execute the CLI on a tiny run and require real artefacts."""
    import json

    import main as cli

    run_id = "pytest-cli"
    run_dir = config.REPORT / "runs" / run_id
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    exit_code = cli.main(["--split", "early", "--models", "mlp", "--seeds", "42",
                          "--mode", "smoke", "--run-id", run_id, "--verify-checkpoints"])
    assert exit_code == 0
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["results"] and summary["results"][0]["n_validation_rows"] > 0
    assert summary["checkpoint_verification"][0]["ok"] is True
    assert (run_dir / "predictions" / "early" / "mlp_seed42.csv").exists()
    assert (run_dir / "checkpoints" / "early" / "mlp_seed42.pt").exists()
    import shutil

    shutil.rmtree(run_dir)

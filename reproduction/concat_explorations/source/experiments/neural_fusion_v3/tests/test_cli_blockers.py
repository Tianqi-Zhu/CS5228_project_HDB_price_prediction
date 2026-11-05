"""Guards for the three blockers a real CLI run found and unit tests did not.

1. `--help` must exit 0 — a missing attribute in an argparse default crashes every
   invocation before any work happens.
2. The macro token source must register its encoders as `nn.Module` submodules, and must
   not assign to `self.modules` (that name is the registry accessor, not a free slot).
3. A checkpoint reload must rebuild the *saved* scalers and never refit them.

The CLI tests exercise the shipped entry point in a subprocess; the object tests build the
real `_Source` class through `main._macro_source` rather than a stand-in.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))

from neural_fusion_v2.settings import CAPACITY  # noqa: E402
from neural_fusion_v3.models import Shapes, build_control  # noqa: E402

PYTHON = str(ROOT / ".venv" / "bin" / "python")
FEATURES = ["FLOOR_AREA_SQM", "LATITUDE", "LONGITUDE", "NEAREST_OPEN_MRT_KM",
            "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]


def _run(args: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "HOME": str(Path.home())}
    return subprocess.run([PYTHON, "-m", "experiments.neural_fusion_v3.main"] + args,
                          cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
                          env=env)


# ------------------------------------------------------------------ blocker 1: the CLI
def test_cli_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0, f"--help failed:\n{result.stderr[-2000:]}"
    assert "--macro" in result.stdout and "--macro-policy" in result.stdout


def test_cli_rejects_an_unknown_contract_with_a_typed_exit_code():
    result = _run(["--contract-id", "fusion-contract-does-not-exist", "--split", "orig"])
    assert result.returncode == 3, result.stderr[-800:]
    assert "CONTRACT NOT READY" in result.stderr


def test_cli_rejects_an_unknown_macro_mode():
    result = _run(["--macro", "dbs+coe"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower() or "unknown" in result.stderr.lower()


def test_cli_macro_policy_default_is_one_of_the_declared_policies():
    import experiments.neural_fusion_v3.macro as macro

    assert isinstance(macro.POLICIES, tuple) and macro.POLICIES
    result = _run(["--help"])
    assert macro.POLICIES[0] in result.stdout, "the default policy must be a real one"


# ----------------------------------------------------------- blocker 2: source registry
def _real_source(monkeypatch, mode: str = "dbs"):
    """Build the actual `_Source` main.py ships, with macro loading stubbed at the edges."""
    import experiments.macro_encoders_v3.api as macro_api
    import experiments.macro_encoders_v3.data as macro_data
    import experiments.macro_encoders_v3.encoders as macro_encoders

    import experiments.neural_fusion_v3.main as main_module

    monkeypatch.setattr(macro_data, "load_windows", lambda *a, **k: object())
    monkeypatch.setattr(macro_api, "build_window_tensors", lambda *a, **k: object())
    monkeypatch.setattr(macro_encoders, "build_encoders", macro_encoders.build_encoders)

    class FakeStage:
        split, stage = "orig", "A"
        scaler = object()
        window_index = __import__("numpy").zeros(4, dtype="int64")

    return main_module._macro_source(FakeStage(), {"contract_id": "test"}, mode)


@pytest.mark.parametrize("mode,expected", [("dbs", 1), ("coe", 1), ("both", 2)])
def test_real_source_is_an_nn_module_with_registered_encoders(monkeypatch, mode, expected):
    source = _real_source(monkeypatch, mode)
    assert isinstance(source, torch.nn.Module)
    assert source.n_tokens == expected
    # the registry must be intact and contain the encoders
    registered = dict(source.named_modules())
    assert "encoders" in registered, sorted(registered)
    assert "encoders.dbs" in registered and "encoders.coe" in registered
    assert callable(source.modules), "`modules` must still be the nn.Module accessor"
    params = dict(source.named_parameters())
    assert any(name.startswith("encoders.") for name in params), sorted(params)[:5]


def test_modules_attribute_would_be_shadowed_by_state():
    """Documents the exact failure mode the fix avoids, so nobody 'simplifies' it back.

    Verified against torch 2.8: `modules` is a *class-level method*, so instance attribute
    lookup finds the bound method before `nn.Module.__getattr__` is ever consulted.
    Assigning `self.modules = <MacroEncoderSet>` therefore does **not** make `self.modules`
    return the encoder set — it still returns the bound method, and the encoder set is
    silently parked in `_modules["modules"]` (so it also gets a `modules.` state-dict
    prefix). A source written that way hands a *method* to `encode_batch`.
    """
    state = torch.nn.Module()
    state.encoders = torch.nn.Linear(2, 2)
    assert callable(state.modules) and list(state.modules()) != []

    broken = torch.nn.Module()
    broken.modules = torch.nn.Linear(2, 2)          # type: ignore[assignment]
    assert "modules" in broken._modules, "the assignment registers a submodule"
    assert callable(broken.modules), (
        "reading `self.modules` still yields the bound method, not the assigned module")
    assert not isinstance(broken.modules, torch.nn.Linear), (
        "this is the bug: what is read back is not what was assigned")
    with pytest.raises(AttributeError):
        broken.modules.dbs                         # the read that would crash a real source


def test_optimizer_sees_every_encoder_parameter(monkeypatch):
    source = _real_source(monkeypatch, "both")
    model = build_control("static_transformer",
                          Shapes(n_categorical=4, numeric_dim=len(FEATURES), history_dim=24,
                                 context_dim=11, cardinalities=(27, 7, 21, 2)),
                          FEATURES, token_mode="block")
    from experiments.neural_fusion_v3.fusion_macro import MacroFusion

    fused = MacroFusion(model, source, "both")
    optimizer = torch.optim.SGD(fused.parameters(), lr=0.1)
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in source.named_parameters():
        assert id(parameter) in covered, f"{name} is not in the optimizer"


# ------------------------------------------------------------- blocker 3: reload refits?
def test_build_stage_macro_does_not_refit_when_given_scalers(monkeypatch):
    from experiments.macro_encoders_v3 import contract as macro_contract
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import scaling as macro_scaling

    import experiments.neural_fusion_v3.macro_bridge as bridge

    manifest = macro_contract.load_manifest(macro_contract.latest_v3_contract_dir())
    windows = macro_data.load_windows(manifest, "orig", "A")
    rows = macro_data.load_rows(manifest, "orig", "A")
    scalers = macro_scaling.fit_macro_scalers(windows, rows)      # fitted before the guard

    def forbid(*args, **kwargs):
        raise AssertionError("build_stage_macro refitted the scalers during a reload")

    monkeypatch.setattr(macro_scaling, "fit_macro_scalers", forbid)
    result = bridge.build_stage_macro(manifest, "orig", "A", scalers=scalers)
    assert result.scaler is scalers
    assert result.n_rows == len(rows.row_id)


def test_scalers_round_trip_through_a_checkpoint_payload():
    from experiments.macro_encoders_v3 import contract as macro_contract
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import scaling as macro_scaling
    from experiments.macro_encoders_v3.scaling import MacroScalers

    import experiments.neural_fusion_v3.main as main_module

    manifest = macro_contract.load_manifest(macro_contract.latest_v3_contract_dir())
    windows = macro_data.load_windows(manifest, "orig", "B")
    rows = macro_data.load_rows(manifest, "orig", "B")
    original = macro_scaling.fit_macro_scalers(windows, rows)

    stored = {"macro": {
        "contract_id": manifest["contract_id"], "policy": manifest["policy"],
        "stage_b": {"split": "orig", "stage": "B"},
        "scalers_b": {name: getattr(original, name).to_dict()
                      for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")},
    }}
    rebuilt = main_module._scalers_from_checkpoint(stored, "B")
    assert isinstance(rebuilt, MacroScalers)
    for name in ("dbs", "dbs_scalar", "coe", "coe_scalar"):
        a, b = getattr(original, name), getattr(rebuilt, name)
        assert a.channel_names == b.channel_names
        assert a.mean.tolist() == b.mean.tolist()
        assert a.scale.tolist() == b.scale.tolist()
        assert a.fitted_on == b.fitted_on
    assert rebuilt.split_id == "orig" and rebuilt.stage == "B"


def test_a_saved_checkpoint_carries_the_scalers_it_trained_with():
    """Whatever a macro run writes must be enough to reload it without refitting."""
    summaries = sorted((ROOT / "reports" / "neural_fusion_v3" / "runs").glob("*/run_summary.json"))
    macro_summaries = []
    for path in summaries:
        payload = json.loads(path.read_text())
        if payload.get("macro", {}).get("enabled"):
            macro_summaries.append((path.parent, payload))
    if not macro_summaries:
        pytest.skip("no macro run has completed yet")
    run_dir, payload = macro_summaries[-1]
    tag = (f"{payload['results'][0]['control']}__{payload['results'][0]['token_mode']}"
           f"__macro-{payload['results'][0]['macro_mode']}")
    checkpoint = torch.load(run_dir / "checkpoints" / payload["split"] / f"{tag}_seed42.pt",
                            map_location="cpu", weights_only=False)
    assert "macro" in checkpoint
    for key in ("scalers_a", "scalers_b", "policy", "contract_id"):
        assert key in checkpoint["macro"], key

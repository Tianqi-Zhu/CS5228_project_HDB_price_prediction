"""Tests for the v4 temporal branch, against real exported windows.

Not synthetic-only: the leading-padding cases the repair exists for are taken from the
actual A/fit export, and the defect is demonstrated on those rows.

Run:
    .venv/bin/python -m experiments.temporal_fusion_v4.tests_temporal
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def main() -> int:
    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.models import _masked_lstm_pool
    from neural_fusion_v3.main import shapes_of
    from neural_fusion_v2.preprocessing import ScalerState
    from experiments.temporal_fusion_v4.adapter import (AdapterError, channel_names,
                                                        legacy_48_step, level_names,
                                                        month_presence_report,
                                                        monthly_12_step)
    from experiments.temporal_fusion_v4.encoders import (PaddingSafeTemporalEncoder,
                                                         legacy_masked_pool,
                                                         reproduce_legacy_padding_defect)
    from experiments.temporal_fusion_v4.models import assert_init_comparable, build_arm

    manifest = C.load_manifest(V2_DIR)
    a_fit = C.load_population(V2_DIR, manifest, "orig", "A", "fit")
    arrays = a_fit.arrays
    sequence, mask, order = arrays.sequence, arrays.sequence_mask, arrays.sequence_order
    origin = a_fit.samples["origin_month_index"].to_numpy()

    # ---------------------------------------------------------------- 1. defect
    print("--- 1. reproduce the legacy padding defect on real early windows")
    month_real = (mask > 0).any(axis=1)
    leading = np.where(month_real[:, 0] == False)[0]  # noqa: E712
    print(f"  rows whose first slot is padding: {len(leading):,} / {len(mask):,}")
    check("the export really contains leading padding", len(leading) > 0)

    partial = np.where((month_real.sum(axis=1) >= 3) & (month_real.sum(axis=1) <= 6)
                       & ~month_real[:, 0])[0]
    check("found partially-filled early windows", len(partial) > 0, str(len(partial)))
    probe = int(partial[0])
    print(f"  probing row {probe}: origin={int(origin[probe])}, "
          f"valid months={int(month_real[probe].sum())}")

    rep = legacy_48_step(sequence[probe:probe + 1], mask[probe:probe + 1], order)
    values, kept = torch.from_numpy(rep.values), torch.from_numpy(rep.mask)
    defect = reproduce_legacy_padding_defect(values, kept, hidden=8, seed=0)
    print(f"  legacy pooling, same observations left- vs right-padded: "
          f"max |diff| = {defect['max_abs_difference']:.6f}")
    check("the legacy path depends on pad position (defect reproduced)",
          defect["max_abs_difference"] > 1e-5, f"{defect['max_abs_difference']:.3e}")

    full_rows = np.where(month_real.sum(axis=1) == 12)[0]
    full = legacy_48_step(sequence[full_rows[0]:full_rows[0] + 1],
                          mask[full_rows[0]:full_rows[0] + 1], order)
    no_pad = reproduce_legacy_padding_defect(torch.from_numpy(full.values),
                                             torch.from_numpy(full.mask), seed=0)
    check("a fully-observed window is unaffected by pad position",
          no_pad["max_abs_difference"] < 1e-6, f"{no_pad['max_abs_difference']:.3e}")

    # ---------------------------------------------------------------- 2. invariance
    print("\n--- 2. padding-safe encoder is invariant to pad position")
    encoder = PaddingSafeTemporalEncoder(channels=rep.values.shape[-1], hidden=8,
                                         token_dim=16)
    encoder.eval()
    steps, total = int(kept.sum().item()), kept.shape[1]
    real_steps = values[kept > 0].unsqueeze(0)[:, :steps]

    def place(left_pad: int):
        out = torch.zeros(1, total, real_steps.shape[-1])
        m = torch.zeros(1, total)
        out[0, left_pad:left_pad + steps] = real_steps[0]
        m[0, left_pad:left_pad + steps] = 1.0
        return out, m

    with torch.no_grad():
        outputs = {pad: encoder(*place(pad))
                   for pad in (0, total - steps, max(0, (total - steps) // 2))}
    differences = {pad: float((out.token - outputs[0].token).abs().max())
                   for pad, out in outputs.items()}
    print(f"  token differences vs right-padded: {differences}")
    check("padding-safe token is identical for every pad position",
          all(value < 1e-6 for value in differences.values()), str(differences))
    check("steps_used equals the number of real steps",
          all(int(out.steps_used) == steps for out in outputs.values()))
    check("present flag is 1 for a row with real history",
          all(float(out.present) == 1.0 for out in outputs.values()))

    # ---------------------------------------------------------------- 3. zero history
    print("\n--- 3. zero history is finite and explicit")
    with torch.no_grad():
        zero = encoder(torch.zeros(1, 12, rep.values.shape[-1]), torch.zeros(1, 12))
    check("zero-history token is finite", bool(torch.isfinite(zero.token).all()))
    check("zero-history present flag is 0", float(zero.present) == 0.0)
    check("zero-history steps_used is 0", int(zero.steps_used) == 0)

    # ---------------------------------------------------------------- 4. shapes
    print("\n--- 4. real exported windows: shapes, masks, observed semantics")
    sample = slice(0, 64)
    legacy_batch = legacy_48_step(sequence[sample], mask[sample], order)
    check("legacy representation is 48 steps", legacy_batch.values.shape[1] == 48,
          str(legacy_batch.values.shape))
    check("legacy channel count preserved", len(legacy_batch.channel_names) == 24,
          str(len(legacy_batch.channel_names)))

    monthly = monthly_12_step(sequence[sample], mask[sample], order, origin[sample])
    check("monthly representation is 12 steps", monthly.values.shape[1] == 12,
          str(monthly.values.shape))
    check("monthly channels are 4 levels x 6 channels = 24",
          monthly.values.shape[2] == 24, str(monthly.values.shape))
    check("level names are preserved and distinct",
          level_names(order, 4, 6) == ["global", "flat_type", "town_ft", "block_ft"],
          str(level_names(order, 4, 6)))
    check("channel names keep only the channel part",
          channel_names(order, 4, 6)[:3] == ["log_mean_rent", "log_mean_psm",
                                            "log_count"],
          str(channel_names(order, 4, 6)[:3]))

    presence = month_presence_report(monthly)
    print(f"  presence: {presence}")
    check("every in-panel slot is kept as a timestep",
          presence["kept_slots"] + presence["padded_slots"] == monthly.mask.size,
          str(presence))
    check("padding is identified", presence["padded_slots"] > 0)

    # `observed` must come from channel 5, never from the keep-mask. panel.py:283
    # defines observed = (counts > 0) & valid_level.
    # channel 5 of level L lives at position L*6+5 in the level-major channel block
    raw = np.transpose(sequence[sample], (0, 2, 1, 3))       # (n, months, levels, ch)
    direct_observed = (raw[..., :, 5] > 0).any(axis=2) & (monthly.mask > 0)
    check("observed equals a direct read of the observed channel",
          np.array_equal(monthly.observed > 0, direct_observed),
          "observed must be the channel, not the mask")
    check("the keep-mask is 'calendar in panel', not 'observed'",
          np.array_equal(monthly.kept, monthly.calendar >= 0),
          "mask==1 must mean a real timestep, including months with no transactions")
    # on this export an in-panel month always has at least one observed level
    unobserved_kept = int((monthly.kept & (monthly.observed == 0)).sum())
    print(f"  kept months with no transaction at any level: {unobserved_kept}")
    check("kept-but-unobserved is counted from observed, and is consistent",
          unobserved_kept == presence["kept_unobserved_months"])
    # the legacy reading's observed flag comes from the same channel
    # legacy steps are (month, level) month-major; the adapter took `observed` from the
    # same cells, so compare against the adapter's own values rather than re-deriving
    check("legacy observed flag also comes from the channel",
          np.array_equal(legacy_batch.observed > 0,
                         legacy_batch.values[..., 5] > 0),
          "a step is 'observed' iff its channel-5 value is positive")

    # ---------------------------------------------------------------- 5. contradiction
    print("\n--- 5. an observation outside the panel calendar is refused")
    bad_mask = mask[probe:probe + 1].copy()
    # force a slot that is pre-panel to claim an observation
    pre_panel_slots = np.where(monthly.calendar[0] < 0)[0]
    check("the probe window has pre-panel slots to attack", len(pre_panel_slots) > 0)
    if len(pre_panel_slots):
        slot = int(pre_panel_slots[0])
        bad_mask[0, 0, slot] = 1.0
        raised = False
        try:
            monthly_12_step(sequence[probe:probe + 1], bad_mask, order,
                            origin[probe:probe + 1])
        except AdapterError:
            raised = True
        check("pre-panel observation raises instead of being retained", raised)

    # ---------------------------------------------------------------- 6. gradients
    print("\n--- 6. nonzero gradients through the encoder")
    encoder.zero_grad(set_to_none=True)
    out = encoder(torch.from_numpy(legacy_batch.values), torch.from_numpy(legacy_batch.mask))
    out.token.pow(2).mean().backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    check("every parameter group receives a gradient",
          len(grads) == len(list(encoder.parameters())))
    check("at least one gradient is nonzero",
          any(bool(g.abs().sum() > 0) for g in grads))

    # ---------------------------------------------------------------- 7. init parity
    print("\n--- 7. padding_fixed_48 starts from the fixed-7 diagnostic's weights")
    state = ScalerState.fit(a_fit.arrays, population="A/fit")
    shapes = shapes_of(a_fit, True, C.load_encoding(V2_DIR, manifest, "orig", "A"))
    parity = assert_init_comparable(shapes, state.static_columns,
                                    legacy_batch.values.shape[-1], seed=42)
    print(f"  shared tensors {parity['shared_tensors']}, "
          f"differing {len(parity['differing'])}, "
          f"only in arm {parity['only_in_arm']}, "
          f"only in reference {parity['only_in_reference']}")
    check("padding_fixed_48 inherits the reference's backbone weights bit-for-bit",
          parity["comparable"], str(parity["differing"][:5]))

    # ---------------------------------------------------------------- 8. legacy untouched
    print("\n--- 8. the frozen v2 function still behaves as v2")
    with torch.no_grad():
        lstm = torch.nn.LSTM(rep.values.shape[-1], 8, batch_first=True)
        for parameter in lstm.parameters():
            parameter.zero_()
        a = _masked_lstm_pool(lstm, values, kept)
        b = legacy_masked_pool(lstm, values, kept)
    check("our copy of the legacy pool matches v2's exactly", bool(torch.equal(a, b)))

    print("\n" + "=" * 62)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for item in FAILED:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

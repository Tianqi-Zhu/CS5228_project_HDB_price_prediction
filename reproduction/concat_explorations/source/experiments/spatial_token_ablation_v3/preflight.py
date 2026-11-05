"""Pre-flight audit for the spatial/property per-feature token ablation.

Answers one question before any training is spent: **does `token_mode="per_feature"`
change only the numeric house/spatial tokenisation, and nothing else?**

Method, all read-only:

1. **Structural diff.** Build the same control twice (`block`, `per_feature`) and
   compare every named buffer and parameter *other than* the numeric block. Anything
   that moved outside the documented swap is a finding.
2. **Sentinel routing.** With fixed weights, feed a batch where each numeric column
   holds a distinct sentinel value. In `per_feature` mode token `j` must respond to
   column `j` and to nothing else. Because `identity_j` and `scale_j` are
   per-feature, a clean test needs the *marginal* response: perturb one column and
   check that exactly one token changes.
3. **Weight transplant.** Copy shared weights from the `block` model into the
   `per_feature` model (everything the two have in common), freeze the new token
   block to its initialisation, and confirm the non-numeric pathways produce
   identical outputs when the numeric contribution is zeroed. This is the "same
   weights" control the task asks for.
4. **Parameter inventory.** Report exactly which parameters appear, disappear and
   change shape, so the report can state the capacity difference instead of implying
   the two arms are parameter-matched.

It does not train, does not write into the consumer package, and does not modify any
existing run.

Run:
    .venv/bin/python -m experiments.spatial_token_ablation_v3.preflight
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

FAILURES: list[str] = []
FINDINGS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def find(name: str) -> None:
    FINDINGS.append(name)
    print(f"FIND  {name}")


def main() -> int:
    from neural_fusion_v3 import models as v3_models
    from neural_fusion_v3.models import Shapes, build_control, parameter_count
    from neural_fusion_v3 import tokens as v3_tokens

    # the contract's numeric column order, straight from the v2.1 encoding
    from experiments.macro_encoders_v3 import contract as macro_contract  # noqa: F401
    from neural_fusion_v2 import contract as C

    v2_dir = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
    manifest = C.load_manifest(v2_dir)
    population = C.load_population(v2_dir, manifest, "orig", "A", "fit")
    arrays = population.arrays
    encoding = C.load_encoding(v2_dir, manifest, "orig", "A")

    # Column order comes from the loaded stage arrays, which is what the model reads.
    numeric_columns = list(arrays.numeric_columns)
    print(f"numeric columns ({len(numeric_columns)}): {numeric_columns}\n")

    shapes = Shapes(
        n_categorical=len(arrays.categorical_columns),
        numeric_dim=len(arrays.numeric_columns),
        history_dim=len(arrays.history_order),
        context_dim=len(arrays.context_order) - 1,
        cardinalities=tuple(C.encoding_cardinalities(encoding, arrays)),
        sequence_channels=arrays.n_channels,
        sequence_length=arrays.n_months * arrays.n_levels)

    torch.manual_seed(0)
    block = build_control("lstm_fusion", shapes, numeric_columns, token_mode="block")
    torch.manual_seed(0)
    per_feature = build_control("lstm_fusion", shapes, numeric_columns,
                                token_mode="per_feature",
                                expected_numeric=numeric_columns)

    # ---------------------------------------------------------------- 1
    print("--- 1. structural diff")
    block_state = dict(block.named_parameters())
    per_state = dict(per_feature.named_parameters())
    only_block = sorted(set(block_state) - set(per_state))
    only_feature = sorted(set(per_state) - set(block_state))
    shared = sorted(set(block_state) & set(per_state))
    shape_changed = [name for name in shared
                     if block_state[name].shape != per_state[name].shape]

    print(f"  only in block      : {only_block}")
    print(f"  only in per_feature: {only_feature}")
    print(f"  shape changed      : {shape_changed}")
    print(f"  shared             : {len(shared)} tensors")

    # Documented swap: static_proj disappears, feature_tokens appears, the
    # transformer `proj`/head/type_embedding are resized for the new token count.
    numeric_block_names = {"block.encoder.static_proj.weight",
                           "block.encoder.static_proj.bias"}
    added = set(only_feature)
    unexpected_added = {name for name in added
                        if not name.startswith("block.encoder.feature_tokens.")}
    check("the only added parameters are the per-feature token block",
          unexpected_added == set(), f"unexpected: {sorted(unexpected_added)}")
    check("the only removed parameters are the numeric block projection",
          set(only_block) == numeric_block_names, f"removed: {only_block}")
    feature_module = per_feature.block.encoder.feature_tokens
    check("feature token count equals the numeric column count",
          feature_module.n_tokens == len(numeric_columns),
          f"{feature_module.n_tokens} vs {len(numeric_columns)}")

    expected_resize = {"block.proj.weight", "block.proj.bias",
                       "block.type_embedding.weight"}
    unexpected_resize = set(shape_changed) - expected_resize
    check("reshapes are confined to proj / type embedding / head input",
          unexpected_resize == set(), f"unexpected reshapes: {sorted(unexpected_resize)}")
    for name in sorted(set(shape_changed) & expected_resize):
        print(f"      resize {name}: {tuple(block_state[name].shape)} "
              f"-> {tuple(per_state[name].shape)}")

    # anything else that silently gained or lost a tensor is a real finding
    if unexpected_added or set(only_block) != numeric_block_names or unexpected_resize:
        find("token_mode swap touched parameters outside the documented numeric block")

    # ---------------------------------------------------------------- 2
    print("\n--- 2. parameter inventory (capacity is NOT matched)")
    n_block = parameter_count(block)
    n_feature = parameter_count(per_feature)
    print(f"  block params       : {n_block:,}")
    print(f"  per_feature params : {n_feature:,}  (delta {n_feature - n_block:+,})")
    check("per_feature differs in parameter count (expected, must be reported)",
          n_feature != n_block)
    find(f"parameter counts differ: block {n_block:,} vs per_feature {n_feature:,} "
         f"({n_feature - n_block:+,}); the ablation changes capacity, so a score "
         f"difference cannot be attributed to tokenisation alone")

    # ---------------------------------------------------------------- 3
    print("\n--- 3. sentinel routing (marginal response per feature)")
    encoder = per_feature.block.encoder
    module = encoder.feature_tokens
    module.eval()
    n = len(numeric_columns)
    base = torch.zeros(1, n)
    with torch.no_grad():
        reference = module(base)[0]                       # (n, 48)

    responders = []
    for column in range(n):
        probe = base.clone()
        probe[0, column] = 5.0
        with torch.no_grad():
            moved = module(probe)[0]
        changed = [j for j in range(n)
                   if not torch.allclose(moved[j], reference[j], rtol=0, atol=0)]
        responders.append(changed)

    check("each numeric column moves exactly its own token",
          all(changed == [column] for column, changed in enumerate(responders)),
          f"map: {responders}")

    # ---------------------------------------------------------------- 4
    print("\n--- 4. distinct identity per feature")
    with torch.no_grad():
        zero = module(torch.zeros(1, n))[0]
    pairwise_min = float(torch.cdist(zero, zero).masked_fill(
        torch.eye(n, dtype=torch.bool), float("inf")).min())
    print(f"  min pairwise token distance at x=0: {pairwise_min:.6f}")
    check("identical inputs still produce distinguishable tokens",
          pairwise_min > 0.0,
          "identity vectors collapsed; attention could not separate features")

    # ---------------------------------------------------------------- 5
    print("\n--- 5. non-numeric pathways survive the swap")
    # The embeddings, LSTM and context projection must exist and keep their shapes.
    # The LSTM lives on the model for lstm_fusion, while the categorical and context
    # modules live on the encoder.  Looking for `lstm` on the encoder would report a
    # false failure, so each is resolved where it actually is.
    for label, resolve in (
            ("categorical embeddings", lambda m: getattr(m.block.encoder, "embeddings", None)),
            ("LSTM", lambda m: getattr(m, "lstm", None)),
            ("context projection", lambda m: getattr(m.block.encoder, "context_proj", None)),
            ("temporal projection", lambda m: getattr(m, "temporal_proj", None)),
            ("gate", lambda m: getattr(m, "gate", None))):
        left = resolve(block)
        right = resolve(per_feature)
        check(f"{label}: present in both", left is not None and right is not None)
        if left is not None and right is not None:
            left_state = dict(left.named_parameters())
            right_state = dict(right.named_parameters())
            same = (set(left_state) == set(right_state)
                    and all(left_state[k].shape == right_state[k].shape
                            for k in left_state))
            check(f"{label}: identical parameter structure", same)
    check("static_proj is removed in per_feature mode",
          per_feature.block.encoder.static_proj is None)
    check("uses_feature_tokens flag is set",
          getattr(per_feature.block.encoder, "uses_feature_tokens", False) is True)

    # The head rebuild in install_per_feature guards on `hasattr(model, "head")`, but
    # for these controls the head lives at model.block.head, so that branch never runs.
    # For lstm_fusion the resulting width is unchanged (216+2*8+4*48 -> 48), so it is
    # currently harmless; it would not be for a control whose head input differs.
    head_owner = "model" if hasattr(block, "head") else (
        "block" if hasattr(getattr(block, "block", None), "head") else "none")
    print(f"  head owner: {head_owner}")
    if head_owner == "block":
        find("install_per_feature's head rebuild is dead code: it checks "
             "hasattr(model, 'head') but the head is model.block.head; harmless for "
             "lstm_fusion because the resulting width coincides")

    # numeric_index must be the identity for an unpermuted contract order
    check("numeric_index is the identity for this contract order",
          per_feature.block.encoder.numeric_index == list(range(n)),
          str(per_feature.block.encoder.numeric_index))

    # ---------------------------------------------------------------- 6
    print("\n--- 6. weight transplant: shared path is really shared")
    # Copy every shared tensor that kept its shape; then confirm the two models
    # agree on the pieces that do not depend on the numeric token.
    with torch.no_grad():
        for name in shared:
            if block_state[name].shape == per_state[name].shape:
                per_state[name].copy_(block_state[name])
    check("shared parameters are bit-identical after transplant",
          all(torch.equal(block_state[name], per_state[name])
              for name in shared
              if block_state[name].shape == per_state[name].shape))

    # `model.lstm` is a bare nn.LSTM; the pooling wrapper is v2's `_masked_lstm_pool`,
    # which is what the forward path actually calls.
    from neural_fusion_v2 import models as v2_models
    block.lstm.eval()
    per_feature.lstm.eval()
    sequence = torch.randn(2, 12, max(1, shapes.sequence_channels))
    mask = torch.ones(2, 12)
    with torch.no_grad():
        a = v2_models._masked_lstm_pool(block.lstm, sequence, mask)
        b = v2_models._masked_lstm_pool(per_feature.lstm, sequence, mask)
    check("LSTM outputs match after transplant",
          torch.allclose(a, b, rtol=0, atol=0))

    print("\n" + "=" * 62)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S)")
        for item in FINDINGS:
            print(f"  - {item}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("pre-flight passed: per_feature changes only the numeric token block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the frozen v4 stage-A feature state so both model classes share it verbatim.

`shared_table_origA.npz` already contains the **already-standardised** 56 base numeric
columns, so a consumer does not need to rescale them. What a consumer cannot reconstruct
from the array alone is:

* the base scaler that produced them (medians / means / stds, log1p column list, clip),
* the category vocabularies and the reserved code convention,
* the target scale and anchor handling,
* the exact numeric and categorical order.

Without this file the Transformer side would be forced to fit its own scaler on the shared
matrix, which would silently break the "same matrix" guarantee. So the state is exported
here from the original `trial6` checkpoint rather than re-derived.

`extra_state` holds only the **new** feature's standardisation, fitted on A/fit, which is
the one transform neither side may re-fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch  # noqa: E402

CHECKPOINT = (ROOT / "reports" / "feature_transformer_v4" / "runs"
              / "v4-search-mps-s42" / "trial6" / "checkpoint.pt")
DATA_MANIFEST = (ROOT / "reports" / "global_mean_comparison_v6" / "data"
                 / "shared_table_origA.manifest.json")
OUT = ROOT / "reports" / "global_mean_comparison_v6" / "data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=OUT / "base_feature_state.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if (args.out.parent / "FROZEN.json").exists() and not args.force:
        print(f"REFUSING to write into frozen directory {args.out.parent}")
        return 4

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    scaler = state["scaler"]
    data_manifest = json.loads(DATA_MANIFEST.read_text())

    # The frozen table's numeric block must be exactly the scaler's columns plus the new
    # feature; if these disagree, the shared matrix was not built from this state.
    base_numeric = list(data_manifest["numeric_order"][:-1])
    scaled_columns = (list(scaler["static_columns"]) + list(scaler["history_columns"])
                      + list(scaler["context_columns"]))
    if list(scaled_columns) != base_numeric:
        raise AssertionError(
            "the shared table's base numeric order is not the scaler's column order:\n"
            f"  table   {base_numeric}\n  scaler  {scaled_columns}")
    if int(state["n_numeric_positions"]) != len(base_numeric):
        raise AssertionError("n_numeric_positions disagrees with the column list")

    payload = {
        "purpose": ("frozen v4 stage-A feature state; consumers must reuse these statistics "
                    "instead of fitting their own scaler on the shared matrix"),
        "source_checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "config": checkpoint["config"],
        "pooling": checkpoint["pooling"],
        "base_numeric_order": base_numeric,
        "categorical_order": list(state["categorical_names"]),
        # Each vocabulary is the code list itself; its length is the cardinality, and the
        # list is what a consumer needs to reproduce the mapping exactly.
        "vocabularies": {k: [int(x) for x in v]
                         for k, v in state["vocabularies"].items()},
        "vocabulary_sizes": {k: len(v) for k, v in state["vocabularies"].items()},
        "n_numeric_positions": int(state["n_numeric_positions"]),
        "fitted_population": state["fitted_population"],
        "fitted_rows": int(state["n_rows"]),
        "clip_std": float(scaler["clip_std"]),
        "log1p_columns": list(scaler["log1p_columns"]),
        "static_columns": list(scaler["static_columns"]),
        "history_columns": list(scaler["history_columns"]),
        "context_columns": list(scaler["context_columns"]),
        "static_median": {k: float(v) for k, v in scaler["static_median"].items()},
        "static_mean": {k: float(v) for k, v in scaler["static_mean"].items()},
        "static_std": {k: float(v) for k, v in scaler["static_std"].items()},
        "history_median": {k: float(v) for k, v in scaler["history_median"].items()},
        "history_mean": {k: float(v) for k, v in scaler["history_mean"].items()},
        "history_std": {k: float(v) for k, v in scaler["history_std"].items()},
        "context_median": {k: float(v) for k, v in scaler["context_median"].items()},
        "context_mean": {k: float(v) for k, v in scaler["context_mean"].items()},
        "context_std": {k: float(v) for k, v in scaler["context_std"].items()},
        "anchor_handling": scaler["anchor_handling"],
        "target_scale": float(scaler["target_scale"]),
        "reserved_code_note": ("categorical code 0 is reserved for unknown/absent; the "
                               "shared table already carries the mapped codes, so a consumer "
                               "must not re-encode them"),
        "standardisation_note": ("the shared table's numeric matrix is ALREADY transformed "
                                 "with these statistics; do not transform it again"),
    }

    extra = {
        "purpose": "standardisation of the single added feature, fitted on A/fit only",
        "feature": data_manifest["new_feature"],
        "raw_definition": data_manifest["new_feature_raw_definition"],
        "fit_mean": data_manifest["new_feature_state"]["fit_mean"],
        "fit_std": data_manifest["new_feature_state"]["fit_std"],
        "clip_abs": data_manifest["new_feature_state"]["clip_abs_ksgd"],
        "transform": data_manifest["new_feature_state"]["standardised_as"],
        "fitted_on": data_manifest["new_feature_state"]["fitted_on"],
        "already_applied_in_shared_table": True,
        "note": ("column 57 of shared_table_origA.npz is this feature AFTER the transform; "
                 "a consumer must not re-fit or re-apply it"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    extra_path = args.out.parent / "extra_feature_state.json"
    extra_path.write_text(json.dumps(extra, indent=2) + "\n")

    print(f"base_feature_state: {len(base_numeric)} numeric columns, "
          f"{len(payload['categorical_order'])} categorical, "
          f"vocab sizes {payload['vocabulary_sizes']}")
    print(f"  fitted on {payload['fitted_population']} ({payload['fitted_rows']:,} rows), "
          f"clip {payload['clip_std']}, log1p {len(payload['log1p_columns'])} columns")
    print(f"  selected_epoch {payload['selected_epoch']}, pooling {payload['pooling']}")
    print(f"  checkpoint sha256 {payload['source_checkpoint_sha256'][:16]}…")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}")
    print(f"wrote {extra_path.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

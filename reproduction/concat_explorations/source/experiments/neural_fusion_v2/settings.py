"""Settings for the Gate 1 control comparison.

Only things this pane controls live here: paths, the frozen model capacity, which
controls run, and the target scale. Split definitions, feature lists, sample
generation and every target-derived input belong to the contract owner and are read
from the contract manifest at runtime — they are never duplicated here.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
REPORT_ROOT = ROOT / "reports" / "neural_fusion_v2"
V1_ROOT = ROOT / "experiments" / "neural_fusion_v1"

# --- frozen capacity for every Gate 1 control (do not enlarge) ---------------
CAPACITY = {
    "d_model": 48,
    "layers": 2,
    "heads": 4,
    "lstm_hidden": 32,
    "mlp_hidden": 128,
}

CONTROLS = ["mlp", "static_transformer", "lstm_concat", "lstm_fusion"]

# Target scale: the networks minimise MSE in kSGD, metrics are reported in SGD.
TARGET_SCALE = 1000.0

# Training defaults; the epoch count is chosen on the inner chronological holdout.
TRAIN = {
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 60,
    "patience": 8,
    "clip_grad": 1.0,
    "seed": 42,
    "threads": 2,
}

MODES = {
    "smoke": {"epochs": 3, "patience": 2, "batch_size": 256, "row_cap": 4000},
    "probe": {"epochs": 20, "patience": 5, "batch_size": 512, "row_cap": None},
    "full": {"epochs": 60, "patience": 8, "batch_size": 512, "row_cap": None},
}


def contract_dir(contract_id: str) -> Path:
    return CONTRACT_ROOT / contract_id

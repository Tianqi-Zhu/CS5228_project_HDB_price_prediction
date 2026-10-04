"""Shared, row-preserving input for the controlled spatial studies."""

from __future__ import annotations

import numpy as np

from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.config import DATA, STUDY_REPORTS
from analysis.pipeline.data import load_main


SOURCE = DATA / "train.csv"
OUTPUT = STUDY_REPORTS / "spatial_features"


def load_spatial_training():
    """Build the canonical spatial frame directly from immutable source CSVs."""
    frame = build_auxiliary_features(
        load_main(SOURCE, require_target=True), include_mixed_primary=True,
    )
    frame["SOURCE_ROW_ID"] = np.arange(len(frame))
    frame["APPROVAL_MONTH"] = frame["APPROVAL_MONTH_OF_YEAR"]
    return frame

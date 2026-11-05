"""Separate human-readable reports from disposable caches and model binaries."""

from pathlib import Path

from analysis.pipeline.config import CACHE, EXPERIMENT_REPORTS, MODELS


def model_artifact_dir(report_dir: Path) -> Path:
    """Mirror default experiment folders under output/models; custom runs stay local."""
    directory = Path(report_dir).resolve()
    reports = EXPERIMENT_REPORTS.resolve()
    if directory.is_relative_to(reports):
        return MODELS / directory.relative_to(reports)
    return directory / "artifacts"


def feature_cache_path(report_dir: Path, fold: str, profile: str) -> Path:
    directory = Path(report_dir).resolve()
    if directory == (EXPERIMENT_REPORTS / "comparison_model_search").resolve():
        return CACHE / "comparison_model_search" / fold / profile / "prepared.joblib"
    return directory / ".cache" / fold / profile / "prepared.joblib"

"""Unified CLI: ``python -m analysis.main --help``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.pipeline.config import FEATURE_SETS, MODEL_FAMILIES, REPORTS, VALIDATION_START
from analysis.pipeline.data import check_output_path
from analysis.pipeline.training import TrainerConfig


def feature_set(value):
    aliases = {"A": "A_property", "B": "B_property_time", "C": "C_property_time_town"}
    value = aliases.get(value.upper(), value)
    if value not in FEATURE_SETS:
        raise argparse.ArgumentTypeError("Use A, B, C or a full feature-set name")
    return value


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def add_model_options(parser):
    parser.add_argument("--model", choices=MODEL_FAMILIES, default="lightgbm")
    parser.add_argument("--features", type=feature_set, default="C_property_time_town", metavar="A|B|C")
    parser.add_argument("--fit-window-months", type=positive_integer, help="Use only this many recent fitting months")


def run_eda(scope="all", output_dir=None):
    from analysis.eda.run import main as eda_main
    return eda_main(scope=scope, output_dir=output_dir)


def run_baseline(output_dir=None):
    from analysis.experiments.baseline import main as baseline
    return baseline(output_dir=output_dir)


def run_compare(output_dir=None, baseline_dir=None):
    from analysis.experiments.compare_models import main as compare
    baseline_dir = Path(baseline_dir) if baseline_dir is not None else REPORTS / "model_baseline"
    if not (baseline_dir / "results.json").exists():
        run_baseline(baseline_dir)
    return compare(output_dir=output_dir, baseline_dir=baseline_dir)


def run_report(eda_dir=None, output_path=None):
    from analysis.eda.pdf_report import main as report
    return report(eda_dir=eda_dir, output_path=output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CS5228 HDB rent: EDA, experiments, temporal validation and submission")
    commands = parser.add_subparsers(dest="command")
    eda = commands.add_parser("eda", help="Run consolidated EDA into one reports/eda tree")
    eda.add_argument("--scope", choices=("main", "auxiliary", "overview", "all"), default="all")
    eda.add_argument("--output-dir", type=Path, help="EDA root directory; defaults to reports/eda")
    baseline = commands.add_parser("baseline", help="Reproduce the mean and CatBoost A/B/C feature comparison")
    baseline.add_argument("--output-dir", type=Path)
    compare = commands.add_parser("compare", help="Compare Ridge, Random Forest, CatBoost and LightGBM")
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--baseline-dir", type=Path)
    validation = commands.add_parser("validate", help="Time validation with monthly, town and flat-type errors")
    add_model_options(validation)
    validation.add_argument("--validation-start", default=VALIDATION_START, metavar="YYYY-MM")
    validation.add_argument("--output-dir", type=Path)
    submission = commands.add_parser("submit", help="Refit and write Id,Predicted; does not upload to Kaggle")
    add_model_options(submission)
    submission.add_argument("--output", type=Path)
    submission.add_argument("--model-path", type=Path)
    prediction = commands.add_parser("predict", help="Generate predictions using a saved submit-command artifact")
    prediction.add_argument("--model-path", type=Path, required=True)
    prediction.add_argument("--output", type=Path)
    report = commands.add_parser("report", help="Rebuild the preserved EDA progress PDF")
    report.add_argument("--eda-dir", type=Path)
    report.add_argument("--output", type=Path)
    deep = commands.add_parser("deep", help="Train a GPU tabular deep-learning experiment")
    deep.add_argument("--variant", choices=("main", "auxiliary"), default="auxiliary")
    deep.add_argument("--architecture", choices=("resnet", "mlp", "transformer"), default="resnet")
    deep.add_argument("--validation-start", default=VALIDATION_START)
    deep.add_argument("--fit-window-months", type=positive_integer)
    deep.add_argument("--recency-half-life", type=float, default=24.0)
    deep.add_argument("--target-encoding", action="store_true")
    deep.add_argument("--target-encoding-window", type=positive_integer, default=18)
    deep.add_argument("--macro-auxiliary", action="store_true")
    deep.add_argument("--residual-baseline", action="store_true")
    deep.add_argument("--residual-warmup-months", type=positive_integer, default=12)
    deep.add_argument("--width", type=positive_integer, default=256)
    deep.add_argument("--blocks", type=positive_integer, default=3)
    deep.add_argument("--dropout", type=float, default=0.10)
    deep.add_argument("--batch-size", type=positive_integer, default=1024)
    deep.add_argument("--epochs", type=positive_integer, default=40)
    deep.add_argument("--learning-rate", type=float, default=1e-3)
    deep.add_argument("--weight-decay", type=float, default=1e-4)
    deep.add_argument("--patience", type=positive_integer, default=7)
    deep.add_argument("--seed", type=int, default=42)
    deep.add_argument("--output-dir", type=Path)
    boost = commands.add_parser("aux-boost", help="Train the auxiliary LightGBM benchmark")
    boost.add_argument("--num-leaves", type=positive_integer, default=31)
    boost.add_argument("--seed", type=int, default=42)
    boost.add_argument("--without-target-encoding", action="store_true")
    boost.add_argument("--output-dir", type=Path)
    deep_summary = commands.add_parser("deep-summary", help="Summarize deep models and rebuild ensembles")
    deep_summary.add_argument("--output-dir", type=Path)
    all_steps = commands.add_parser("all", help="Run all EDA, both historical experiments and the PDF report")
    all_steps.add_argument("--run-dir", type=Path, help="Write a fresh run separately from historical reports")
    all_steps.add_argument("--submit", action="store_true", help="Also refit and write a local competition submission")
    add_model_options(all_steps)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    # Validate destination paths before expensive training or directory creation.
    for attr in ("output_dir", "baseline_dir", "output", "run_dir", "model_path"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(args, attr, check_output_path(value))
    if args.command == "eda":
        return run_eda(args.scope, args.output_dir)
    if args.command == "baseline":
        return run_baseline(args.output_dir)
    if args.command == "compare":
        return run_compare(args.output_dir, args.baseline_dir)
    if args.command in ("validate", "submit"):
        from analysis.pipeline.workflow import run_validation, run_submission
        config = TrainerConfig(model_family=args.model, feature_set=args.features,
                               fit_window_months=args.fit_window_months,
                               validation_start=getattr(args, "validation_start", VALIDATION_START))
        if args.command == "validate":
            return run_validation(config, args.output_dir)
        return run_submission(config, args.output, args.model_path)
    if args.command == "predict":
        from analysis.pipeline.workflow import predict_saved
        return predict_saved(args.model_path, args.output)
    if args.command == "report":
        return run_report(args.eda_dir, args.output)
    if args.command == "deep":
        from analysis.experiments.deep_learning import ExperimentConfig, train
        values = {key: value for key, value in vars(args).items()
                  if key not in ("command", "output_dir")}
        config = ExperimentConfig(**values)
        label = f"{config.variant}_{config.architecture}_seed{config.seed}"
        return train(config, args.output_dir or REPORTS / "deep_learning" / label)
    if args.command == "aux-boost":
        from analysis.experiments.auxiliary_boosting import run
        return run(args.output_dir or REPORTS / "deep_learning" / "aux_lightgbm",
                   num_leaves=args.num_leaves, seed=args.seed,
                   target_encoding=not args.without_target_encoding)
    if args.command == "deep-summary":
        from analysis.experiments.summarize_deep_learning import run
        return run(output=args.output_dir)
    run_dir = args.run_dir
    run_eda(output_dir=None if run_dir is None else run_dir / "eda")
    base_dir = None if run_dir is None else run_dir / "model_baseline"
    run_baseline(base_dir)
    run_compare(None if run_dir is None else run_dir / "model_families", base_dir)
    run_report(None if run_dir is None else run_dir / "eda",
               None if run_dir is None else run_dir / "eda" / "progress_report.pdf")
    if args.submit:
        from analysis.pipeline.workflow import run_submission
        config = TrainerConfig(model_family=args.model, feature_set=args.features,
                               fit_window_months=args.fit_window_months)
        run_submission(config, None if run_dir is None else run_dir / "submission.csv",
                       None if run_dir is None else run_dir / "model.joblib")


if __name__ == "__main__":
    main()

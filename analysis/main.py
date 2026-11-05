"""Unified CLI: ``python -m analysis.main --help``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.pipeline.config import (
    EXPERIMENT_REPORTS, FEATURE_SETS, MODEL_FAMILIES, VALIDATION_START,
)


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
    baseline_dir = (
        Path(baseline_dir) if baseline_dir is not None
        else EXPERIMENT_REPORTS / "baseline_mean_catboost"
    )
    if not (baseline_dir / "results.json").exists():
        run_baseline(baseline_dir)
    return compare(output_dir=output_dir, baseline_dir=baseline_dir)


def run_report(eda_dir=None, output_path=None):
    from analysis.eda.pdf_report import main as report
    return report(eda_dir=eda_dir, output_path=output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CS5228 HDB rent: EDA, experiments, temporal validation and submission")
    commands = parser.add_subparsers(dest="command")
    from analysis.experiment_catalog import GROUPS
    experiment_list = commands.add_parser("experiment-list", help="List local experiment modules without running them")
    experiment_list.add_argument("--group", choices=GROUPS)
    experiment_list.add_argument("--json", action="store_true", dest="as_json")
    from analysis.resource_preflight import PROFILES
    preflight = commands.add_parser("preflight", help="Check resources without importing models, installing or training")
    preflight.add_argument("--profile", choices=PROFILES, default="core")
    preflight.add_argument("--json", action="store_true", dest="as_json")
    artifacts = commands.add_parser("artifact-list", help="List current model files, fusion parameters and metric references without loading models")
    artifacts.add_argument("--json", action="store_true", dest="as_json")
    artifacts.add_argument("--hashes", action="store_true")
    artifacts.add_argument("--write", action="store_true")
    artifacts.add_argument("--usage", choices=("all", "final", "development", "submission_component", "discarded"), default="all")
    commands.add_parser("experiment-refresh", aliases=["reports-refresh"], help="Refresh selected experiment summaries from existing results; no training")
    eda = commands.add_parser("eda", help="Run consolidated EDA into one experiment/eda tree")
    eda.add_argument("--scope", choices=("main", "auxiliary", "overview", "all"), default="all")
    eda.add_argument("--output-dir", type=Path, help="EDA root directory; defaults to experiment/eda")
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
    verify = commands.add_parser("verify", help="Audit environment and saved-result consistency")
    verify.add_argument("--output", type=Path, help="Optional JSON audit output")
    feature_selection = commands.add_parser("feature-select", help="Compare feature policies with one LightGBM model and fixed-origin folds")
    feature_selection.add_argument("--output-dir", type=Path)
    feature_optimization = commands.add_parser("feature-optimize", help="Audit redundancy and test causal history and market representations")
    feature_optimization.add_argument("--output-dir", type=Path)
    feature_optimization.add_argument("--stage", choices=("features", "market", "refined", "audit", "all"), default="all")
    feature_evidence = commands.add_parser("feature-evidence", help="Retrain single-column/group deletions and report temporal error diagnostics")
    feature_evidence.add_argument("--output-dir", type=Path)
    common_models = commands.add_parser("common-models", help="Compare architectures with the same 27 feature inputs and outer training rows")
    common_models.add_argument("--output-dir", type=Path)
    common_models.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    canonical = commands.add_parser("canonical-submit", help="Refit the fixed 27-feature LightGBM baseline and export locally (no upload)")
    canonical.add_argument("--output-dir", type=Path)
    commands.add_parser("project-review", help="Build the current project report and evidence index from completed results")
    model_search = commands.add_parser("model-search", help="Nested temporal comparison of tree and official neural models")
    model_search.add_argument("action", choices=("prepare", "run", "summarize", "diagnose", "blend", "report"))
    model_search.add_argument("--fold", choices=("2024-04", "2023-04", "2022-04", "2023-12"), default="2024-04")
    model_search.add_argument("--families", nargs="+")
    model_search.add_argument("--max-epochs", type=positive_integer, default=40)
    model_search.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    model_search.add_argument("--output-dir", type=Path)
    source_audit = commands.add_parser("source-audit", help="Audit overlap with public HDB rental disclosures, separate from forecasting")
    source_audit.add_argument("--download", action="store_true")
    source_audit.add_argument("--cache", type=Path)
    source_audit.add_argument("--output-dir", type=Path)
    commands.add_parser("aux-ablation", help="Run staged spatial/economic feature ablation")
    commands.add_parser("economic-signals", help="Run prior-month economic signal analysis")
    commands.add_parser("economic-ablation", help="Compare engineered economic features")
    commands.add_parser("group-temporal", help="Run grouped temporal panel analysis")
    commands.add_parser("spatial-controls", help="Run spatial overview, fixed-effect and matching studies")
    commands.add_parser("spatiotemporal", help="Run expanding-window historical/spatial forecasts")
    tabm = commands.add_parser("tabm", help="Run the TabM-style neural/tree experiment")
    tabm.add_argument("--epochs", type=positive_integer, default=40)
    tabm.add_argument("--batch-size", type=positive_integer, default=2048)
    tabm.add_argument("--ensemble-size", type=positive_integer, default=8)
    tabm.add_argument("--hidden-dim", type=positive_integer, default=256)
    tabm.add_argument("--depth", type=positive_integer, default=3)
    tabm.add_argument("--dropout", type=float, default=0.10)
    tabm.add_argument("--learning-rate", type=float, default=1e-3)
    tabm.add_argument("--weight-decay", type=float, default=1e-5)
    tabm.add_argument("--seed", type=int, default=42)
    tabm.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tabm.add_argument("--max-folds", type=int, choices=(1, 2, 3))
    tabm.add_argument("--skip-final", action="store_true")
    tabm.add_argument("--output-dir", type=Path)
    relu = commands.add_parser("relu", help="Run nested-validation sklearn ReLU MLP")
    relu.add_argument("--max-epochs", type=positive_integer, default=80)
    relu.add_argument("--patience", type=positive_integer, default=12)
    relu.add_argument("--seed", type=int, default=42)
    relu.add_argument("--write-submission", action="store_true")
    relu.add_argument("--output-dir", type=Path)
    all_steps = commands.add_parser("all", help="Run all EDA, both historical experiments and the PDF report")
    all_steps.add_argument("--run-dir", type=Path, help="Write a fresh run separately from historical reports")
    all_steps.add_argument("--submit", action="store_true", help="Also refit and write a local competition submission")
    add_model_options(all_steps)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "experiment-list":
        from analysis.experiment_catalog import show
        return show(args.group, args.as_json)
    if args.command in ("experiment-refresh", "reports-refresh"):
        from analysis.report_bundle import refresh
        return refresh()
    if args.command == "preflight":
        from analysis.resource_preflight import main as preflight_main
        return preflight_main(["--profile", args.profile] + (["--json"] if args.as_json else []))
    if args.command == "artifact-list":
        from analysis.artifact_catalog import main as artifacts_main
        options = ["--usage", args.usage]
        for flag in ("json", "hashes", "write"):
            if getattr(args, "as_json" if flag == "json" else flag):
                options.append("--" + flag)
        return artifacts_main(options)
    from analysis.pipeline.data import check_output_path
    from analysis.pipeline.training import TrainerConfig
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
        return train(config, args.output_dir or EXPERIMENT_REPORTS / "baseline_deep_models" / label)
    if args.command == "aux-boost":
        from analysis.experiments.auxiliary_boosting import run
        return run(args.output_dir or EXPERIMENT_REPORTS / "baseline_deep_models" / "aux_lightgbm",
                   num_leaves=args.num_leaves, seed=args.seed,
                   target_encoding=not args.without_target_encoding)
    if args.command == "deep-summary":
        from analysis.experiments.summarize_deep_learning import run
        return run(output=args.output_dir)
    if args.command == "verify":
        from analysis.verification import main as verify_main
        return verify_main([] if args.output is None else ["--output", str(args.output)])
    if args.command == "feature-select":
        from analysis.experiments.feature_selection import run
        return run(args.output_dir or EXPERIMENT_REPORTS / "features_selection")
    if args.command == "feature-optimize":
        from analysis.experiments.feature_optimization import main as optimize
        options = ["--stage", args.stage]
        if args.output_dir is not None:
            options += ["--output-dir", str(args.output_dir)]
        return optimize(options)
    if args.command == "feature-evidence":
        from analysis.experiments.feature_evidence import run
        return run(args.output_dir or EXPERIMENT_REPORTS / "features_ablation_evidence")
    if args.command == "common-models":
        from analysis.experiments.common_feature_models import run
        return run(args.output_dir or EXPERIMENT_REPORTS / "comparison_same_27_features", args.device)
    if args.command == "canonical-submit":
        from analysis.experiments.canonical_final import run
        from analysis.pipeline.config import ROOT
        return run(args.output_dir or ROOT / "output/submissions/baselines/baseline_27_features")
    if args.command == "project-review":
        from analysis.project_review import run
        return run()
    if args.command == "model-search":
        from analysis.experiments.model_search import main as search
        options = [args.action, "--fold", args.fold, "--max-epochs", str(args.max_epochs), "--device", args.device]
        if args.output_dir is not None:
            options += ["--output-dir", str(args.output_dir)]
        if args.families is not None:
            options += ["--families", *args.families]
        return search(options)
    if args.command == "source-audit":
        from analysis.studies.source_overlap import main as audit_source
        options = ["--download"] if args.download else []
        if args.cache is not None:
            options += ["--cache", str(args.cache)]
        if args.output_dir is not None:
            options += ["--output-dir", str(args.output_dir)]
        return audit_source(options)
    if args.command == "aux-ablation":
        from analysis.experiments.auxiliary_ablation import main as run
        return run()
    if args.command == "economic-signals":
        from analysis.studies.economic_signals import main as run
        return run()
    if args.command == "economic-ablation":
        from analysis.experiments.economic_feature_ablation import main as run
        return run()
    if args.command == "group-temporal":
        from analysis.studies.group_temporal_panel import main as run
        return run()
    if args.command == "spatial-controls":
        from analysis.studies.spatial.run import main as run
        return run()
    if args.command == "spatiotemporal":
        from analysis.experiments.spatiotemporal_hybrid import main as run
        return run()
    if args.command == "tabm":
        from analysis.experiments.neural_tabm import main as run
        values = vars(args).copy(); values.pop("command")
        argv = []
        for key, value in values.items():
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            if value is True:
                argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        return run(argv)
    if args.command == "relu":
        from analysis.experiments.relu_mlp import run
        return run(
            args.output_dir or EXPERIMENT_REPORTS / "baseline_relu_mlp", max_epochs=args.max_epochs,
            patience=args.patience, seed=args.seed, write_submission=args.write_submission,
        )
    run_dir = args.run_dir
    run_eda(output_dir=None if run_dir is None else run_dir / "eda")
    base_dir = None if run_dir is None else run_dir / "baseline_mean_catboost"
    run_baseline(base_dir)
    run_compare(None if run_dir is None else run_dir / "baseline_classical_models", base_dir)
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

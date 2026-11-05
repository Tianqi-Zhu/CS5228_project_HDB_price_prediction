"""Single entry point for every EDA layer.

The overview layer writes the canonical midterm figures, tables and
summary to ``experiment/eda``. The two focused audits are retained under
``experiment/eda/supplementary`` because they use different units of analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.pipeline.config import DATA, EDA_REPORTS
from analysis.pipeline.data import check_output_path, protect_sources

from analysis.eda import overview
from analysis.eda.audits import auxiliary, main_data


DEFAULT_OUT = EDA_REPORTS


def main(scope: str = "all", output_dir: Path | str | None = None) -> None:
    root = check_output_path(Path(output_dir) if output_dir is not None else DEFAULT_OUT)
    destinations = {
        "main": root / "supplementary" / "train_test",
        "auxiliary": root / "supplementary" / "auxiliary",
        "overview": root,
    }
    modules = {
        "main": main_data,
        "auxiliary": auxiliary,
        "overview": overview,
    }
    selected = modules if scope == "all" else {scope: modules[scope]}
    with protect_sources(sorted(DATA.rglob("*.csv"))):
        for name, module in selected.items():
            print(f"Running {name} EDA -> {destinations[name]}", flush=True)
            module.main(output_dir=destinations[name])


def cli(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the consolidated HDB EDA")
    parser.add_argument("--scope", choices=("main", "auxiliary", "overview", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, help="Root directory; defaults to experiment/eda")
    args = parser.parse_args(argv)
    main(args.scope, args.output_dir)


if __name__ == "__main__":
    cli()

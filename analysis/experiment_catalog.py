"""List local experiment entry points without importing or running experiments."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path


GROUPS = (
    "baseline", "features", "models", "report", "frontier",
    "wave02", "wave03", "wave04", "wave05", "wave06", "wave07",
    "wave08", "wave09", "wave10", "final", "other",
)
WAVES = dict(zip(
    ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"),
    GROUPS[5:14],
))


def experiment_group(name):
    if name in {"best_ensemble_submission", "final_submission_selection"}:
        return "final"
    for prefix, group in WAVES.items():
        if name.startswith(prefix + "_wave_"):
            return group
    if name.startswith("frontier_"):
        return "frontier"
    if name.startswith("report_"):
        return "report"
    if name.startswith("feature_"):
        return "features"
    if name in {"model_search", "common_feature_models", "canonical_final"}:
        return "models"
    if name in {"baseline", "compare_models", "auxiliary_ablation", "auxiliary_boosting", "auxiliary_catboost"}:
        return "baseline"
    return "other"


def describe(path):
    """Parse source only: some legacy modules perform work when imported."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    executable = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(isinstance(op, ast.Eq) for op in node.test.ops)
        and any(isinstance(value, ast.Constant) and value.value == "__main__"
                for value in node.test.comparators)
        for node in tree.body
    )
    module = "analysis.experiments." + path.stem
    summary = (ast.get_docstring(tree) or "").splitlines()
    return {
        "module": module,
        "source": (path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
                   if path.resolve().is_relative_to(Path(__file__).resolve().parents[1]) else path.name),
        "group": experiment_group(path.stem),
        "summary": summary[0] if summary else "",
        "executable": executable,
        "command": "python -m " + module if executable else None,
    }


def list_experiments(group=None):
    rows = [describe(path) for path in sorted((Path(__file__).parent / "experiments").rglob("*.py"))
            if path.stem != "__init__"]
    return [row for row in rows if group is None or row["group"] == group]


def show(group=None, as_json=False):
    rows = list_experiments(group)
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    print("Local experiment index (listing only; no experiment is imported or run)")
    counts = Counter(row["group"] for row in rows)
    print("  ".join(f"{name}: {counts[name]}" for name in GROUPS if counts[name]))
    for row in rows:
        entry = row["command"] or (row["module"] + " [support module; no script entry]")
        print(f"[{row['group']}] {entry}")
    print("Arguments and prerequisites: docs/guides/running_experiments.md")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    show(args.group, args.as_json)


if __name__ == "__main__":
    main()

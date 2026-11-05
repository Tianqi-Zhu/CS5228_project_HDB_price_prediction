"""Review isolated TabPFN sources without overlaying the validated F1 package."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "workflows/tabpfn/manifest.json"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def verify(bundle):
    failures = []
    for item in bundle["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append({"path": item["path"], "issue": "missing or outside repository"})
        elif digest(path) != item["sha256"]:
            failures.append({"path": item["path"], "issue": "hash mismatch"})
    return {"files_checked": len(bundle["files"]), "failures": failures,
            "f2_status": bundle["f2"]["status"], "training_performed": False}


def metric_audit(bundle):
    lookup = {item["original_path"]: ROOT / item["path"] for item in bundle["reports"]}
    with (ROOT / "data/train.csv").open(encoding="utf-8", newline="") as stream:
        population = collections.Counter(row["RENT_APPROVAL_DATE"] for row in csv.DictReader(stream))
    checks, omitted = [], []
    for report in [*bundle["main_background_reports"], "tabpfn_abcde_ensemble"]:
        scores = lookup.get(f"reports/{report}/scores.csv")
        monthly = lookup.get(f"reports/{report}/monthly_scores.csv")
        if scores is None or monthly is None:
            continue
        with scores.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with monthly.open(encoding="utf-8", newline="") as stream:
            months = list(csv.DictReader(stream))
        for row in rows:
            if row.get("origin", "2024-06") != "2024-06":
                omitted.append({"report": report, "model": row["model"], "origin": row["origin"],
                                "reason": "Sampled historical folds require their original population weights and row predictions."})
                continue
            selected = [m for m in months if m["model"] == row["model"]
                        and m.get("origin") == row.get("origin")]
            if not selected:
                raise ValueError(f"Missing monthly scores: {report} / {row['model']}")
            counts = [int(m.get("n") or m.get("rows") or population[m["month"]]) for m in selected]
            count = sum(counts)
            if count != int(row.get("n") or row["rows"]):
                raise ValueError(f"Monthly counts disagree: {report} / {row['model']}")
            values = {"rmse_sgd": math.sqrt(sum(n * float(m["rmse_sgd"]) ** 2 for n, m in zip(counts, selected)) / count)}
            for metric in ["mae_sgd", "bias_sgd"]:
                values[metric] = sum(n * float(m[metric]) for n, m in zip(counts, selected)) / count
            differences = {key: abs(value - float(row[key])) for key, value in values.items()}
            checks.append({"report": report, "model": row["model"], "rows": count,
                           "reaggregated": values, "absolute_differences": differences,
                           "passed": all(v < 1e-5 for v in differences.values())})
    return {"checks": checks, "all_passed": bool(checks) and all(x["passed"] for x in checks),
            "omitted": omitted,
            "scope": "Reaggregate saved monthly summaries only; this does not reproduce model inference, row alignment or independent validation.",
            "f2_evaluated": False}


def preflight(bundle, workspace=None):
    execution_root = workspace_path(workspace or ROOT / "output/tabpfn-workspace")
    lookup = {item["original_path"]: ROOT / item["path"] for item in bundle["reports"]}
    issues = []
    for original, path in lookup.items():
        if path.name != "settings.json":
            continue
        settings = json.loads(path.read_text(encoding="utf-8"))
        for group in ["source_sha256", "code_sha256"]:
            values = settings.get(group, {})
            if not isinstance(values, dict):
                continue
            for relative, expected in values.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
                relative = relative.replace("\\", "/")
                target = execution_root / relative
                if not target.is_file() and not relative.startswith(".cache/"):
                    target = (ROOT / "workflows/tabpfn" / relative if relative.startswith("analysis/")
                              else lookup.get(relative, ROOT / relative))
                if not target.is_file():
                    issues.append({"settings": original, "path": relative, "issue": "resource not shipped"})
                elif digest(target) != expected:
                    raw = target.read_bytes()
                    lf = raw.replace(b"\r\n", b"\n")
                    newline_match = any(hashlib.sha256(candidate).hexdigest() == expected
                                        for candidate in (lf, lf.replace(b"\n", b"\r\n")))
                    issues.append({"settings": original, "path": relative,
                                   "issue": "newline-only byte difference" if newline_match
                                   else "recorded bytes differ beyond newline conversion",
                                   "expected_sha256": expected})
    checkpoint = execution_root / ".cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors"
    return {"tabpfn_installed": importlib.util.find_spec("tabpfn") is not None,
            "execution_workspace": str(execution_root),
            "local_checkpoint_present": checkpoint.is_file(), "legacy_resource_issues": issues,
            "f2_status": bundle["f2"]["status"], "fresh_training_verified": False,
            "scope": "Local inspection only; no downloads, authentication or paid API calls."}


def workspace_path(value):
    target = Path(value).resolve()
    output = (ROOT / "output").resolve()
    if not target.is_relative_to(output) or target == output:
        raise ValueError("Workspace must be a child of this repository's output/ directory.")
    return target


def prepare(bundle, value):
    target = workspace_path(value)
    target.mkdir(parents=True, exist_ok=True)
    def copy(source, relative):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(source) != digest(destination):
                raise ValueError(f"Existing workspace file differs; choose a new workspace: {destination}")
        else:
            shutil.copyfile(source, destination)
    for item in bundle["files"]:
        relative = item["path"]
        if relative.startswith("workflows/tabpfn/"):
            suffix = relative[len("workflows/tabpfn/"):]
            if suffix != "manifest.json":
                copy(ROOT / relative, suffix)
        elif item.get("source_kind") == "derived-dataset":
            copy(ROOT / relative, relative)
    for item in bundle["reports"]:
        copy(ROOT / item["path"], item["original_path"])
    for source in (ROOT / "data").rglob("*.csv"):
        if "processed" not in source.relative_to(ROOT / "data").parts:
            copy(source, source.relative_to(ROOT))
    marker = {"bundle_manifest_sha256": digest(BUNDLE), "branch": bundle["branch"],
              "note": "Generated workspace. Original root analysis and archived evidence remain separate."}
    (target / ".tabpfn-workspace.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return target


def get_workspace(bundle, value):
    target = workspace_path(value)
    marker = target / ".tabpfn-workspace.json"
    if not marker.is_file():
        return prepare(bundle, target)
    if json.loads(marker.read_text(encoding="utf-8"))["bundle_manifest_sha256"] != digest(BUNDLE):
        raise ValueError("Workspace belongs to another source snapshot; choose a new output directory.")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ["list", "verify", "metrics", "preflight"]:
        command = commands.add_parser(action)
        if action == "preflight":
            command.add_argument("--workspace", type=Path, default=ROOT / "output/tabpfn-workspace")
    make = commands.add_parser("prepare", help="Copy source, shared data and archived reports into output/")
    make.add_argument("--workspace", type=Path, default=ROOT / "output/tabpfn-workspace")
    test = commands.add_parser("test", help="Run this branch's isolated, original TabPFN tests")
    test.add_argument("--workspace", type=Path, default=ROOT / "output/tabpfn-workspace")
    run = commands.add_parser("run", help="Run an explicitly selected original local entry point")
    run.add_argument("--workspace", type=Path, default=ROOT / "output/tabpfn-workspace")
    run.add_argument("entry")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    bundle = load()
    if args.action == "list":
        print(json.dumps({"branch": bundle["branch"], "entrypoints": bundle["entrypoints"], "f2": bundle["f2"]}, indent=2))
    elif args.action == "verify":
        result = verify(bundle)
        print(json.dumps(result, indent=2))
        return int(bool(result["failures"]))
    elif args.action == "metrics":
        result = metric_audit(bundle)
        print(json.dumps(result, indent=2))
        return int(not result["all_passed"])
    elif args.action == "preflight":
        print(json.dumps(preflight(bundle, args.workspace), indent=2))
    elif args.action == "prepare":
        print(prepare(bundle, args.workspace))
    else:
        if args.action == "test":
            target = get_workspace(bundle, args.workspace)
            command = [sys.executable, "run_original_tests.py"]
        else:
            if args.entry not in bundle["entrypoints"]:
                parser.error("Entry is not included in this branch; use 'list'.")
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            if not arguments:
                parser.error("Pass explicit original arguments, for example: -- --help")
            target = get_workspace(bundle, args.workspace)
            if not any(arg in arguments for arg in ["--help", "-h", "--prepare-only"]):
                resources = preflight(bundle, target)
                if not resources["tabpfn_installed"] or not resources["local_checkpoint_present"]:
                    parser.error("Local TabPFN package/checkpoint absent. See 'preflight' and the workflow guide.")
            command = [sys.executable, bundle["entrypoints"][args.entry], *arguments]
        return subprocess.call(command, cwd=target, env=os.environ.copy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check the complete shipped main tree without importing archived trainers."""
from pathlib import Path
import argparse
import ast
import json
import re

from analysis.pipeline.config import ROOT
from analysis.validation.prepare_inputs import sha


def audit():
    manifest = json.loads((ROOT / "review/manifest.json").read_text(encoding="utf-8"))
    paths = [item["path"] for item in manifest["files"]]
    failures, syntax, links = [], 0, 0
    if len(paths) != len(set(paths)):
        failures.append({"issue": "duplicate manifest paths"})
    active_docs = {"README.md", "data/README.md", "requirements/README.md", "review/CHECKS.md", "experiment/main/README.md"}
    active_docs.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "analysis").rglob("README.md"))
    active_docs.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs/guides").glob("*.md"))
    active_docs.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs/planning").glob("*.md"))
    active_docs.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs/methods").glob("*.md"))
    active_docs.update({'experiment/main/M01/spatial_evidence/README.md',
                        'experiment/main/M01/cleaning_evidence/README.md',
                        'experiment/main/M03/feature_schemas/README.md', 'reproduction/f1/README.md',
                        'experiment/main/M01/eda/README.md', 'tests/README.md'})
    active_docs.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "experiment/main").glob("M*/README.md"))
    for item in manifest["files"]:
        p = (ROOT / item["path"]).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            failures.append({"path": item["path"], "issue": "missing or unsafe path"})
            continue
        if sha(p) != item["sha256"]:
            failures.append({"path": item["path"], "issue": "hash mismatch"})
        if p.suffix == ".py":
            try:
                ast.parse(p.read_bytes(), filename=item["path"])
                syntax += 1
            except (ValueError, SyntaxError) as error:
                failures.append({"path": item["path"], "issue": str(error)})
        if item["path"] in active_docs:
            for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                destination = p.parent / target.split("#")[0]
                links += 1
                if not destination.exists():
                    failures.append({"path": item["path"], "issue": "broken active navigation", "target": target})
    return {"files_checked": len(paths), "python_files_parsed": syntax, "active_links_checked": links,
            "failures": failures, "all_passed": not failures,
            "scope": "Hash all registered source/data/evidence, parse every shipped Python file, check current navigation. Frozen historical report links and archived training runtime are not executed.",
            "training_performed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit()
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(ROOT):
            parser.error("Output must remain within repository")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))
    return int(not result["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())

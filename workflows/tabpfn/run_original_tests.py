"""Run unchanged source tests, explicitly skipping declared unavailable inputs."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parent


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    declarations = json.loads((ROOT / "test-resources.json").read_text(encoding="utf-8"))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    skipped = []
    for test in cases(suite):
        required = declarations.get(test.id(), [])
        missing = [p for p in required if not (ROOT / p).is_file()]
        if missing:
            reason = "Original source Git omits archived external inputs: " + ", ".join(missing)
            method = getattr(type(test), test._testMethodName)
            setattr(type(test), test._testMethodName, unittest.skip(reason)(method))
            skipped.append({"test": test.id(), "missing": missing})
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "passed": result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
              "errors": len(result.errors), "failures": len(result.failures), "skipped": len(result.skipped),
              "declared_missing_resource_skips": skipped, "successful": result.wasSuccessful(),
              "original_test_sources_modified": False}
    (ROOT / "original-test-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())

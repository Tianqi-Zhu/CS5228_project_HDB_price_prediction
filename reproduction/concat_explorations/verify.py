"""Verify archived bytes and Python syntax without importing historical runners."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((Path(__file__).parent / 'source_manifest.json').read_text())
failures = []
python_files = 0
for item in manifest['files']:
    path = ROOT / item['path']
    if not path.is_file():
        failures.append({'path': item['path'], 'reason': 'missing'})
        continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        failures.append({'path': item['path'], 'reason': 'sha256'})
    if path.suffix == '.py':
        python_files += 1
        try:
            ast.parse(path.read_bytes(), filename=item['path'])
        except (SyntaxError, ValueError) as error:
            failures.append({'path': item['path'], 'reason': str(error)})
print(json.dumps({'files_checked': len(manifest['files']),
                  'python_sources_parsed': python_files, 'failures': failures,
                  'training_performed': False}, indent=2))
raise SystemExit(bool(failures))

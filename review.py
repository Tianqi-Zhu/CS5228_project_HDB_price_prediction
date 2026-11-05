"""Read-only review navigation and integrity checks; no training or network."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'verify'], nargs='?', default='list')
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT/'review/manifest.json').read_text(encoding='utf-8'))
    if args.action == 'list':
        print('Local review branch:', manifest['branch'])
        for row in manifest['experiments']:
            print(row['id'], row['status'], row['purpose'])
        return
    failures=[]
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append({'path':row['path'],'reason':'missing or outside workspace'})
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            failures.append({'path':row['path'],'reason':'sha256 mismatch'})
    result={'branch':manifest['branch'],'files_checked':len(manifest['files']),
            'experiments':len(manifest['experiments']), 'failures':failures,
            'training_performed':False, 'pending':[r['id'] for r in manifest['experiments'] if r['status']=='pending']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__': main()

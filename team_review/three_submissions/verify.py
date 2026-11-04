"""Verify the three original submissions using only Python's standard library."""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent


def verify():
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    checked=[]
    for item in manifest['submissions']:
        path=HERE/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
        with path.open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream);assert reader.fieldnames==['Id','Predicted']
            count=0
            for count,row in enumerate(reader,1):
                assert int(row['Id'])==count-1
                assert math.isfinite(float(row['Predicted']))
            assert count==50000
        checked.append({'file':item['file'],'rows':count,'kaggle_score':item['kaggle_score'],'sha256':item['sha256']})
    for item in manifest['reconstruction_inputs']:
        assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256'],item['file']
    sources=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for item in sources:
        assert hashlib.sha256((HERE.parents[1]/item['path']).read_bytes()).hexdigest()==item['snapshot_sha256'],item['path']
    print(json.dumps({'verified':checked,'input_hashes_valid':True,'source_files_checked':len(sources)},ensure_ascii=False,indent=2))
    return checked


if __name__=='__main__':verify()

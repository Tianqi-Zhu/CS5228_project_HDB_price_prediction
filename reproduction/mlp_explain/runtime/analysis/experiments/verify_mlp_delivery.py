"""Verify the published MLP evidence and predictions without training a model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def verify(bundle, root):
    bundle=bundle.resolve();root=root.resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    for entry in manifest['files']:
        path=(bundle/entry['path']).resolve()
        path.relative_to(bundle)
        if not path.is_file() or digest(path)!=entry['sha256']:
            raise ValueError(f"Changed or missing evidence: {entry['path']}")
    delivery=json.loads((bundle/'delivery.json').read_text())
    for name,expected in delivery['raw_data_sha256'].items():
        if digest(root/name)!=expected and hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise ValueError(f'Original dataset changed: {name}')
    train=[(row['RENT_APPROVAL_DATE'],float(row['MONTHLY_RENT'])) for row in rows(root/'data/train.csv')]
    test_months=[row['RENT_APPROVAL_DATE'] for row in rows(root/'data/test.csv')]
    sample_ids=[int(row['Id']) for row in rows(root/'data/example-submission.csv')]
    expected_outer=[i for i,(month,_) in enumerate(train) if month>='2024-04']
    validation_results=[]
    for artifact in delivery['validation_artifacts']:
        data=list(rows(bundle/artifact['path']))
        ids=[int(row['source_row']) for row in data]
        if ids!=expected_outer or len(data)!=artifact['rows']:
            raise ValueError(f"Validation IDs differ: {artifact['path']}")
        total=0.
        for row,i in zip(data,ids):
            month,actual=train[i]
            if row['month']!=month or float(row['actual_sgd'])!=actual:
                raise ValueError('Validation month or label differs')
            prediction=float(row['prediction_sgd'])
            if not math.isfinite(prediction):
                raise ValueError('Nonfinite validation prediction')
            total+=(prediction-actual)**2
        score=math.sqrt(total/len(data))
        if abs(score-artifact['rmse_sgd'])>1e-4:
            raise ValueError(f"Validation score differs: {artifact['path']}")
        validation_results.append({'path':artifact['path'],'n':len(data),'rmse_sgd':score})
    test_results=[]
    for artifact in delivery['test_predictions']:
        data=list(rows(bundle/artifact['path']))
        if len(data)!=artifact['rows'] or len(data)!=len(test_months):
            raise ValueError('Test length differs')
        for row,expected_id,month in zip(data,sample_ids,test_months):
            if int(row['Id'])!=expected_id or row['month']!=month:
                raise ValueError('Test ID order or month differs')
            names=['prediction_sgd','ensemble_std_sgd','pi90_lower_sgd','pi90_upper_sgd','pi95_lower_sgd','pi95_upper_sgd']
            prediction,std,lo90,hi90,lo95,hi95=[float(row[name]) for name in names]
            if not all(math.isfinite(v) for v in [prediction,std,lo90,hi90,lo95,hi95]):
                raise ValueError('Nonfinite test output')
            if std<0 or not lo95<=lo90<=prediction<=hi90<=hi95:
                raise ValueError('Invalid prediction interval ordering')
            if 'actual_sgd' in row:
                raise ValueError('Test output unexpectedly contains labels')
        test_results.append({'path':artifact['path'],'n':len(data),'id_order_and_intervals_verified':True})
    return {'manifest_files_verified':len(manifest['files']),'validation':validation_results,'test':test_results,
            'test_rmse':'unavailable: test labels are not supplied'}


def main():
    root=Path(__file__).resolve().parents[5]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=root/'experiment/appendix/X06/mlp_explain')
    args=parser.parse_args()
    print(json.dumps(verify(args.bundle,root),indent=2))


if __name__=='__main__':main()

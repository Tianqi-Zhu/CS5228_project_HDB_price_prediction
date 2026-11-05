"""Reconstruct three submissions from saved model forecasts and learned weights.

This is an independent CPU replay, not model retraining. It does not import
the repository's training pipeline, read test targets, or access the network.
"""
from pathlib import Path
import csv
import json
import platform
import sys
import numpy as np
from verify import verify,HERE


def compose(matrix,names,support24,support36):
    assert matrix.shape==(50000,len(names)) and np.isfinite(matrix).all()
    weights=json.loads((HERE/'weights.json').read_text(encoding='utf-8'))
    history=np.zeros(len(matrix))
    for member in weights['history']['leaf_members']:
        history+=member['weight']*matrix[:,names.index(member['name'])]
    macro_weights=weights['macro']
    macro=macro_weights['previous_best']*history+macro_weights['current_cat8']*matrix[:,names.index('given_macro/current_cat8')]
    final_model=weights['structure']
    selected=np.ascontiguousarray(matrix[:,[names.index(name) for name in final_model['names']]])
    structure=selected@np.asarray(final_model['global_weights'])
    labels=np.array([str(bool(a))+'|'+str(bool(b)) for a,b in zip(support24,support36)])
    for group,learned in final_model['group_weights'].items():
        mask=labels==group
        structure[mask]=selected[mask]@np.asarray(learned['weights'])
    return {'kaggle_best_history.csv':history,'kaggle_best_macro.csv':macro,'kaggle_structure_blend.csv':structure}


def main():
    verify()
    archive=np.load(HERE/'components.npz',allow_pickle=False)
    generated=compose(archive['predictions'],archive['names'].tolist(),archive['support24'],archive['support36'])
    root=HERE.parents[1];out=root/'output/review_three_submissions_reproduced';out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,prediction in generated.items():
        with (HERE/'submissions'/name).open(newline='',encoding='utf-8') as stream:
            reference=np.array([float(row['Predicted']) for row in csv.DictReader(stream)])
        difference=float(np.max(np.abs(reference-prediction)))
        np.testing.assert_allclose(reference,prediction,atol=1e-8,rtol=0)
        path=out/name
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(['Id','Predicted'])
            writer.writerows((i,float(value)) for i,value in enumerate(prediction))
        results.append({'file':str(path.relative_to(root)).replace('\\','/'),'rows':len(prediction),'max_abs_difference':difference})
    report={'reproduced':results,'python':platform.python_version(),'numpy':np.__version__,
        'executable':sys.executable,'mode':'frozen-model-forecast replay; no training or network'}
    (out/'verification_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

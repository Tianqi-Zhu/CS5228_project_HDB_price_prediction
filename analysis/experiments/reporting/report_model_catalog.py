"""Audit all fetched branches without checking out or changing working files."""
from __future__ import annotations
import io
import hashlib
import json
import re
import subprocess
from pathlib import Path
import pandas as pd
from analysis.pipeline.config import ROOT

OUT=ROOT/'output/experiment_runs/00_report_and_final'


def git(*args):
    return subprocess.check_output(['git','-c','core.quotepath=false',*args],cwd=ROOT).decode('utf-8')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    refs=[line.split('|') for line in git('for-each-ref','refs/remotes/origin',
        '--format=%(refname:short)|%(objectname)|%(committerdate:iso8601)|%(subject)|%(symref)').splitlines()]
    branches=[]
    inventory=[]
    tables=[]
    unique={}
    for ref,commit,date,subject,symref in refs:
        if symref:
            continue
        branch=ref.removeprefix('origin/')
        files=[]
        for line in git('ls-tree','-r','-l',ref).splitlines():
            meta,path=line.split('\t',1)
            mode,kind,blob,size=meta.split()
            if kind!='blob':
                continue
            files.append(path)
            if not (path.endswith(('.py','.md','.json','.csv','.tex')) and
                (path.startswith(('reports/','analysis/','results/')) or path.endswith('.md'))):
                continue
            inventory.append({'branch':branch,'commit':commit,'path':path,'blob':blob,'bytes':int(size)})
            # Preserve small report evidence by object ID, deduplicated across
            # branches; do not copy predictions or source datasets.
            if not path.startswith(('reports/','results/')) or int(size)>200000:
                continue
            filename=Path(path).name
            excluded=('prediction','latent','feature','monthly','curve','epochs','bootstrap',
                'covariance','correlation','coefficient','errors','coverage','context','learning','manifest')
            if any(word in filename.lower() for word in excluded):
                continue
            if path.endswith(('.csv','.json','.md')) and re.search(r'metric|score|result|report|comparison|protocol|setting|selection|summary',filename):
                if blob not in unique:
                    content=git('show',f'{ref}:{path}')
                    unique[blob]={'content':content,'suffix':Path(path).suffix}
                content=unique[blob]['content']
                if path.endswith('.csv'):
                    try:
                        frame=pd.read_csv(io.StringIO(content))
                        rmse=[c for c in frame if 'rmse' in c.lower()]
                        if rmse:
                            tables.append({'branch':branch,'path':path,'blob':blob,'rows':len(frame),
                                'columns':list(frame),'records':frame.to_dict('records')})
                    except (ValueError,pd.errors.ParserError):
                        pass
        changed=git('diff','--name-only','origin/main',ref).splitlines()
        branches.append({'branch':branch,'commit':commit,'commit_date':date,'subject':subject,
            'file_count':len(files),'changed_paths_vs_main':len(changed),
            'report_directories':sorted({str(Path(p).parent).replace('\\','/') for p in files
                if p.startswith('reports/') and Path(p).name in ('report.md','results.json','comparison_report.md')})})
    snapshots=OUT/'branch_evidence'
    snapshots.mkdir(parents=True,exist_ok=True)
    for blob,item in unique.items():
        raw=item['content'].encode('utf-8')
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob
        (snapshots/(blob+item['suffix'])).write_bytes(raw)
    (OUT/'branches.json').write_text(json.dumps(branches,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'branch_metric_tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(inventory).to_csv(OUT/'branch_inventory.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(branches).drop(columns='report_directories').to_csv(OUT/'branches.csv',index=False,encoding='utf-8-sig')
    brief=[]
    for table in tables:
        frame=pd.DataFrame(table['records'])
        for c in frame:
            if 'rmse' not in c.lower() or any(s in c.lower() for s in ('delta','change','improv')):
                continue
            numeric=pd.to_numeric(frame[c],errors='coerce')
            if not numeric.notna().any():
                continue
            row=frame.loc[numeric.idxmin()].to_dict()
            brief.append({'branch':table['branch'],'path':table['path'],'metric':c,
                'minimum':float(numeric.min()),'row':json.dumps(row,ensure_ascii=False)})
    pd.DataFrame(brief).to_csv(OUT/'branch_score_index.csv',index=False,encoding='utf-8-sig')
    all_scores=[]
    for table in tables:
        for number,row in enumerate(table['records']):
            for metric,value in row.items():
                if 'rmse' not in metric.lower():
                    continue
                try:
                    numeric=float(value)
                except (TypeError,ValueError):
                    continue
                all_scores.append({'branch':table['branch'],'path':table['path'],'blob':table['blob'],
                    'csv_row_zero_based':number,'metric':metric,'value':numeric,
                    'dimensions_json':json.dumps({k:v for k,v in row.items() if k!=metric},ensure_ascii=False),
                    'warning':'Source index only; includes subgroup, pooled, calibration and non-rental metrics. Do not rank without protocol review.'})
    pd.DataFrame(all_scores).to_csv(OUT/'all_branch_scores.csv',index=False,encoding='utf-8-sig')
    local=[]
    for path in sorted((ROOT/'output/experiment_runs').rglob('results.json')):
        if OUT in path.parents:
            continue
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):
            continue
        metrics=record.get('scores',{})
        if not isinstance(metrics,dict):
            metrics={}
        value=record.get('rmse_sgd',metrics.get('rmse_sgd'))
        if value is None:
            continue
        local.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),
            'stage':record.get('stage','see source protocol'),'rmse_sgd':value,
            'features':record.get('features',record.get('n_features')),
            'spec_json':json.dumps(record.get('spec',record.get('selection',{})),ensure_ascii=False),
            'warning':'Includes screening and different temporal protocols; not a common leaderboard.'})
    pd.DataFrame(local).to_csv(OUT/'local_run_index.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({'branches':len(branches),'indexed_files':len(inventory),
        'unique_evidence_blobs':len(unique),'metric_tables':len(tables)},indent=2))


if __name__=='__main__':
    run()

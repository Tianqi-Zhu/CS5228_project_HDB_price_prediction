"""Local validation orchestration; supplied dependency paths are read-only.

Creates fresh logs under output/. Does not install packages or fit research models.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dependency-cache',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('Use a new output/ validation directory')
    out.mkdir(parents=True)
    cache=args.dependency_cache.resolve() if args.dependency_cache else None
    checks=[('manifest',['review.py','verify'],'core'),
            ('catalog',['-m','analysis.appendix_catalog','audit'],'core'),
            ('inputs',['-m','analysis.appendix_review','verify'],'core'),
            ('statistics-tests',['-m','unittest','discover','-s','tests','-p','test_appendix_statistics.py','-v'],'core'),
            ('core-tests',['-m','unittest','discover','-s','tests','-v'],'core'),
            ('three-submissions',['reproduction/three_submissions/reproduce.py'],'core'),
            ('concat-archive',['reproduction/concat_explorations/verify.py'],'core'),
            ('mlp-archive',['reproduction/mlp_explain/run.py','verify'],'core'),
            ('mlp-tests',['reproduction/mlp_explain/run.py','test'],'core'),
            ('mlp-diagnose',['reproduction/mlp_explain/run.py','diagnose','--output-dir',str(out/'mlp-diagnostics')],'core'),
            ('tabpfn-manifest',['review_tabpfn.py','verify'],'sdk'),
            ('tabpfn-tests',['review_tabpfn.py','test','--workspace',str(out/'tabpfn-tests')],'sdk')]
    results=[]
    for name,command,kind in checks:
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
        if cache:
            paths=([cache/'tabpfn-split-deps'] if kind=='sdk' else [])+[cache/'deep-deps',cache/'model-deps']
            env['PYTHONPATH']=os.pathsep.join(map(str,paths))
        start=time.monotonic()
        with (out/(name+'.log')).open('wb') as stream:
            p=subprocess.run([sys.executable,'-B',*command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        item={'check':name,'exit_code':p.returncode,'seconds':round(time.monotonic()-start,2),'log':(out/(name+'.log')).relative_to(ROOT).as_posix()}
        results.append(item)
        (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(item),flush=True)
    return int(any(r['exit_code'] for r in results))

if __name__=='__main__':raise SystemExit(main())

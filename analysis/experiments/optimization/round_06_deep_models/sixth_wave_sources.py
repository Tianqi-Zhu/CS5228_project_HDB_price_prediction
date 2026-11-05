"""Download small MIT-licensed author files at an immutable Git revision."""
import hashlib
import json
import urllib.request
from analysis.experiments.frontier_search import ROOT,write_json

DEST=ROOT/'.cache/deep-sources'
OUT=ROOT/'output/experiment_runs/round_06_deep_models'
COMMIT='39b54024902955e51063a34bce17d514cd68ff42'


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'CS5228-model-experiment'})
    with urllib.request.urlopen(request,timeout=60) as response:
        return response.read()


def main():
    repo='LAMDA-Tabular/TALENT'
    commit=COMMIT
    paths=['TALENT/model/models/modernNCA.py','TALENT/model/lib/tabr/utils.py','LICENSE',
        'TALENT/configs/default/modernNCA.json']
    files=[]
    for path in paths:
        url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
        raw=fetch(url)
        destination=DEST/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw)
        files.append({'url':url,'path':str(destination.relative_to(ROOT)),
            'sha256':hashlib.sha256(raw).hexdigest()})
    write_json(OUT/'author_source.json',{'repository':repo,'commit':commit,'files':files,
        'note':'Unmodified author ModernNCA implementation and its numerical-embedding utility; own training adapter and context budgeting are documented separately.'})
    print(json.dumps({'commit':commit,'files':len(files)}),flush=True)


if __name__=='__main__':
    main()

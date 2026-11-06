"""Download public author Transformer checkpoints at recorded immutable revisions."""
from __future__ import annotations
import hashlib
import json
import time
from urllib.request import Request,urlopen
from analysis.experiments.frontier_search import ROOT,write_json

OUT=ROOT/'output/experiment_runs/round_07_pretrained_transformers'
DEST=ROOT/'.cache/transformer-models'
MODELS={'tabdpt13':('Layer6/TabDPT','tabdpt1_3.safetensors'),
    'nori15':('Synthefy/Nori-1-5','nori.pt'),
    'nori6m':('Synthefy/Nori','nori.pt')}


def request(url):
    return urlopen(Request(url,headers={'User-Agent':'CS5228-local-transformer-research'}),timeout=60)


def main():
    manifest_path=OUT/'checkpoint_manifest.json'
    records=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name,(repo,filename) in MODELS.items():
        destination=DEST/name/filename
        if name in records and destination.exists():
            with destination.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest==records[name]['sha256']:
                print('CHECKPOINT VERIFIED',name,flush=True);continue
        with request(f'https://huggingface.co/api/models/{repo}?blobs=true') as response:
            info=json.load(response)
        if info.get('gated'):
            raise RuntimeError(f'{repo} requires user access; no gated download attempted')
        revision=records[name]['revision'] if name in records else info['sha']
        file_info=next(item for item in info['siblings'] if item['rfilename']==filename)
        expected=file_info.get('lfs',{}).get('sha256') if revision==info['sha'] else records[name]['sha256']
        url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
        destination.parent.mkdir(parents=True,exist_ok=True)
        partial=destination.with_suffix(destination.suffix+'.partial')
        received=0;digest=hashlib.sha256();last=time.monotonic()
        print('CHECKPOINT DOWNLOAD',name,revision,flush=True)
        with request(url) as response,partial.open('wb') as stream:
            while chunk:=response.read(1024**2):
                stream.write(chunk);digest.update(chunk);received+=len(chunk)
                if time.monotonic()-last>10:
                    print(f'  {name} downloaded {received/1024**2:.1f} MiB',flush=True);last=time.monotonic()
        checksum=digest.hexdigest()
        if expected and checksum!=expected:
            raise ValueError(f'Author LFS digest mismatch: {name}')
        partial.replace(destination)
        records[name]={'repository':repo,'filename':filename,'revision':revision,'url':url,
            'path':str(destination.relative_to(ROOT)),'sha256':checksum,'bytes':received,
            'author_lfs_sha256':expected,'gated':False}
        write_json(manifest_path,records)
        print('CHECKPOINT SAVED',name,received,flush=True)


if __name__=='__main__':
    main()

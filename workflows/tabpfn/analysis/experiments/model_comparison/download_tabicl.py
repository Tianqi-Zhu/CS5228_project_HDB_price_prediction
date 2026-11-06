"""Download the official public TabICLv2 regression checkpoint into this project."""
from __future__ import annotations
import hashlib
import json
import time
from urllib.request import Request, urlopen
from analysis.pipeline.config import ROOT

FILENAME = 'tabicl-regressor-v2-20260212.ckpt'
DIRECTORY = ROOT / '.cache/foundation-models'
CHECKPOINT = DIRECTORY / FILENAME
URL = 'https://huggingface.co/jingang/TabICL/resolve/main/' + FILENAME


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path = DIRECTORY / 'tabicl_manifest.json'
    if CHECKPOINT.exists() and manifest_path.exists():
        expected = json.loads(manifest_path.read_text())['sha256']
        if hashlib.file_digest(CHECKPOINT.open('rb'), 'sha256').hexdigest() == expected:
            print('Verified existing TabICLv2 checkpoint', flush=True)
            return
    temporary = CHECKPOINT.with_suffix('.partial')
    started, last_update = time.monotonic(), time.monotonic()
    request = Request(URL, headers={'User-Agent': 'CS5228-local-research/1.0'})
    with urlopen(request, timeout=45) as response, temporary.open('wb') as output:
        total = int(response.headers.get('Content-Length') or 0)
        digest, received = hashlib.sha256(), 0
        print('Downloading official checkpoint; bytes:', total or 'unknown', flush=True)
        while chunk := response.read(1024*1024):
            output.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if time.monotonic()-last_update > 5:
                print(f'Downloaded {received/1024**2:.1f} MiB', flush=True)
                last_update = time.monotonic()
        if total and received != total:
            raise ValueError('Incomplete checkpoint response')
    temporary.replace(CHECKPOINT)
    result = {'repository': 'jingang/TabICL', 'filename': FILENAME, 'url': URL,
        'sha256': digest.hexdigest(), 'bytes': received, 'seconds': time.monotonic()-started,
        'official_implementation': 'https://github.com/soda-inria/tabicl'}
    manifest_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

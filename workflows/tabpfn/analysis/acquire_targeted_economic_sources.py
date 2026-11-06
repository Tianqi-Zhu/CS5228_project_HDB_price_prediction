"""Acquire pinned original HDB documents and preserve external-data audit evidence."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from analysis.config import ROOT,REPORTS
from analysis.data import sha256
from analysis.run_targeted_housing_supply import DOCUMENTS,SOURCES


def main():
    SOURCES.mkdir(parents=True,exist_ok=True)
    manifest_path=SOURCES/'source_download_manifest.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    expected={r['file']:r for r in old['documents']} if old else {}
    records=[]
    for spec in DOCUMENTS:
        path=SOURCES/spec['file']
        if not path.exists():
            session=requests.Session();session.trust_env=False
            start=time.monotonic();parts=[]
            with session.get(spec['url'],timeout=(10,20),stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(65536):
                    parts.append(chunk)
                    if time.monotonic()-start>180:raise TimeoutError('Document acquisition exceeded180s')
            content=b''.join(parts)
            if not content.startswith(b'%PDF'):raise ValueError('Expected original PDF')
            temporary=path.with_suffix('.download');temporary.write_bytes(content)
            if spec['file'] in expected and sha256(temporary)!=expected[spec['file']]['sha256']:
                raise ValueError(f'Official file changed; retain download for inspection: {temporary}')
            temporary.rename(path)
        digest=sha256(path)
        if spec['file'] in expected:assert digest==expected[spec['file']]['sha256']
        records.append(dict(file=spec['file'],url=spec['url'],sha256=digest,bytes=path.stat().st_size,
                            used_for='B1 archived supply',available=spec['available'],timing=spec['timing']))
        print('VERIFIED',spec['file'],digest[:12],flush=True)
    if not old:
        manifest_path.write_text(json.dumps(dict(registered_utc=datetime.now(timezone.utc).isoformat(),documents=records,
            note='Original annual-report snapshots. Cached files are never overwritten. First registration follows independent acquisition and visual/table-total checks; exact later replay requires these hashes.'),indent=2),encoding='utf-8')
    # These sources document exclusion, not admitted historical feature values.
    audits=[('mom_current_page.html','https://www.mom.gov.sg/foreign-workforce-numbers'),
            ('rent_metadata.json','https://api-production.data.gov.sg/v2/public/api/datasets/d_23000a00c52996c55106084ed0339566/metadata')]
    for filename,url in audits:
        path=SOURCES/filename
        if not path.exists():
            session=requests.Session();session.trust_env=False;r=session.get(url,timeout=(10,20));r.raise_for_status();path.write_bytes(r.content)
    print('Official current-snapshot evidence retained; current snapshots are not used as historical features',flush=True)


if __name__=='__main__':main()

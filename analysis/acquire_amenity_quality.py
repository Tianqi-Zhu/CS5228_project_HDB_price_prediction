"""Cache small public source documents for amenity-quality measurements."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from analysis.config import ROOT
from analysis.data import sha256

NAME = "tabpfn_amenity_quality_80_20"
CACHE = ROOT / ".cache" / NAME
OUT = ROOT / "reports" / NAME


class Sources:
    def __init__(self):
        (CACHE / "sources").mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "source_manifest.json"
        self.manifest = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers["User-Agent"] = "HDB-CS5228-research/1.0 (public source caching)"

    def get(self, name, url, *, policy, max_mib=25):
        path = CACHE / "sources" / name
        if name in self.manifest and path.exists() and self.manifest[name].get("sha256") == sha256(path):
            return path
        entry = {"url": url, "policy": policy, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            r = self.session.get(url, timeout=45, stream=True)
            r.raise_for_status()
            content = bytearray()
            for chunk in r.iter_content(1024 * 1024):
                content.extend(chunk)
                if len(content) > max_mib * 1024**2:
                    raise ValueError("Source exceeds the declared download cap")
            path.write_bytes(content)
            entry.update(status="acquired", bytes=len(content), sha256=sha256(path), path=str(path.relative_to(ROOT)),
                         content_type=r.headers.get("Content-Type"))
            print(f"Acquired {name}: {len(content):,} bytes", flush=True)
        except Exception as exc:
            entry.update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
            print(f"Blocked {name}: {entry['reason']}", flush=True)
        self.manifest[name] = entry
        self.path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return path if entry["status"] == "acquired" else None


def acquire(stage):
    source = Sources()
    source.get('jurong_point_2022.html','https://www.linkreit.com/en/media/news-releases/link-enters-singapore-through-the-acquisition-of-jurong-point-and-swing-by-thomson-plaza/',policy='Link REIT acquisition release dated December 28 2022; Jurong Point NLA approximately 720,000 sqft. Partial Thomson Plaza interest is excluded.')
    if stage in ("schools", "all"):
        for year in (2020, 2021, 2022, 2023):
            for region in ("all",):
                source.get(f"schools_{year}_{region}.html", f"https://sgschooling.com/year/{year}/{region}",
                           policy="MOE-derived historical P1 vacancy/application facts, preserved by SG Schooling; published tables are a secondary archive, not official school rankings. Original registration-year availability must be audited.")
    if stage in ("malls", "all"):
        for year in (2019, 2020, 2021, 2022, 2023):
            source.get(f"fct_{year}.pdf", f"https://fct.frasersproperty.com/misc/FCT-AR{year}.pdf",
                       policy="Owner annual report; fiscal year ends September. Measurements enter from a conservative next-year January origin, after publication verification.")
        for year in (2020, 2021, 2022, 2023):
            source.get(f"cict_{year}_index.html", f"https://investor.cict.com.sg/misc/ar{year}/",
                       policy="Owner annual report digital edition; used to discover the actual source PDF and publication date.")
    if stage in ("bus", "all"):
        for year in (2020, 2021, 2022, 2023):
            path = source.get(f"bus_commit_{year}.json", f"https://api.github.com/repos/cheeaun/sgbusdata/commits?path=data/v1/stops.json&until={year}-12-31T23:59:59Z&per_page=1",
                              policy="Public git snapshot of LTA-derived stops/services; commit time is the earliest verifiable archive availability, not an assumed current network.")
            if not path:
                continue
            rows = json.loads(path.read_text())
            if not rows:
                continue
            commit = rows[0]["sha"]
            for file in ("stops.json", "services.json"):
                source.get(f"bus_{year}_{file}", f"https://raw.githubusercontent.com/cheeaun/sgbusdata/{commit}/data/v1/{file}",
                           policy=f"LTA-derived public snapshot; git {commit}; do not confuse unique bus services with bus frequency or passengers.")
    return source


def followups():
    from bs4 import BeautifulSoup
    source = Sources()
    for year in (2020, 2021, 2022, 2023):
        source.get(f"schools_{year}_all.html", f"https://sgschooling.com/year/{year}/all", policy="Historical MOE-derived school registration facts; SG Schooling archive, all planning areas; originals published during the registration year.")
    page = source.get("cict_reports.html", "https://investor.cict.com.sg/ar.html", policy="Owner annual report download index; actual PDF links, not guessed filenames.")
    if page:
        soup = BeautifulSoup(page.read_bytes(), "lxml")
        links = [(a.get("href"), a.get_text(" ", strip=True)) for a in soup.select("a[href]") if ".pdf" in a.get("href", "").lower()]
        print("CICT PDF candidates", links, flush=True)
        from urllib.parse import urljoin
        for url, title in links:
            for year in (2019, 2020, 2021, 2022, 2023):
                if str(year) in url and ("ar20" in url.lower() or "annual" in url.lower()):
                    source.get(f"cict_{year}.pdf", urljoin("https://investor.cict.com.sg",url), policy="Owner calendar-year annual report; use no earlier than verified publication or conservative May of the following year.")
    page = source.get("fct_reports.html", "https://fct.frasersproperty.com/ar.html", policy="Owner financial report index")
    if page:
        soup = BeautifulSoup(page.read_bytes(), "lxml")
        print("FCT PDF candidates", [(a.get('href'),a.get_text(' ',strip=True)) for a in soup.select('a[href]') if '.pdf' in a.get('href','').lower()][:40], flush=True)
        from urllib.parse import urlparse, parse_qs, unquote
        for a in soup.select('a[href]'):
            for year in (2019,2020,2021,2022,2023):
                if f"FY{year} Annual Report" in a.get_text(' ',strip=True):
                    url=a['href']; url=parse_qs(urlparse(url).query).get('redirect',[url])[0]
                    source.get(f"fct_{year}.pdf",url,policy="Owner fiscal-year report; conservative January following year for availability, except publication verified later.")
    source.get("cict_2019.pdf", "https://investor.cict.com.sg/newsroom/20200409_063017_C38U_131S366GOBVE1WAR.2.pdf", policy="Owner CMT Annual Report 2019, published April 9 2020; available May 2020 conservatively")
    source.get("bus_commit_2020_legacy.json", "https://api.github.com/repos/cheeaun/sgbusdata/commits?path=data/stops.json&until=2020-12-31T23:59:59Z&per_page=1", policy="Resolve the historical pre-v1 directory")
    source.get("train_kaggle_metadata.json", "https://www.kaggle.com/api/v1/datasets/list/yorkyong/singapore-passenger-volume-by-train-stations", policy="Public archive of LTA station passenger counts; inspect observation coverage before download")
    source.get("train_2023_archive.zip", "https://www.kaggle.com/api/v1/datasets/download/yorkyong/singapore-passenger-volume-by-train-stations", policy="LTA passenger volume archive covering August–October 2023, uploaded December 6 2023; available January 2024 conservatively. No imputation into earlier dates.",max_mib=5)
    source.get("bus_first_commit.json", "https://api.github.com/repos/cheeaun/sgbusdata/commits?path=data/v1/stops.json&until=2021-06-30T23:59:59Z&per_page=1", policy="Earlier public bus snapshot to improve chronological source coverage")
    p=CACHE/'sources/bus_first_commit.json'
    if p.exists() and (rows:=json.loads(p.read_text())):
        for file in ('stops.json','services.json'):
            source.get(f'bus_first_{file}',f"https://raw.githubusercontent.com/cheeaun/sgbusdata/{rows[0]['sha']}/data/v1/{file}",policy="LTA-derived public bus archive; effective from the documented git commit month end")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["schools", "malls", "bus", "all", "followups"], default="all")
    args = parser.parse_args()
    followups() if args.stage == "followups" else acquire(args.stage)

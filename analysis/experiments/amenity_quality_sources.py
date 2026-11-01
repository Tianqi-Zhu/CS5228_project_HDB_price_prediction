"""Extract dated factual amenity measurements; no rental labels are accepted."""
from __future__ import annotations
import json
from pathlib import Path
import re
import unicodedata
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


def school_key(name):
    name=unicodedata.normalize('NFKD',str(name)).lower().replace('saint','st')
    name=re.sub(r'\b(primary|school)\b','',name)
    return re.sub('[^a-z0-9]','',name)


def parse_school_page(path,year):
    soup=BeautifulSoup(Path(path).read_bytes(),'lxml')
    table=soup.select('table')[0]
    rows=table.select('tr'); headers=[x.get_text(' ',strip=True) for x in rows[0].find_all(['th','td'])][1:]
    c=headers.index('2C'); result=[]; name=None; vacancy=None
    for tr in rows[1:]:
        cells=tr.find_all(['th','td']); text=[x.get_text(' ',strip=True) for x in cells]
        if not text: continue
        if cells[0].find('a'):
            name=cells[0].find('a').get_text(' ',strip=True); vacancy=None
        elif name and 'Vacancy' in text[0]:
            value=text[c+1]; vacancy=int(value.replace(',','')) if re.fullmatch(r'[\d,]+',value) else None
        elif name and 'Applied' in text[0]:
            value=text[c+1]; applied=int(value.replace(',','')) if re.fullmatch(r'[\d,]+',value) else None
            if vacancy is not None and applied is not None:
                result.append(dict(school_name=name,school_key=school_key(name),year=year,vacancies=vacancy,applications=applied,
                                   log_demand=float(np.log((applied+1)/(vacancy+1))),available_at=f'{year}-10-01',
                                   source_file=Path(path).name,source_type='MOE-derived historical archive; not official quality ranking'))
    frame=pd.DataFrame(result)
    if len(frame)<170 or frame.school_key.duplicated().any(): raise ValueError(f'Incomplete/ambiguous school table {path}')
    return frame


CICT_NAMES={'Tampines Mall':r'TAMPINES\s+MALL','Junction 8':r'JUNCTION\s+8','Plaza Singapura':r'PLAZA\s+SINGAPURA',
 'Bugis Junction':r'BUGIS\s+JUNCTION','Lot 1':r'LOT\s+ONE\s+SHOPPERS[^a-zA-Z0-9]{0,4}\s*MALL',
 'Bukit Panjang Plaza':r'BUKIT\s+PANJANG\s+PLAZA','Bugis+':r'BUGIS\s*\+','Bedok Mall':r'BEDOK\s+MALL','Westgate':r'WESTGATE'}
FCT_NAMES=['Causeway Point','Waterway Point','Tampines 1','Tiong Bahru Plaza','Changi City Point','Century Square','Hougang Mall','White Sands','NEX','Yew Tee Point']


def pdf_pages(path,cache):
    dest=Path(cache)/(Path(path).stem+'.json')
    if dest.exists(): return json.loads(dest.read_text(encoding='utf-8'))
    pages=[p.extract_text() for p in PdfReader(path).pages]
    dest.parent.mkdir(exist_ok=True,parents=True); dest.write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
    return pages


def parse_malls(source_dir,text_dir):
    rows=[]
    # Exact retail values from CMT AR2019 printed pp134-135. Mixed-use/partial
    # interests and unknown revenue are excluded rather than mislabelled.
    values=[('Tampines Mall',356228,82.9,25.8),('Junction 8',254209,61.2,33.6),('Plaza Singapura',484154,92.1,25.7),
            ('Bugis Junction',396604,84.9,39.9),('Lot 1',227671,43.2,16.8),('Bukit Panjang Plaza',163625,np.nan,12.2),
            ('Bugis+',214408,33.7,20.5),('Bedok Mall',222469,57.9,20.2),('Westgate',410535,74.9,51.1)]
    if (Path(source_dir)/'cict_2019.pdf').exists():
        for name,nla,rev,traffic in values:
            rows.append(dict(mall=name,year=2019,nla_sqm=nla*.09290304,revenue_sgd_m=rev,footfall_m=traffic,
                             available_at='2020-05-01',source_file='cict_2019.pdf',pdf_page=136,scope='whole standalone retail property',extraction='curated published portfolio row'))
        for name,nla,traffic in [('Funan',317387,7.9),('IMM',424161,16.1),('Raffles City',427400,30.1)]:
            rows.append(dict(mall=name,year=2019,nla_sqm=nla*.09290304,revenue_sgd_m=np.nan,footfall_m=traffic,
                             available_at='2020-05-01',source_file='cict_2019.pdf',pdf_page=136,scope='retail component only; office/warehouse/stake revenue excluded',extraction='curated labelled retail NLA; no mixed-use revenue'))
    if (Path(source_dir)/'jurong_point_2022.html').exists():
        rows.append(dict(mall='Jurong Point',year=2022,nla_sqm=720000*.09290304,revenue_sgd_m=np.nan,footfall_m=np.nan,
                         available_at='2023-01-01',source_file='jurong_point_2022.html',pdf_page=np.nan,scope='whole mall approximate disclosed NLA',extraction='Link REIT 28 December 2022 acquisition release; approximate NLA as published'))
    number=r'([\d,]+(?:\.\d+)?)'
    for year in (2020,2021,2022,2023):
        path=Path(source_dir)/f'cict_{year}.pdf'
        if not path.exists(): continue
        pages=pdf_pages(path,text_dir)
        for name,title in CICT_NAMES.items():
            matches=[(i,t) for i,t in enumerate(pages) if re.search(r'(?:^|\n)'+title+r'\d*(?:\s*\n|\s*$)',t,re.I) and 'Net Lettable Area' in t and 'Gross Revenue' in t and re.search(r'As at 31 December\s+'+str(year-1)+r'\s+'+str(year),t,re.I)]
            if len(matches)!=1: print('CICT profile matches',year,name,len(matches),flush=True); continue
            i,t=matches[0]
            def current(label):
                m=re.search(label+r'\s*'+number+r'\s+'+number,t,re.I)
                return float(m.group(2).replace(',','')) if m else np.nan
            nla=current(r'Net Lettable Area\s*\d*\s*\(sq\s*ft\)')
            rev=current(r'Gross Revenue\s*\d*\s*\(S\$\s*million\)')
            traffic=current(r'Annual Shopper Traffic\s*\(million\)')
            if re.search(r'Other Assets\s*\(Bugis\+ and Bukit Panjang Plaza\)',t,re.I): rev=np.nan
            if not np.isfinite(nla) or not 50000<nla<1000000: raise ValueError(f'CICT NLA parse failed: {year} {name} {nla}')
            rows.append(dict(mall=name,year=year,nla_sqm=nla*.09290304,revenue_sgd_m=rev,footfall_m=traffic,
                             available_at=f'{year+1}-05-01',source_file=path.name,pdf_page=i+1,scope='whole standalone retail property',extraction='standalone profile current-year column'))
    for year in (2019,2020,2021,2022,2023):
        path=Path(source_dir)/f'fct_{year}.pdf'
        if not path.exists(): continue
        pages=pdf_pages(path,text_dir)
        for name in FCT_NAMES:
            # Require the property heading and the labelled NLA field; skip
            # portfolio summaries, office properties, wings, and comparisons.
            matches=[(i,t) for i,t in enumerate(pages) if re.search(r'(?:^|\n)'+re.escape(name)+r'\s*(?:\n|$)',t,re.I) and re.search(r'(?m)^Net Lettable Area',t,re.I)]
            if not matches: continue
            candidates=[]
            for i,t in matches:
                if sum(bool(re.search(r'(?:^|\n)'+re.escape(n)+r'\s*(?:\n|$)',t,re.I)) for n in FCT_NAMES)>1: continue
                m=re.search(r'(?m)^Net Lettable Area[^\n]{0,20}\n\s*'+number+r'\s*(?:square\s+met(?:er|re)s|sqm|sq\s*m\b)',t,re.I)
                if not m: continue
                nla=float(m.group(1).replace(',',''))
                traffic=re.search(r'Annual Shopper Traffic\s*:?\s*\n?\s*'+number+r'\s*million',t,re.I)
                revenue=re.search(r'Gross Revenue\s+'+number,t,re.I)
                candidates.append((i,nla,float(revenue.group(1).replace(',','')) if revenue else np.nan,
                                   float(traffic.group(1).replace(',','')) if traffic else np.nan))
            if not candidates: print('FCT NLA parse unavailable',year,name,[i for i,_ in matches],flush=True); continue
            # Profile appears once; repeated same-value pages are harmless.
            i,nla,rev,traffic=candidates[0]
            if not 1000<nla<100000: raise ValueError(f'FCT NLA invalid: {year} {name} {nla}')
            # Current-year group revenue table has absolute property revenue in
            # thousands; joint ventures without reported totals remain missing.
            for t in pages:
                if 'Gross Revenue' not in t or 'Property Expenses' not in t: continue
                section=t.split('Gross Revenue',1)[1].split('Property Expenses',1)[0]
                m=re.search(r'(?m)^'+re.escape(name)+r'\d*\s+'+number+r'\s+'+number,section)
                if m and float(m.group(1).replace(',',''))>1000:
                    rev=float(m.group(1).replace(',',''))/1000; break
            if name in ('Waterway Point','NEX'): rev=np.nan  # JV stake revenue is not whole-mall revenue.
            rows.append(dict(mall=name,year=year,nla_sqm=nla,revenue_sgd_m=rev,footfall_m=traffic,
                             available_at=f'{year+1}-01-01',source_file=path.name,pdf_page=i+1,scope='whole standalone retail property',extraction='labelled profile and owner gross-revenue table; not tenant sales'))
    result=pd.DataFrame(rows)
    if result.duplicated(['mall','year']).any(): raise ValueError('Duplicate mall year')
    return result.sort_values(['mall','year'])


def train_passengers(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        records=pd.concat([pd.read_csv(z.open(n)) for n in z.namelist() if n.endswith('.csv')],ignore_index=True)
    monthly=records.groupby(['YEAR_MONTH','PT_CODE'])[['TOTAL_TAP_IN_VOLUME','TOTAL_TAP_OUT_VOLUME']].sum().sum(axis=1).rename('taps').reset_index()
    monthly['days']=pd.to_datetime(monthly.YEAR_MONTH+'-01').dt.days_in_month
    monthly['daily_taps']=monthly.taps/monthly.days
    return monthly.groupby('PT_CODE').daily_taps.mean().reset_index().assign(available_at='2024-01-01',observation_period='2023-08 to 2023-10',source='LTA via yorkyong public archive, uploaded 2023-12-06')

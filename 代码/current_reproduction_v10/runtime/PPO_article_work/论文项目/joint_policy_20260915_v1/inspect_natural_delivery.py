"""Read-only compact inspection for the manuscript delivery."""
import json, re, hashlib
from pathlib import Path
from study_io import BASE

case = BASE / 'paper_evidence_review/natural_closed_case'
for name in ('joint', 'fixed', 's2'):
    m = json.loads((case / f'{name}_paper_metrics.json').read_text(encoding='utf-8'))
    print(name, 'keys', list(m))
    for k, v in m.items():
        if k not in ('corners', 'cruise'):
            print(k, str(v)[:6500])
binding=json.loads((case/'paper_result_binding.json').read_text(encoding='utf-8'))
for name, row in binding['primary_methods'].items():
    for key in ('trace','result','metrics'):
        p=BASE/row[key]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row[key+'_sha256'],p
print('BINDING_HASHES_PASS')
meta=json.loads((BASE/'manuscript_reference_metadata_v1/results.json').read_text(encoding='utf-8'))
for r in meta['results']:
    if r['key'] in ('Sun2022','Lee2011','Erkorkmaz2001','Sencer2015','PhamPham2018TOPPRA','Pham2018','Li2020','Wabersich2021','Schulman2017PPO'):
        print('REF',json.dumps(r,ensure_ascii=False))
print('BIB',re.findall(r'@\w+\{[^@]+', (BASE/'manuscript_joint_v1/references.bib').read_text(encoding='utf-8'))[-4:])

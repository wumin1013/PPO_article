"""Prepare trace-derived supplements and verify only the references actually cited."""
import csv
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote
import requests
from study_io import BASE, ROOT, sha, write_json

CASE=BASE/'paper_evidence_review/natural_closed_case'
PAPER=BASE/'manuscript_joint_v1'
OUT=PAPER/'natural_case_supplements'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def csv_write(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def bib_field(block, key):
    m=re.search(r'\b'+key+r'\s*=\s*\{(.*)\}\s*,?\s*$', block, re.M)
    return m.group(1) if m else None


def main():
    OUT.mkdir(exist_ok=True)
    for p in PAPER.glob('natural_*.tex'):
        p.write_text(p.read_text(encoding='utf-8').replace('\\S2','\\STwo'),encoding='utf-8')
    binding=read(CASE/'paper_result_binding.json')
    names=('Joint','Fixed','S2')
    metrics={name:read(BASE/binding['primary_methods'][name]['metrics']) for name in names}
    reporting=read(CASE/'geometry_reporting.json')
    section_rows=[]
    for name,m in metrics.items():
        for label in ('Long straights','Short straights','Circular arcs'):
            spans=[s for s in reporting['spans'] if
                ('Circular arcs' if s['kind']=='arc' else 'Long straights' if s['length_mm']>700 else 'Short straights')==label]
            indexes={i for s in spans for i in s['original_primitive_indices']}
            bins=[x for x in m['sections'] if x['index'] in indexes]
            section_rows.append(dict(method=name,span_class=label,span_count=len(spans),
                reference_length_mm=sum(s['length_mm'] for s in spans),
                assigned_time_s=sum(x['time_s'] for x in bins),
                highspeed_time_s=sum(x['cruise99_s'] for x in bins),
                assigned_distance_mm=sum(x['travel_distance_mm'] for x in bins),
                highspeed_distance_mm=sum(x['cruise99_distance_mm'] for x in bins)))
    assert sum(r['span_count'] for r in section_rows if r['method']=='Joint')==38
    csv_write('span_class_statistics.csv', section_rows)
    corners=[dict(method=name,**r) for name,m in metrics.items() for r in m['corners']]
    csv_write('all_34_corners_three_methods.csv',corners)
    thresholds=[dict(method=name,threshold_mm_s=th,**m['cruise'][str(th)]) for name,m in metrics.items() for th in (95,98,99)]
    csv_write('cruise_thresholds.csv',thresholds)
    csv_write('all_source_sections.csv',[dict(method=name,**r) for name,m in metrics.items() for r in m['sections']])
    tex=[]
    for label in ('Long straights','Short straights','Circular arcs'):
        items=[next(x for x in section_rows if x['method']==name and x['span_class']==label) for name in names]
        tex.append(label+' & '+' & '.join(f"{x['highspeed_time_s']:.3f} / {x['assigned_time_s']:.3f}" for x in items)+r' \\')
    tx='\n'.join(tex)
    th='\n'.join(str(th)+r' mm/s & '+' & '.join(f"{100*metrics[name]['cruise'][str(th)]['distance_fraction']:.3f}" for name in names)+r' \\' for th in (95,98,99))
    (PAPER/'natural_supplement_macros.tex').write_text('\\newcommand{\\SectionRows}{'+tx+'}\n\\newcommand{\\ThresholdRows}{'+th+'}\n',encoding='utf-8')

    # Resolve the S2 source identity in the new manuscript only; retain the old library.
    for file in ('natural_intro.tex','natural_method.tex'):
        p=PAPER/file
        t=p.read_text(encoding='utf-8').replace('Sencer2015','Sencer2014').replace('public development case','disclosed development case').replace('all34','all 34')
        p.write_text(t,encoding='utf-8')
    p=PAPER/'natural_findings.tex'
    p.write_text(p.read_text(encoding='utf-8').replace('below95','below 95'),encoding='utf-8')
    all_tex='\n'.join(p.read_text(encoding='utf-8') for p in PAPER.glob('natural_*.tex'))
    cited=sorted(set(key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',all_tex) for key in group.split(',')))
    blocks={re.match(r'@\w+\{([^,]+)',x).group(1):x.strip() for x in re.split(r'\n(?=@)',(PAPER/'references.bib').read_text(encoding='utf-8')) if x.startswith('@')}
    assert set(cited)<=set(blocks)
    # unsrtnat does not display eprint; retain the verified arXiv identifier in print.
    if 'Schulman2017PPO' in cited:
        b=blocks['Schulman2017PPO']
        blocks['Schulman2017PPO']=b[:-1].rstrip()+',\n  note = {arXiv:1707.06347}\n}'
    (PAPER/'natural_references.bib').write_text('\n\n'.join(blocks[k] for k in cited)+'\n',encoding='utf-8')
    verification=[]
    for key in cited:
        block=blocks[key];doi=bib_field(block,'doi')
        if doi:
            local=BASE/f'manuscript_reference_metadata_v1/{key}_crossref.json'
            if local.exists():
                src=local;raw=read(local);route='existing verified project metadata'
            else:
                src=OUT/(key+'_crossref.json')
                if not src.exists():
                    response=requests.get('https://api.crossref.org/works/'+quote(doi,safe=''),timeout=30)
                    response.raise_for_status();write_json(src,response.json())
                raw=read(src);route='official Crossref DOI metadata'
            message=raw.get('message',raw)
            assert message['DOI'].lower()==doi.lower(),key
            title=message['title'][0]
            score=SequenceMatcher(None,normalized(title),normalized(bib_field(block,'title'))).ratio()
            assert score>.95,(key,score)
            years={str(message[f]['date-parts'][0][0]) for f in ('published','published-print','published-online','issued') if f in message}
            assert bib_field(block,'year') in years,(key,years)
            for field in ('volume','page'):
                bibkey='pages' if field=='page' else field
                if field in message and bib_field(block,bibkey):
                    assert normalized(str(message[field]))==normalized(bib_field(block,bibkey)),(key,field)
            level='identity and broad method classification; no new full-text reading claimed'
            if key=='Sencer2014':level='identity plus retained full-text/equation-to-code audit (pp.1979-1986 and appendix)'
            verification.append(dict(key=key,doi=doi,title=title,years=sorted(years),status='verified',title_similarity=score,
                source=str(src.relative_to(BASE)),source_sha256=sha(src),route=route,claim_support=level))
        else:
            assert key=='Schulman2017PPO'
            pdf=ROOT/'论文项目/research_20260915_joint_policy_v1/sources/ppo.pdf'
            assert sha(pdf)=='e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b'
            assert bib_field(block,'eprint')=='1707.06347'
            verification.append(dict(key=key,identifier='arXiv:1707.06347',status='verified',source='https://arxiv.org/abs/1707.06347',
                source_sha256=sha(pdf),claim_support='local paper pp.1-3: surrogate objective and repeated minibatch updates'))
    write_json(OUT/'citation_verification.json',dict(verified_utc=datetime.now(timezone.utc).isoformat(),cited_count=len(cited),
        bib_sha256=sha(PAPER/'natural_references.bib'),items=verification,
        corrected_mismatch='S2 text now cites Sencer2014 (IJAMT 76, 1977-1992, volume year 2015), not the distinct CIRP Sencer2015 article.',
        unknown_or_placeholder_citations=0))
    lines=['# 引用核验','',f'实际引用 {len(cited)} 条，均核验身份；无占位引用。只为缺少本地元数据的两条 DOI 查询 Crossref。','',
        '正文 S2 引用已由旧 Sencer2015 键改为 Sencer2014 键；该键对应 IJAMT 卷年 2015、在线年 2014。原始书目保留不改。TOPPRA 与 OptLayer 分别使用不同引用键和 DOI。','',
        '| 引用键 | 标识 | 支持层级 |','|---|---|---|']
    lines += [f"| {r['key']} | {r.get('doi',r.get('identifier'))} | {r['claim_support']} |" for r in verification]
    lines += ['', '身份核验不等于全文精读。除 Sencer 的既有逐式核读记录和 PPO 已读页外，其余引文仅用于题名与元数据可支持的一般方法归类，不承载详细数值或公式主张。未宣称已进行全文级引文计量核验。']
    (OUT/'引用核验.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'derived_data_receipt.json',dict(binding_sha256=sha(CASE/'paper_result_binding.json'),
        new_training_or_simulation=False,files={p.name:sha(p) for p in OUT.glob('*.csv')}))
    print('SUPPLEMENTS_READY',len(corners),'corner rows;',len(section_rows),'class rows;',len(cited),'verified citations')
    print(tx)


if __name__=='__main__':main()

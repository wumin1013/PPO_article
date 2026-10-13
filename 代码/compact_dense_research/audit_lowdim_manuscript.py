"""Traceable checks for the original-format low-dimensional paper revision."""
import json
import argparse
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from build_lowdim_original_manuscript import ROOT, WORK, BASE, DELIVERY, ORIGINAL, SRC, sha, read, write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-dir',type=Path,default=WORK/'build')
    parser.add_argument('--audit-dir',type=Path,default=DELIVERY)
    args=parser.parse_args()
    args.audit_dir.mkdir(parents=True,exist_ok=True)
    b=read(DELIVERY/'paper_result_binding.json')
    for p,h in b['source_sha256'].items(): assert sha(p)==h,p
    for case in b['cases'].values():
        for key in ['reference','geometry']: assert sha(case[key])==case[key+'_sha256']
    assert sha(b['selected_checkpoint'])==b['selected_checkpoint_sha256']
    text=(SRC/'manuscript.tex').read_text(encoding='utf-8')
    log=(args.build_dir/'manuscript.log').read_text(encoding='utf-8')
    assert not re.search(r'Overfull|undefined|multiply defined',log,re.I)
    assert not (SRC/'manuscript.tex').read_bytes().startswith(b'\xef\xbb\xbf')
    assert not (SRC/'references.bib').read_bytes().startswith(b'\xef\xbb\xbf')
    assert '\ufffd' not in text
    cited=set(k.strip() for m in re.findall(r'\\cite\{([^}]+)\}',text) for k in m.split(','))
    bib=(SRC/'references.bib').read_text(encoding='utf-8')
    bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bib))
    bbl_path=args.build_dir/'manuscript.bbl'
    if not bbl_path.exists():bbl_path=SRC/'manuscript.bbl'
    bbl=bbl_path.read_text(encoding='utf-8')
    bblkeys=set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',bbl))
    assert cited==bblkeys and cited<=bibkeys and len(cited)==24
    prior_path=BASE/'paper_evidence_review/verified_results/all34_original_revision/citation_audit.json'
    prior=read(prior_path);lookup={x['key']:x for x in prior['references']}
    assert cited<=lookup.keys()
    refs=[]
    for k in sorted(cited):
        record={key:value for key,value in lookup[k].items() if key!='current_citing_paragraphs'}
        record['current_citing_paragraphs']=[p for p in text.split('\n\n') if any(k in x.split(',') for x in re.findall(r'\\cite\{([^}]+)\}',p))]
        refs.append(record)
    write(args.audit_dir/'citation_audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),tex_sha256=sha(SRC/'manuscript.tex'),bib_sha256=sha(SRC/'references.bib'),bbl_sha256=sha(bbl_path),prior_identity_audit=str(prior_path),prior_identity_sha256=sha(prior_path),cited_and_resolved_count=24,references=refs,scope='Preserved identity verification and current topic-level claim review; no claim that all source papers were newly reread in full.'))
    corners=[]
    for i,dt in enumerate([.001,.0005]):
        main_metric=read(b['methods']['J-NNC']['original'][i]['metrics'])
        m=main_metric.get('metrics',main_metric)['corners'];assert len(m)==34
        for method in ['S2','Two-action']:
            baseline=read(b['methods'][method]['original'][i]['metrics']);bb=baseline.get('metrics',baseline)['corners']
            assert all(c['complete_window'] for c in m+bb)
            lookup2={x['id']:x for x in bb}
            gains=[lookup2[x['id']]['speed_deficit_mm']-x['speed_deficit_mm'] for x in m]
            assert min(gains)>=-1e-7
            corners.append(dict(method=method,dt_s=dt,noninferior_corners=sum(v>=-1e-7 for v in gains),count=34))
    protected={str(i):sha(SRC/'figures_lowdim_revision'/f'Figure_{i}.pdf') for i in range(1,4)}
    assert all(protected[str(i)]==sha(ORIGINAL/'figures'/f'Figure_{i}.pdf') for i in range(1,4))
    experiments=text.split(r'\section{Experiments and Results}')[1].split(r'\section{Conclusion}')[0]
    assert not re.search(r'generaliz|unseen|held.out',experiments,re.I)
    assert text.count(r'\includegraphics')==10 and text.count(r'\ContinuedFloat')==1
    assert r'\subsection{PPO' not in text
    page_text=(args.build_dir/'final_pages.txt').read_text(encoding='utf-8')
    pages=[p for p in page_text.split('\f') if p.strip()]
    assert 20<=len(pages)<=25 and 'Appendix Appendix' not in page_text
    assert all(len(p.strip())>450 for p in pages)
    assert all(str(x) in page_text for x in ['444.178','444.195','14.860','14.9185','6.04','6.14','442.554'])
    # Verify the short local plots do not reach the next dense-case vertices.
    g=read(b['cases']['dense']['geometry']);r=np.loadtxt(b['cases']['dense']['reference'],delimiter=',',skiprows=1)
    local=[]
    for c in g['sharp_corners']:
        if c['id'] in ['corner_00','corner_27']:
            j=c['reference_vertex_index'];dist=np.linalg.norm(np.diff(r[j-1:j+2],axis=0),axis=1)
            assert min(dist)>8.5+1.25
            local.append(dict(corner=c['id'],adjacent_lengths_mm=dist.tolist(),display_y_max_mm=8.5,display_x_min_mm=-6.,plot_scope='Adjacent line-line corridor within finite segment interiors.'))
    stats={}
    for name,cases in b['methods'].items():
        stats[name]={case:[{k:v for k,v in record.items() if not isinstance(v,str)} for record in records] for case,records in cases.items()}
    origin=read(args.audit_dir/'revision_origin.json')
    assert origin['source_sha256']==sha(SRC/'manuscript.tex')
    write(args.audit_dir/'manuscript_integrity_audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),pdf=str(args.build_dir/'manuscript.pdf'),pdf_sha256=sha(args.build_dir/'manuscript.pdf'),source_sha256=sha(SRC/'manuscript.tex'),pages=len(pages),source_origin=origin,source_hashes_verified=True,protected_figures_unchanged=protected,citation_count=24,no_unresolved_references=True,no_overfull_boxes=True,all34= corners,figure8_finite_segment_scope=local,metrics=stats,independent_full_pipeline_repetitions=0,selected_model_count=1,main_figure_count=9,final_manuscript_accepted=False,pending_author_action='Original figures1 and3 retain legacy labels by user instruction; proposed edits supplied separately.'))
    print('MANUSCRIPT_INTEGRITY_PASS',len(pages),'pages;24citations;all34;first3unchanged')

if __name__=='__main__':main()

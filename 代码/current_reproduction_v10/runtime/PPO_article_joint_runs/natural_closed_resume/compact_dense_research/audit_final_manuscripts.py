"""Read-only evidence and manuscript audit; writes only an audit report."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re
import numpy as np

ROOT=Path(r'D:\PPO_article_joint_runs\natural_closed_resume\compact_dense_research')
SUP=ROOT/'experimental_supplement_20260923'
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
DELIVERY=BASE/'paper_evidence_review/verified_results/lowdim_original_revision'
NEW=DELIVERY/'selected923202_revision'
SRC=Path(r'D:\PPO_article_work\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    origin=read(DELIVERY/'revision_origin.json')
    assert sha(origin['base_source'])==origin['base_source_sha256']
    assert sha(origin['base_pdf'])==origin['base_pdf_sha256']
    binding=read(NEW/'paper_result_binding.json')
    assert sha(binding['selected_checkpoint'])==binding['selected_checkpoint_sha256']
    verified=[]
    for p,h in binding['source_sha256'].items():
        assert sha(p)==h,p
        verified.append(p)
    for key in ['acceptance','conservative_dense_source']:
        assert sha(binding[key])==binding[key+'_sha256']
    for case,item in binding['cases'].items():
        for key in ['reference','geometry']:
            assert sha(item[key])==item[key+'_sha256']
    protected={}
    for k,h in origin['protected_figures'].items():
        p=SRC/f'figures_lowdim_revision/Figure_{k}.pdf'
        assert sha(p)==h
        assert sha(NEW/f'latex_source_zh/figures_lowdim_revision/Figure_{k}.pdf')==h
        protected[k]=h
    en=(SRC/'manuscript.tex').read_text(encoding='utf-8')
    zh=(NEW/'latex_source_zh/manuscript_zh.tex').read_text(encoding='utf-8')
    original=Path(origin['base_source']).read_text(encoding='utf-8')
    assert re.search(r'\\title\{([^}]+)',en).group(1)==re.search(r'\\title\{([^}]+)',original).group(1)
    sections=re.findall(r'\\section\{([^}]+)',en)
    assert sections[:5]==origin['main_sections']
    eq=lambda s: re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',s,re.S)
    assert eq(en)==eq(zh)
    cites=lambda s: re.findall(r'\\cite\{([^}]+)',s)
    assert cites(en)==cites(zh)
    citekeys=set(','.join(cites(en)).split(','))
    bib=(SRC/'references.bib').read_text(encoding='utf-8')
    assert len(citekeys)==24
    assert citekeys<=set(re.findall(r'@\w+\{([^,]+),',bib))
    assert sha(SRC/'references.bib')==origin['bibliography_sha256']
    figure_paths=re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)',en)
    assert len(figure_paths)==10 and len(set(re.findall(r'Figure_(\d+)',','.join(figure_paths))))==9
    assert figure_paths==re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)',zh)
    for f in figure_paths:
        assert sha(SRC/'figures_lowdim_revision'/f)==sha(NEW/'latex_source_zh/figures_lowdim_revision'/f)
    aln=read(NEW/'translation_alignment.json')
    assert sha(SRC/'manuscript.tex')==aln['source_sha256']
    assert sha(NEW/'latex_source_zh/manuscript_zh.tex')==aln['translated_sha256']
    # Every displayed complete-time result is checked directly against its bound data.
    for method,cases in binding['methods'].items():
        for case,runs in cases.items():
            for run in runs:
                token=f"{run['complete_time_s']:.3f}" if run['dt_s']==.001 else f"{run['complete_time_s']:.4f}"
                assert token in en and token in zh,(method,case,token)
                token=f"{run['maximum_distance_mm']:.6f}"
                assert token in en and token in zh,(method,case,token)
    students=read(SUP/'three_selected_student_acceptance_v2.json')
    assert students['all_three_selected_checkpoints_pass']
    for student in students['records']:
        assert sha(ROOT/student['checkpoint'])==student['checkpoint_sha256']
        assert student['all_gates_pass']
        for key in ['original_times_s','dense_times_s']:
            for i,t in enumerate(student[key]):
                token=f'{t:.3f}' if i==0 else f'{t:.4f}'
                assert token in en and token in zh
    nnc={}
    for case in ['original','dense']:
        d=SUP/'original_nnc_current_cases'/case
        r=read(d/'result.json'); p=np.load(d/'recorded_prefix.npz')
        assert sha(d/'recorded_prefix.npz')==r['trace_sha256']
        dt=.001
        v=np.diff(p['positions'],axis=0)/dt
        a=np.diff(v,axis=0,prepend=np.zeros((1,2)))/dt
        j=np.diff(a,axis=0,prepend=np.zeros((1,2)))/dt
        vectors=[v,a,j]
        keys=['speed_mm_s','acceleration_mm_s2','jerk_mm_s3']
        peaks={k:float(np.linalg.norm(q,axis=1).max()) for k,q in zip(keys,vectors)}
        for k in keys: assert np.isclose(peaks[k],r['prefix_peaks'][k],rtol=1e-12)
        assert not r['common_complete_success'] and r['complete_time_s'] is None
        assert r['first_common_violation']['tick']==1
        nnc[case]=dict(result=str(d/'result.json'),result_sha256=sha(d/'result.json'),trace=str(d/'recorded_prefix.npz'),trace_sha256=sha(d/'recorded_prefix.npz'),prefix_peaks_independently_reconstructed=peaks,complete_success=False,complete_time_s=None)
    receipt=read(SUP/'original_nnc_current_cases/receipt.json')
    assert receipt['sources_unchanged'] and all(x['exit_code']==0 for x in receipt['records'])
    write(NEW/'nnc_evidence_binding.json',dict(records=nnc,receipt=str(SUP/'original_nnc_current_cases/receipt.json'),receipt_sha256=sha(SUP/'original_nnc_current_cases/receipt.json'),scope='Original preserved NNC planar adaptation; feasibility of first-second prefix only. No whole-run speed or time comparison.'))
    old_citations=read(DELIVERY/'citation_audit.json')
    refs=[]
    for r in old_citations['references']:
        item=dict(r)
        item['current_citing_paragraphs']=[p for p in en.split('\n\n') if any(r['key'] in x.split(',') for x in cites(p))]
        refs.append(item)
    write(NEW/'citation_audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),tex_sha256=sha(SRC/'manuscript.tex'),bib_sha256=sha(SRC/'references.bib'),cited_and_resolved_count=24,prior_identity_audit=str(DELIVERY/'citation_audit.json'),prior_identity_sha256=sha(DELIVERY/'citation_audit.json'),verification_scope='Unchanged bibliography checked against previously verified identity metadata; current citation contexts inspected. No new external literature search.',references=refs))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),base_pdf_sha256=origin['base_pdf_sha256'],base_source_sha256=origin['base_source_sha256'],english_source_sha256=sha(SRC/'manuscript.tex'),chinese_source_sha256=sha(NEW/'latex_source_zh/manuscript_zh.tex'),protected_figures=protected,verified_bound_sources=verified,main_figure_count=9,figure4_continued=True,main_sections=sections[:5],equation_environments_identical=len(eq(en)),citation_sequence_identical=True,cited_reference_count=24,full_result_table_values_checked=True,student_table_values_and_checkpoints_checked=True,nnc_prefix_independent_xy_reconstruction_checked=True,selected_controller_frozen=True,uniform_full_pipeline_three_seed_repetitions=False,paired_action_only_five_percent_demonstrated=False,remaining_author_action='Update original Figure 1 and Figure 3 labels per revision notes; Figure 2 unchanged.',final_manuscript_accepted=False)
    write(NEW/'content_audit.json',report)
    print('Content audit passed:',len(verified),'bound sources;',len(eq(en)),'matching equations; 24 citations; 9 main figures; both result tables; NNC XY audit.')

if __name__=='__main__':main()

"""Verify final evidence bindings and artifacts without repeating any rollout."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageChops
from study_io import BASE, ROOT, sha, write_json

case=BASE/'paper_evidence_review/natural_closed_case'
paper=BASE/'manuscript_joint_v1'
checks=[]


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def check_hash(p,expected,role):
    assert p.is_file(),p
    actual=sha(p)
    assert actual==expected,(role,p,actual,expected)
    checks.append(dict(path=str(p),sha256=actual,role=role,status='pass'))


binding=read(case/'paper_result_binding.json')
freeze=read(case/'method_freeze.json')
check_hash(case/'reference.csv',binding['reference_sha256'],'unchanged common reference')
check_hash(case/'geometry.json',binding['geometry_sha256'],'unchanged analytic geometry')
check_hash(BASE/freeze['checkpoint'],freeze['checkpoint_sha256'],'selected learned checkpoint')
check_hash(case/'paper_result_binding.json',freeze['paper_result_binding_sha256'],'freeze binding')
for name,entry in binding['primary_methods'].items():
    for kind in ('trace','result','metrics'):
        check_hash(BASE/entry[kind],entry[kind+'_sha256'],name+' '+kind)
    r=read(BASE/entry['result']);m=read(BASE/entry['metrics'])
    assert all((r['success'],r['geometry']['ordered_complete'],r['audit']['kinematic_pass'],
                r['audit']['rest_pass'],r['analytic']['analytic_corridor_pass']))
    assert r['geometry']['gates_passed']==r['geometry']['gate_count']==2362
    assert r['analytic']['unresolved_segments']==0
    assert m['all34_windows_complete'] and len(m['corners'])==34 and m['projection_raw_backward_steps']==0
for folder in ('execution_diagnostic','support_calibration','s2_complete','fixed_historical',
               'fixed_support_calibration','fixed_conditional_training','resolution_half_ms'):
    stage=case/folder;receipt=read(stage/'receipt.json')
    for key,name in [('registration_sha256','registration.json'),('result_sha256','result.json'),
                     ('results_sha256','results.json'),('selected_policy_sha256','selected_policy.json')]:
        if key in receipt:check_hash(stage/name,receipt[key],folder+' completed receipt')
    reg=read(stage/'registration.json')
    for rel,expected in reg.get('source_sha256',{}).items():
        original=ROOT/rel
        if original.is_file() and sha(original)==expected:
            check_hash(original,expected,folder+' registered source')
        else:
            preserved=next((p for p in (stage/'source').iterdir() if p.is_file() and sha(p)==expected),None)
            assert preserved is not None,(folder,rel,'no preserved source matches')
            check_hash(preserved,expected,folder+' preserved source snapshot')
    assert receipt.get('wall_s',0)<=1800,(folder,'wall cap exceeded')
for name in ('receipt.json','scope_update_receipt.json'):
    r=read(case/name)
    for rel,expected in r.get('outputs_sha256',{}).items():
        check_hash(BASE/rel,expected,'retained static '+name)
half=read(case/'resolution_half_ms/result.json')
check_hash(case/'resolution_half_ms/trace.npz',half['trace_sha256'],'half-ms physical run')
assert half['audit']['rest_audit_span_s']>=.02 and half['audit']['rest_pass']
figreceipt=read(paper/'natural_case_figures/receipt.json')
check_hash(case/'paper_result_binding.json',figreceipt['source_binding_sha256'],'figure input binding')
for name,expected in figreceipt['files_sha256'].items():
    check_hash(paper/'natural_case_figures'/name,expected,'figure export')
latest=read(paper/'natural_pdf_qa/latest.json');qa=BASE/latest['directory']
check_hash(paper/'build_natural/PPO_article.pdf',latest['pdf_sha256'],'inspected final PDF')
struct=read(qa/'structural_audit.json')
assert struct['page_count']==23 and not struct['latex_issues']
assert not any(p['outside_page_safe_region'] for p in struct['page_records'])
older=paper/'natural_pdf_qa/f66abba17da2'
same=[];changed=[]
for i in range(1,24):
    a=Image.open(qa/f'page-{i:02d}.png').convert('RGB')
    b=Image.open(older/f'page-{i:02d}.png').convert('RGB')
    (same if a.size==b.size and ImageChops.difference(a,b).getbbox() is None else changed).append(i)
assert same==list(range(1,23)) and changed==[23],(same,changed)
cites=read(paper/'natural_case_supplements/citation_verification.json')
check_hash(paper/'natural_references.bib',cites['bib_sha256'],'verified bibliography')
assert cites['cited_count']==9 and all(r['status']=='verified' for r in cites['items'])
csvpath=paper/'natural_case_supplements/all_34_corners_three_methods.csv'
with csvpath.open(encoding='utf-8',newline='') as f:assert len(list(csv.DictReader(f)))==102
textpaths=list(paper.glob('natural_*.tex'))+[paper/'PPO_article.tex',paper/'natural_references.bib']
textpaths += [BASE/n for n in ('bind_natural_paper_results.py','natural_case_metrics.py','draw_natural_paper_figures.py',
    'prepare_natural_manuscript.py','compile_natural_manuscript.py','render_natural_pdf_qa.py','verify_natural_delivery.py')]
for p in textpaths:
    raw=p.read_bytes();assert not raw.startswith(b'\xef\xbb\xbf'),p
    assert '\ufffd' not in raw.decode('utf-8'),p
m=read(case/'joint_paper_metrics.json')
totals={key:sum(c[key] for c in m['corners']) for key in ('reverse_heading_deg','reverse_lateral_mm','outgoing_overshoot_mm')}
assert all(c['first_turn_30mm_audit_inside'] and not c['first_turn_30mm_audit_left_censored'] for c in m['corners'])
report=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='pass',hash_checks=checks,
    utf8_no_bom_checked=len(textpaths),same_rendered_pages_after_bibliographic_identifier_fix=same,
    final_changed_pages_requiring_visual_review=changed,main_all_corner_shape_totals=totals,
    no_training_or_simulation_in_delivery_verification=True)
write_json(paper/'natural_case_supplements/final_evidence_verification.json',report)
print('VERIFIED',len(checks),'hash bindings;',len(textpaths),'UTF-8 files; pages1-22 pixel-identical; final page23 needs explicit review.')
print('SHAPE_TOTALS',totals)

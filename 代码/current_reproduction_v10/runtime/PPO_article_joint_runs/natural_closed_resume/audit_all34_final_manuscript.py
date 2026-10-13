"""Audit the reviewed manuscript and frozen evidence without running simulations."""
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r'D:\PPO_article_work')
BASE = ROOT / '论文项目/joint_policy_20260915_v1'
OUT = Path(r'D:\PPO_article_joint_runs\natural_closed_resume')
R = OUT / 'all34_dominance_research'
D = BASE / 'paper_evidence_review/verified_results/all34_original_revision'
SRC = ROOT / '论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous'
BUILD = OUT / 'all34_original_revision_build'
QA = OUT / 'all34_pdf_page_qa'
FIG = D / 'figures_publication'
now = datetime.now(timezone.utc).isoformat()
checks, hashes = [], {}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    p = Path(p)
    if str(p) not in hashes:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(data)
        hashes[str(p)] = h.hexdigest()
    return hashes[str(p)]

def item(p):
    return {'path': str(p), 'sha256': sha(p)}

def check(name, ok, detail=None):
    checks.append({'name': name, 'passed': bool(ok), 'detail': detail})

def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def bound_pairs(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k.endswith('_sha256') and isinstance(v, str) and k[:-7] in value:
                p = value[k[:-7]]
                if isinstance(p, str):
                    check('bound hash: ' + p, Path(p).is_file() and sha(p) == v)
            elif isinstance(v, dict) and k.endswith('_sha256'):
                for p, digest in v.items():
                    if re.match(r'^[A-Z]:[\\/]', p, re.I):
                        check('bound hash: ' + p, Path(p).is_file() and sha(p) == digest)
            if isinstance(v, (dict, list)):
                bound_pairs(v)
    elif isinstance(value, list):
        for v in value:
            bound_pairs(v)

binding = read(D/'paper_result_binding.json')
origin = read(D/'revision_origin.json')
contract = read(FIG/'figure_contract.json')
research = read(D/'research_acceptance.json')
paired = read(binding['paired_assessment'])
mechanism = read(binding['core_mechanism_review'])
bound_pairs(binding)
bound_pairs(origin)
check('research acceptance', research['passed'])
check('core numerical acceptance', paired['quantitative_core_passed'])
check('core mechanism acceptance', mechanism['passed'])
for res, v in paired['corner_comparisons'].items():
    check('all34 paired ' + res, v['passed'] and len(v['records']) == 34 and all(x['passed'] for x in v['records']))
for name, v in paired['time_comparisons'].items():
    check('strict time margin ' + name, v['passed'] and v['nominal_gain_s'] > v['required_gain_s'] and v['half_ms_gain_s'] > v['required_gain_s'])

tex = (SRC/'manuscript.tex').read_text(encoding='utf-8')
old = Path(origin['base_source']).read_text(encoding='utf-8-sig')
bib = (SRC/'references.bib').read_text(encoding='utf-8')
bbl = (SRC/'manuscript.bbl').read_text(encoding='utf-8')
log = (BUILD/'manuscript.log').read_text(encoding='utf-8', errors='replace')
title = re.search(r'\\title\{([^}]+)\}', tex).group(1)
check('original title unchanged', title == re.search(r'\\title\{([^}]+)\}', old).group(1), title)
def preamble(t):
    return '\n'.join(x for x in t.split(r'\begin{document}')[0].splitlines() if not x.startswith(r'\graphicspath'))
check('original preamble except graphics directory', preamble(tex) == preamble(old))
sections = re.findall(r'\\section\{([^}]+)\}', tex)
check('original sections unchanged', sections == re.findall(r'\\section\{([^}]+)\}', old), sections)
experiment = tex.split(r'\section{Experiments and Results}')[1].split(r'\section{Conclusion}')[0]
subs = re.findall(r'\\subsection\{([^}]+)\}', experiment)
check('original experimental subsections', subs == ['Experimental Settings', 'Main Comparative Results', 'Ablation Results'], subs)
check('no experimental generalization discussion', not re.search(r'generaliz|generalis|unseen|held[ -]out|out.of.distribution', experiment, re.I))
check('no placeholder text', not re.search(r'\bTODO\b|\bTBD\b|PLACEHOLDER|citation needed', tex, re.I))
check('no superseded primary times', not re.search(r'450\.119|454\.700|444\.742|444\.7605|444\.882|444\.8805', tex))
check('dimension optimality not asserted', 'a ranking of alternative dimensions is not asserted' in tex)
check('repetition scope stated', 'not independent PPO training repetitions' in tex and 'not matched end-to-end training budgets' in tex)
check('no unresolved citations or overfull boxes', not re.search(r'Overfull|LaTeX Warning|Package natbib Warning|undefined references|undefined citations', log))
receipt = read(BUILD/'compile_receipt.json')
check('compile completed', receipt['exitCode'] == 0 and receipt['pdfExists'])
info = subprocess.run(['pdfinfo', str(BUILD/'manuscript.pdf')], capture_output=True, text=True, encoding='utf-8', errors='replace')
pages = int(re.search(r'^Pages:\s+(\d+)', info.stdout, re.M).group(1))
check('24 PDF pages', info.returncode == 0 and pages == 24)
check('anonymous PDF metadata', not re.search(r'^Author:\s*\S', info.stdout, re.M))
figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
expected_figures = [f'Figure_{n}.pdf' for n in range(1,10)]
expected_figures.insert(4, 'Figure_4_execution.pdf')
check('nine main figures plus Fig4 continuation', figures == expected_figures and tex.count(r'\ContinuedFloat') == 1, figures)
protected = []
for n in range(1,4):
    name = f'Figure_{n}.pdf'
    paths = [Path(origin['base_source']).parent/'figures'/name, FIG/name, SRC/'figures_all34_revision'/name]
    values = [item(p) for p in paths]
    check('protected figure ' + str(n), len({x['sha256'] for x in values}) == 1)
    protected.append({'figure': n, 'files': values})
for name in ['elsarticle.cls', 'unsrt.bst']:
    check('original style file ' + name, sha(SRC/name) == sha(Path(origin['base_source']).parent/name))
for p, digest in contract['source_sha256'].items():
    check('figure generator ' + p, sha(p) == digest)
for name, digest in contract['output_sha256'].items():
    # The generator's old self-entry was captured before rewriting its manifest.
    # A self-hash is not verifiable. The final audit binds the actual contract externally.
    if name == 'figure_contract.json':
        continue
    check('figure output ' + name, sha(FIG/name) == digest)
    if name in figures:
        check('LaTeX figure copy ' + name, sha(SRC/'figures_all34_revision'/name) == digest)

cite_keys = {key.strip() for group in re.findall(r'\\cite\w*\s*(?:\[[^\]]*\])?\{([^}]+)\}', tex) for key in group.split(',')}
blocks = re.split(r'(?m)^@\w+\s*\{', bib)[1:]
bib_entries = {b.split(',',1)[0].strip():b for b in blocks}
bbl_entries = {k:b for k,b in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})', bbl, re.S)}
prior_audit_path = BASE/'paper_evidence_review/verified_results/original_framework_revision/citation_audit.json'
prior_audit = read(prior_audit_path)
previous = {x['key']: x for x in prior_audit['references']}
metadata_path = BASE/'manuscript_reference_metadata_v1/results.json'
metadata = {x['key']:x for x in read(metadata_path)['results']}
check('24 distinct citations resolve in bib and bbl', len(cite_keys) == 24 and cite_keys <= set(bib_entries) and cite_keys == set(bbl_entries))
def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower().replace('\\newblock', '').replace('\\em', ''))
def field(block, key):
    match = re.search(r'(?mi)^\s*' + key + r'\s*=\s*\{(.*?)\}\s*,?\s*$', block)
    return match.group(1) if match else None
refs = []
for key in sorted(cite_keys):
    block = bib_entries[key]
    p = previous[key]
    title_bib = field(block, 'title')
    if key == 'Schulman2017PPO':
        check('PPO arxiv bibliography', '1707.06347' in block and '1707.06347' in bbl_entries[key])
        bound_pairs(p)
        entry = {'key':key, 'identity':p, 'verification_scope':'Saved arXiv identity and selected relevant passages; no full-paper reread claim.'}
    else:
        m = metadata[key]
        check('verified citation identity ' + key, m['status'] == 'verified_identity')
        check('citation DOI ' + key, p['doi'].lower() in block.lower() and p['doi'].lower() == m['doi'].lower())
        check('citation title ' + key, norm(title_bib or '') == norm(m['retrieved_title']), title_bib)
        check('citation year ' + key, field(block,'year') in m['retrieved_years'])
        check('compiled bibliography title ' + key, norm(title_bib or '') in norm(bbl_entries[key]))
        entry = {'key':key, 'doi':m['doi'], 'title':title_bib, 'year':field(block,'year'), 'metadata_source':m['source'], 'verification_scope':'Publication identity from saved metadata; claim checked at the indicated topic level.'}
    entry['current_citing_paragraphs'] = [x.strip() for x in tex.split('\n\n') if any(key in g.split(',') for g in re.findall(r'\\cite\w*\{([^}]+)\}', x))]
    refs.append(entry)
check('Sencer author correction', all('Kosuke' in bib_entries[k] and 'Kazunori' not in bib_entries[k] for k in ['Sencer2014','Sencer2015']))
check('Li2020 supported claim only', all('reward' not in x.lower() and 'two' not in x.lower() for x in [q for q in tex.split('\n\n') if r'\cite{Li2020}' in q]))

times = {}
for name, v in binding['primary_methods'].items():
    result, fine = read(v['result']), read(binding['resolution_results'][name])
    times[name] = {'1_ms':result['complete_time_s'], '0.5_ms':fine['complete_time_s']}
    check('complete success ' + name, result['success'] and fine['success'])
    check('primary time in paper ' + name, f"{times[name]['1_ms']:.3f}" in tex and f"{times[name]['0.5_ms']:.4f}" in tex)
rows = list(csv.DictReader((D/'all34_corner_statistics.csv').open(encoding='utf-8')))
groups = {(m,res):{r['id']:r for r in rows if r['method']==m and r['resolution']==res} for m in ['Joint','S2','Fixed'] for res in ['1ms','0.5ms']}
all34 = {}
for res in ['1ms','0.5ms']:
    for m in ['Joint','S2','Fixed']:
        check('source CSV all34 ' + m + res, len(groups[m,res]) == 34 and all(x['complete_window']=='True' for x in groups[m,res].values()))
    all34[res] = {}
    for comparator in ['S2','Fixed']:
        gaps = [float(groups['Joint',res][c]['speed_deficit_mm'])-float(groups[comparator,res][c]['speed_deficit_mm']) for c in groups['Joint',res]]
        all34[res][comparator] = {'passed':all(g <= 1e-7 for g in gaps),'strictly_lower_beyond_tolerance':sum(g < -1e-7 for g in gaps),'numerical_ties':sum(abs(g)<=1e-7 for g in gaps),'maximum_gap_mm':max(gaps)}
        check('CSV paired deficit ' + comparator + res, all34[res][comparator]['passed'])
cruise = read(FIG/'Figure_7_cruise_statistics.json')
check('cruise threshold contract', '99' in contract['figure7'] and contract['all_34_corners'])
check('mechanism prescribed columns', contract['prescribed_columns'] == ['corner_01','corner_02','corner_06'])
check('case grid spacing', contract['grid_spacing_mm'] == 10)

code_paths = [SRC/'manuscript.tex', SRC/'references.bib', SRC/'manuscript.bbl', Path(__file__)]
code_paths += [Path(p) for p in origin['drafting_reference']]
code_paths += [Path(p) for p in contract['source_sha256']]
for p in code_paths:
    raw = p.read_bytes()
    try:
        raw.decode('utf-8')
        valid = not raw.startswith(b'\xef\xbb\xbf')
    except UnicodeDecodeError:
        valid = False
    check('UTF-8 without BOM ' + str(p), valid)
branch = subprocess.run(['git','branch','--show-current'], cwd=BASE, capture_output=True, text=True).stdout.strip()
check('same branch', branch == 'codex/complex-path-adaptive-preview', branch)
check('Poppler rendered without error', (QA/'render_stderr.txt').read_text(encoding='utf-8').strip() == '')
page_notes = [
    'Title and abstract; original typography and complete-time claims legible.',
    'Introduction and citations; no overlap or clipping.',
    'Introduction and protected Figure 1; original drawing retained.',
    'Problem formulation and original geometric structure; equations readable.',
    'Protected Figure 2 and execution definitions; no clipping.',
    'Protected Figure 3 rendered intact; labels legible despite extraction CMap warning.',
    'Observation table and causal state; necessary corrections readable.',
    'Action decoder and vector constraint equations; no overflow.',
    'Projection algorithm and reward; corrected inline sensor text fits.',
    'PPO provenance and final refinement; coefficients readable.',
    'Experimental settings and complete acceptance definitions.',
    'Full comparison tables and numerical margins; values match frozen evidence.',
    'Case annotations, extremes and 10 mm short-straight grid clearly resolve.',
    'Figure 4 continuation; complete vector V/A/J and physical rest readable.',
    'Efficiency decomposition and approximate tangent recovery description.',
    'Figure 5 representative local trajectories and enlarged comparison areas.',
    'Representative-corner measurements and six-row Figure 6 mechanism.',
    'Figure 7 all34 deficits and cruise presentation; no dense blue band.',
    'Ablation definitions, table and real intervention failure times readable.',
    'Figure 8 actual local failure trajectories and conditional refinement provenance.',
    'Figure 9 resolution gains and shared-model candidates clearly distinguished.',
    'Discussion continuation and conclusion; complete values and scope accurate.',
    'References 1-12; no clipping, overlap or unresolved entries.',
    'References 13-24; PPO arXiv identity and Sencer author/year corrections present.'
]
page_records = []
for n, note in enumerate(page_notes,1):
    p = QA/f'page-{n:02}.png'
    check('reviewed page image ' + str(n), p.is_file())
    page_records.append({'page':n, 'image':item(p), 'visually_reviewed':True, 'note':note})
qa = {'created_utc':now, 'pdf':item(BUILD/'manuscript.pdf'), 'pages':pages, 'all_pages_visually_reviewed':True,
      'review_scope':'Actual per-page image inspection across this continuing task; pages 1-18 inspected before context continuation, 19-24 inspected after continuation.',
      'page_records':page_records, 'remaining_visual_defects':[],
      'known_nonvisual_messages':['One underfull hbox in the observation table; visually inspected without a layout defect.',
       'xdvipdfmx notes original Figure 3 PDF version/tag compatibility; protected figure preserved and successfully rendered.',
       'TeX Live pdftotext Adobe-GB1 CMap warnings affect original embedded drawing text extraction; page rendering is intact.']}
citations = {'created_utc':now,'tex':item(SRC/'manuscript.tex'),'bib':item(SRC/'references.bib'),'bbl':item(SRC/'manuscript.bbl'),
             'metadata':item(metadata_path),'prior_identity_audit':item(prior_audit_path),'cited_and_resolved_count':24,
             'scope':'Current citation resolution, bibliography identity and current claim-scope review against preserved verification sources. This does not claim every cited paper was reread in full.', 'references':refs}
audit = {'created_utc':now,'passed':all(x['passed'] for x in checks),'checks':checks,'check_count':len(checks),
         'failed_checks':[x for x in checks if not x['passed']], 'complete_times_s':times,'all34_comparisons':all34,
         'protected_figures':protected, 'pdf':item(BUILD/'manuscript.pdf'),'paper_binding':item(D/'paper_result_binding.json'),
         'current_figure_contract':item(FIG/'figure_contract.json'),
         'contract_self_hash_note':'The original output_sha256.figure_contract.json is a stale prior-file self-entry and is not a valid self-certificate. Every figure output and source is independently checked here; this record binds the current contract externally.',
         'pdfinfo':info.stdout,'cruise_statistics':cruise,'verified_file_sha256':hashes}
write(OUT/'all34_final_static_audit.json', audit)
if audit['passed']:
    for name,obj in [('final_static_audit.json',audit),('page_visual_qa.json',qa),('citation_audit.json',citations)]:
        if (D/name).exists():
            raise FileExistsError(f'Final audit already exists: {D/name}')
        write(D/name,obj)
print(json.dumps({'passed':audit['passed'],'checks':len(checks),'failed':audit['failed_checks'],'times':times,'all34':all34}, ensure_ascii=False, indent=2))
if not audit['passed']:
    raise SystemExit(1)

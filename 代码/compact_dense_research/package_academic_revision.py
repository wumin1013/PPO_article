"""Audit and deliver the academic figure revision without rerunning research."""
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
BASE = Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1')
OLD = BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'
PREV = OLD/'figure_revision_20260924'
OUT = OLD/'academic_figures_20260924'
FIG = OUT/'figures'
WORK = ROOT/'lowdim_original_manuscript_work/academic_figure_revision'
SRC = Path('D:/PPO_article_work/论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pixel_sha(p):
    with Image.open(p) as im:
        return hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()


def strip_figures(t):
    return re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', t, flags=re.S)


def equations(t):
    return re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', t, flags=re.S)


def table_numbers(t, label):
    block = next(b for b in re.findall(r'\\begin\{table\}(.*?)\\end\{table\}', t, re.S) if label in b)
    rows = [s for s in block.splitlines() if '&' in s and re.search(r'\d', s)]
    return sorted(re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', '\n'.join(rows)))


def audit():
    inv = read(OUT/'render_inventory.json')
    v2 = read(OUT/'render_inventory_v2.json')
    delivered = read(OUT/'render_inventory_delivery.json')
    notes = read(OUT/'notes_render_inventory.json')
    locations = read(OUT/'figure_table_locations.json')
    qa, texts, source_hashes = {}, {}, {}
    for lang, name, pages in [('en', 'manuscript', 22), ('zh', 'manuscript_zh', 19)]:
        info = inv[lang]
        assert info['pages'] == pages and not info['compile_issues']
        assert sha(info['pdf']) == info['sha256'] and len(PdfReader(info['pdf']).pages) == pages
        reviewed = list(v2[lang]['renders'])
        if lang == 'en':
            reviewed += [info['renders'][5]]  # Final page 6 individually viewed.
        else:
            reviewed += delivered['zh']['renders'][8:]  # Contacts 3--5 viewed.
            reviewed += info['renders'][12:]  # Final compact contacts 4--5 viewed.
            reviewed += [info['renders'][4]]  # Final page 5 individually viewed.
        by_pixel = {pixel_sha(p): p for p in reviewed}
        review_rows = []
        for i, p in enumerate(info['renders'], 1):
            digest = pixel_sha(p)
            assert digest in by_pixel, f'Unreviewed changed page: {lang} {i}'
            review_rows.append(dict(page=i, render=p, pixel_sha256=digest,
                                    review_reference=by_pixel[digest], layout_pass=True))
        qa[lang] = dict(pages=pages, pdf_sha256=info['sha256'], pages_reviewed=review_rows, compile_issues=[])
        directory = OUT/f'latex_source_{lang}'
        t = (directory/(name+'.tex')).read_text(encoding='utf-8')
        old = (PREV/f'latex_source_{lang}'/(name+'.tex')).read_text(encoding='utf-8')
        texts[lang] = t
        main, appendix = t.split('\n\\appendix\n', 1)
        assert main.count('\\begin{figure}') == 9 and appendix.count('\\begin{figure}') == 1
        assert main.count('\\begin{table}') == 3 and main.count('0.5 ms') == 1
        assert not re.search(r'\bS2\b|探索过程|失败修复|逐轮日志|证据目录|exploratory\s+(?:process|history)|failure\s+repair|round[- ]by[- ]round|evidence\s+directory', t, re.I)
        start = '\\section{'+('Methodology' if lang == 'en' else '方法')+'}'
        end = '\\section{'+('Experiments and Results' if lang == 'en' else '实验与结果')+'}'
        assert strip_figures(old.split(start)[1].split(end)[0]) == strip_figures(t.split(start)[1].split(end)[0])
        assert equations(old) == equations(t)
        assert re.findall(r'\\title\{([^}]+)', old) == re.findall(r'\\title\{([^}]+)', t)
        assert re.findall(r'\\(?:sub)*section\{([^}]+)', old) == re.findall(r'\\(?:sub)*section\{([^}]+)', t)
        for label in ['tab:main_results', 'tab:utilization', 'tab:selected_students']:
            assert table_numbers(old, label) == table_numbers(t, label), (lang, label)
        for suffix in [*[str(i) for i in range(1, 10)], 'C1']:
            assert sha(directory/'figures_lowdim_revision'/f'Figure_{suffix}.pdf') == sha(FIG/f'Figure_{suffix}.pdf')
        bbl = Path(info['pdf']).with_suffix('.bbl')
        if not bbl.exists():
            bbl = directory/(name+'.bbl')
        keys = re.findall(r'\\bibitem\{([^}]+)', bbl.read_text(encoding='utf-8'))
        previous_keys = re.findall(r'\\bibitem\{([^}]+)', (PREV/f'latex_source_{lang}'/(name+'.bbl')).read_text(encoding='utf-8'))
        assert len(keys) == 24 and keys == previous_keys
        assert sha(directory/'references.bib') == sha(PREV/f'latex_source_{lang}/references.bib')
        assert sha(directory/(name+'.bbl')) == sha(bbl)
        for p in directory.rglob('*'):
            if p.is_file():
                source_hashes[str(p)] = sha(p)
        all_text = '\n'.join(p.extract_text() for p in PdfReader(info['pdf']).pages)
        assert not re.search(r'\bS2\b', all_text)
        appendix_page = min(v['page'] for k,v in locations[lang].items() if k in ['tab:ppo_details','tab:resolution'])
        assert all(v['page'] < appendix_page for k,v in locations[lang].items() if k.startswith('fig:') and v['number'] != 'C.1')
    assert equations(texts['en']) == equations(texts['zh'])
    assert re.findall(r'\\label\{([^}]+)', texts['en']) == re.findall(r'\\label\{([^}]+)', texts['zh'])
    assert notes['pages'] == 4 and not notes['compile_issues']
    assert notes['pdf_sha256'] == sha(OUT/'图件优化与非实验修改说明.pdf')
    qa['revision_notes'] = dict(pages=4, pdf_sha256=notes['pdf_sha256'], pages_reviewed=[
        dict(page=i,render=p,pixel_sha256=pixel_sha(p),layout_pass=True) for i,p in enumerate(notes['renders'],1)])
    vectors = read(OUT/'original_vector_edit_audit.json')
    for i in [1,3]:
        v = vectors['figures'][str(i)]
        assert sha(v['original']) == v['original_sha256']
        assert sha(FIG/f'Figure_{i}.pdf') == v['output_sha256']
        pdf = PdfReader(FIG/f'Figure_{i}.pdf')
        assert not list(pdf.pages[0].images), 'Original diagrams must remain vectors'
    assert sha(FIG/'Figure_2.pdf') == sha(PREV/'figures/Figure_2.pdf')
    binding = read(OUT/'figure_data_binding.json')
    assert sha(binding['source_binding']) == binding['source_binding_sha256']
    for p,digest in binding['source_sha256'].items():
        assert sha(p) == digest
    c = read(FIG/'Figure_6_corner_summary.json')
    curve = np.loadtxt(FIG/'Figure_6_source_data.csv',delimiter=',',skiprows=1)
    for i,method in enumerate(['J-NNC','Two-action','S2']):
        rows = curve[(curve[:,0] == i) & (abs(curve[:,1]) <= 10)]
        assert abs(rows[:,2].min()-c['stats'][method]['min_speed_mm_s']) < 1e-9
        assert abs(.001*np.maximum(100-rows[:,2],0).sum()-c['stats'][method]['deficit_mm']) < 1e-9
    for key in ['loss_reduction_vs_two_step_pct','loss_reduction_vs_two_action_pct']:
        assert f'{c[key]:.1f}\\%' in texts['en'] and f'{c[key]:.1f}\\%' in texts['zh']
    stats = read(PREV/'utilization_statistics.json')
    for r in stats['records']:
        for key in ['mean_V_percent','p95_A_percent','p95_J_percent']:
            assert f'{r[key]:.3f}' in texts['en'] and f'{r[key]:.3f}' in texts['zh']
    return inv, qa, source_hashes, binding, locations


def main():
    inv, qa, sources, binding, locations = audit()
    if '--audit-only' in sys.argv:
        print('AUDIT PASSED: 22 EN + 19 ZH + 4 notes pages; figures, tables, equations and references bound.')
        return
    now = datetime.now(timezone.utc)
    hist = WORK/'delivery_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
    hist.mkdir(parents=True)
    for p in [OUT/'figure_data_binding.json', ROOT/'record_current_supplement.py']:
        shutil.copy2(p, hist/p.name)
    for lang in ['en','zh']:
        shutil.copy2(inv[lang]['pdf'], OUT/f'manuscript_revised_{lang}.pdf')
    for name in ['utilization_statistics.json','utilization_statistics.csv']:
        shutil.copy2(PREV/name, OUT/name)
    binding.update(updated_utc=now.isoformat(), nnc_binding=str(OLD/'nnc_evidence_binding.json'),
        nnc_binding_sha256=sha(OLD/'nnc_evidence_binding.json'),
        outputs={p.name:sha(p) for p in sorted(FIG.iterdir()) if p.is_file()},
        script_sha256=sha(ROOT/'draw_academic_revision.py'),
        vector_edit_script_sha256=sha(ROOT/'revise_original_vectors_academic.py'),
        original_vector_edit_audit=str(OUT/'original_vector_edit_audit.json'),
        display_order=['J-NNC','Two-action','Two-step planner'])
    binding['metrics']['NNC'] = read(PREV/'new_figure_metrics.json').get('nnc', {})
    write(OUT/'figure_data_binding.json', binding)
    write(OUT/'page_visual_qa.json', dict(created_utc=now.isoformat(),documents=qa,layout_pass=True,
        scope='All pages visually inspected in recorded contacts or individually; unchanged pages matched by RGB pixel hash.'))
    write(OUT/'content_audit.json', dict(created_utc=now.isoformat(), main_figures=9,supporting_figures=1,
        main_result_tables=3,source_sha256=sources,method_prose_and_equations_unchanged=True,
        method_figure3_and_caption_updated=True,titles_and_section_structure_unchanged=True,
        original_figure2_unchanged=True,original_figures1_and3_vector_copies_revised=True,
        original_files_preserved=True,citations_24_unchanged=True,citation_identity_audit=str(OLD/'citation_audit.json'),
        main_half_ms_mentions=1,exploration_logs_in_manuscript=False,figure_table_locations=locations,
        all_result_table_numbers_unchanged=True,local_corner_loss_verified_from_csv=True,
        figures_editorial_only=True,no_new_training_or_simulation=True))
    write(OUT/'final_figure_binding.json',dict(main_figure_count=9,supporting_figure_count=1,
        figures={p.name:dict(source=str(p),sha256=sha(p),data_binding=str(OUT/'figure_data_binding.json'))
                 for p in sorted(FIG.glob('*.pdf'))}, main_model_seed=923202,
        main_model_sha256=read(OLD/'paper_result_binding.json')['selected_checkpoint_sha256'],
        original_vector_edits=str(OUT/'original_vector_edit_audit.json')))
    (OUT/'README.md').write_text('''# 图件学术化修订交付

- manuscript_revised_en.pdf：英文22页；manuscript_revised_zh.pdf：中文19页。
- 图件优化与非实验修改说明.pdf / .md：4页说明，逐图修改、原图标签对照、实验外修改及页码。
- latex_source_en / latex_source_zh：与PDF绑定的中英文可编译源码。
- figures：九幅主图和附录C.1；PDF、可编辑SVG、PNG与绘图数据。

图1、3在作者原矢量PDF副本上修改，原图2逐字节保留，所有历史原图均未覆盖。
Figure_1_editable_labels.svg和Figure_3_editable_labels.svg仅包含新增标签及局部节点，不是完整原图SVG。完整修订图请打开对应PDF；未保留原Illustrator图层/编组历史。图4—9及C.1提供完整SVG。

模型、轨迹、物理约束和实验数值没有变化。本轮不训练、不仿真；原来图1/3的作者强制待办已完成。
正文包含九幅主图与三张结果表，完整V/A/J及停稳图置附录C；参数表及KCM算法留附录。探索过程与逐轮日志不在论文中。

TeX Live + XeLaTeX + BibTeX编译：在对应源码目录执行latexmk -xelatex manuscript.tex或latexmk -xelatex manuscript_zh.tex。文本为UTF-8无BOM。
''', encoding='utf-8')
    archives = [OUT/'latex_sources_en_zh.zip', OUT/'figures_editable.zip']
    shared = ['README.md','图件优化与非实验修改说明.md','图件优化与非实验修改说明.pdf',
              'figure_data_binding.json','final_figure_binding.json','original_vector_edit_audit.json',
              'utilization_statistics.json','utilization_statistics.csv']
    for archive in archives:
        assert not archive.exists(), f'Archive exists; preserve history before rerun: {archive}'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            if archive == archives[0]:
                for lang in ['en','zh']:
                    for p in sorted((OUT/f'latex_source_{lang}').rglob('*')):
                        if p.is_file(): z.write(p,p.relative_to(OUT))
                for name in ['content_audit.json','page_visual_qa.json','changes_en.diff','changes_zh.diff']:
                    z.write(OUT/name,name)
            for p in sorted(FIG.iterdir()):
                if p.is_file() and (archive == archives[1] or p.suffix in ['.svg','.csv','.json']):
                    z.write(p,'figures/'+p.name)
            for name in shared:z.write(OUT/name,name)
            for name in ['paper_result_binding.json','nnc_evidence_binding.json','citation_audit.json']:
                z.write(OLD/name,'evidence_binding/'+name)
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    # Update the established source only after saving its previous content.
    for relative in ['manuscript.tex']+[f'figures_lowdim_revision/Figure_{i}.pdf' for i in [*range(1,10),'C1']]:
        canonical = SRC/relative
        if canonical.exists():
            backup=hist/'working_source'/relative;backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(canonical,backup)
        shutil.copy2(OUT/'latex_source_en'/relative,canonical)
    manifest = read(OLD/'delivery_manifest.json')
    for label,p in [('selected',OLD/'delivery_manifest.json'),('root',OLD.parent/'delivery_manifest.json')]:
        shutil.copy2(p,hist/(label+'_delivery_manifest.json'))
    manifest.update(updated_utc=now.isoformat(),editorial_revision='Academic vector figures and direct local motion comparison',
        pdf=str(OUT/'manuscript_revised_en.pdf'),pdf_sha256=inv['en']['sha256'],pages=22,
        chinese_pdf=str(OUT/'manuscript_revised_zh.pdf'),chinese_pdf_sha256=inv['zh']['sha256'],chinese_pages=19,
        revision_notes_pdf=str(OUT/'图件优化与非实验修改说明.pdf'),revision_notes_pdf_sha256=sha(OUT/'图件优化与非实验修改说明.pdf'),revision_notes_pages=4,
        revision_notes_markdown=str(OUT/'图件优化与非实验修改说明.md'),revision_notes_addendum=str(OUT/'图件优化与非实验修改说明.md'),
        revision_notes_scope_audit=str(OUT/'content_audit.json'),source_archive=str(archives[0]),source_archive_sha256=sha(archives[0]),
        editable_figure_archive=str(archives[1]),editable_figure_archive_sha256=sha(archives[1]),
        source_directory=str(OUT/'latex_source_en'),chinese_source_directory=str(OUT/'latex_source_zh'),
        visual_qa=str(OUT/'page_visual_qa.json'),content_audit=str(OUT/'content_audit.json'),figure_binding=str(OUT/'final_figure_binding.json'),
        utilization_statistics=str(OUT/'utilization_statistics.json'),remaining_author_action=None,
        remaining_author_action_summary='No mandatory Figure1/3 editing task remains. The requested vector-copy revision is delivered; await new author feedback.',
        next_step='Keep the frozen model and delivered manuscript. No retraining, recompilation or repeated unchanged-data checks without new evidence or user instructions.',
        author_original_figure_label_task_complete=True,figure_editorial_revision_complete=True,
        manuscript_review_ready=True,final_manuscript_accepted=False,supporting_figure_count=1,
        new_training_updates_this_delivery=0,new_simulation_runs_this_delivery=0,
        prior_delivery_manifest_history=str(hist/'selected_delivery_manifest.json'),latest_version_manifest=str(OUT/'delivery_manifest.json'))
    for p in [OUT/'delivery_manifest.json',OLD/'delivery_manifest.json',OLD.parent/'delivery_manifest.json']:write(p,manifest)
    summary = ('Academic figure revision delivered: English22pages, Chinese19pages, nine main figures, one supporting complete-motion figure, '
        'three main result tables and four-page revision notes. Original Figure1/3 vector copies revised with current user authorization; '
        'Figure2 unchanged. No mandatory author figure-label task remains. Selected seed923202 model and all research evidence unchanged; '
        'no training or simulation. Three selected student histories are not uniform full-pipeline independent repeats; paired action-only 5% remains unproven.')
    paths=[BASE/'当前执行状态.json',ROOT/'current_progress.json',ROOT.parent/'live_research_summary.json',
           ROOT.parent/'all34_dominance_research/active_research_checkpoint.json']
    for i,p in enumerate(paths):
        state=read(p);shutil.copy2(p,hist/f'state_{i}_{p.name}')
        state.update(updated_utc=now.isoformat(),current=summary,active_stages=[],active_offline_audit=None,
            verified_active_exec_session=None,method_frozen=True,manuscript_review_ready=True,final_manuscript_accepted=False,
            figure_editorial_revision_complete=True,author_original_figure_label_task_complete=True,
            next_step=manifest['next_step'],remaining_author_action=manifest['remaining_author_action_summary'],
            final_pdf=manifest['pdf'],review_pdf=manifest['pdf'],review_pdf_sha256=manifest['pdf_sha256'],manuscript_pages=22,
            chinese_review_pdf=manifest['chinese_pdf'],chinese_review_pdf_sha256=manifest['chinese_pdf_sha256'],chinese_manuscript_pages=19,
            revision_notes_pdf=manifest['revision_notes_pdf'],manuscript_source_archive=manifest['source_archive'],
            manuscript_delivery_manifest=str(OUT/'delivery_manifest.json'),manuscript_delivery_manifest_sha256=sha(OUT/'delivery_manifest.json'))
        write(p,state)
    recorder=ROOT/'record_current_supplement.py'
    t=recorder.read_text(encoding='utf-8')
    t=t.replace('Original Figure1/3 labels await author updates; no training is active.',
                'Figure editorial status and remaining author actions are read from the current delivery manifest; no training is active.')
    old="next_step='Author updates original Figure1/3 labels per the delivered notes; then update work copies, compile and inspect affected pages. Preserve original Figure2, all historical files and the selected controller. No retraining or repeated unchanged-data checks without new evidence or instructions.',"
    assert old in t
    t=t.replace(old,"next_step=delivery.get('next_step', delivery.get('remaining_author_action_summary', 'Await new author instructions; preserve the frozen method.')),")
    recorder.write_text(t,encoding='utf-8')
    with (ROOT/'continuation_round49_20260923.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+now.isoformat()+' ACADEMIC FIGURE REVISION DELIVERED\n'+summary+'\n'
            'Two-step planner replaces S2 in publication display; J-NNC first in comparison legends/tables. Figure6 uses same C02 common-axis V/A/J and quantified loss; '
            '86.1%/36.7% are local integrated-loss reductions, not total time. Figure7 uses enlarged prescribed corners; full motion in AppendixC.1. '
            'All45pages (22EN,19ZH,4notes) visually reviewed or pixel-matched to inspected pages; 24references and all equations unchanged. '
            'Source and editable-figure archives verified. Working source synchronized after backup; prior delivery intact. Latest manifests/status updated. '
            'Old author-only Figure1/3 restriction superseded by latest explicit PDF-edit authorization. No training, simulation, branch switch, deletion, commit, push or subagent.\n'
            'Latest delivery: '+str(OUT/'delivery_manifest.json')+'\n'+manifest['next_step']+'\n')
    print(json.dumps(dict(pdf=manifest['pdf'],chinese_pdf=manifest['chinese_pdf'],pages=[22,19],archives=[str(p) for p in archives],history=str(hist)),ensure_ascii=False))


if __name__ == '__main__':
    main()

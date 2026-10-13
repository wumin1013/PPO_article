"""Verify the two authorized manuscript edits, then package a preserved new revision."""
import sys,re,json,shutil,zipfile
from datetime import datetime,timezone
from PIL import Image
from pathlib import Path
from appendix_action_revision import ROOT,BASE,OLD,PREV,OUT,WORK,FIG,read,write,sha,source_text
from package_academic_revision import pixel_sha


def inspect():
    inv=read(OUT/'render_inventory_final.json');oldqa=read(PREV/'page_visual_qa.json')['documents']
    locations={};qa={};changed={};sources={}
    for lang,name in [('en','manuscript'),('zh','manuscript_zh')]:
        info=inv[lang];assert not info['compile_issues']
        assert sha(info['pdf'])==info['sha256']
        verified={r['pixel_sha256']:r for r in oldqa[lang]['pages_reviewed']}
        rows=[];changed[lang]=[]
        for i,p in enumerate(info['renders'],1):
            digest=pixel_sha(p);prior=verified.get(digest)
            rows.append(dict(page=i,render=p,pixel_sha256=digest,review_reference=prior['render'] if prior else None,
                             pixel_identical_to_previous_reviewed_page=bool(prior)))
            if prior is None:changed[lang].append(i)
        qa[lang]=dict(pages=info['pages'],pdf_sha256=info['sha256'],pages_reviewed=rows,compile_issues=[])
        directory=OUT/f'latex_source_{lang}'
        current=(directory/(name+'.tex')).read_text(encoding='utf-8')
        old=(PREV/f'latex_source_{lang}'/(name+'.tex')).read_text(encoding='utf-8')
        expected,block=source_text(old,lang)
        assert current==expected, 'Unexpected change beyond the two authorized edits'
        main,appendix=current.split('\n\\appendix\n',1)
        assert main.count('\\begin{figure}')==9 and main.count('\\begin{table}')==3
        assert appendix.count('\\begin{figure}')==2
        for p in (PREV/f'latex_source_{lang}/figures_lowdim_revision').glob('*.pdf'):
            assert sha(directory/'figures_lowdim_revision'/p.name)==sha(p)
        assert sha(directory/'figures_lowdim_revision/Figure_B1.pdf')==sha(FIG/'Figure_B1.pdf')
        assert sha(directory/'references.bib')==sha(PREV/f'latex_source_{lang}/references.bib')
        bbl=directory/(name+'.bbl')
        assert len(re.findall(r'\\bibitem\{([^}]+)',bbl.read_text(encoding='utf-8')))==24
        for p in directory.rglob('*'):
            if p.is_file():sources[str(p)]=sha(p)
        aux=Path(info['pdf']).with_suffix('.aux').read_text(encoding='utf-8')
        locations[lang]={key:dict(number=num,page=int(page)) for key,num,page in re.findall(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
        assert locations[lang]['fig:action_correspondence']['number']=='B.1'
        assert locations[lang]['fig:complete_support']['number']=='C.1'
    for p in (PREV/'figures').iterdir():
        if p.is_file():assert sha(p)==sha(FIG/p.name)
    binding=read(OUT/'appendix_action_binding.json')
    for field in ['trace','metrics','geometry','reference']:
        assert sha(binding[field])==binding[field+'_sha256']
    for name,digest in {**binding['outputs'],**binding['csv_outputs']}.items():assert sha(FIG/name)==digest
    write(OUT/'figure_table_locations.json',locations)
    write(OUT/'pending_page_qa.json',dict(documents=qa,changed_pages=changed))
    write(OUT/'content_audit.json',dict(scope='Exactly two requested edits: contour-deviation header/footnote and Appendix B.1 figure/caption.',
        exact_bilingual_expected_diff=True,main_figures=9,supporting_figures=2,main_result_tables=3,
        previous_figures_all_byte_identical=True,all_previous_result_values_unchanged=True,
        methodology_and_equations_unchanged=True,reference_list_24_unchanged=True,source_sha256=sources,
        no_training_or_simulation=True,figure_table_locations=locations))
    print(json.dumps(dict(pages={k:v['pages'] for k,v in inv.items()},changed_pages=changed,locations=locations),ensure_ascii=False))


def finish():
    inv=read(OUT/'render_inventory_final.json');qa=read(OUT/'pending_page_qa.json')
    reviewed=read(WORK/'reviewed_changed_pages.json')
    for lang,info in qa['documents'].items():
        assert sorted(reviewed[lang])==sorted(qa['changed_pages'][lang])
        for r in info['pages_reviewed']:
            assert pixel_sha(r['render'])==r['pixel_sha256']
            r.update(layout_pass=True)
            if not r['review_reference']:r['review_reference']=r['render']
    qa.update(layout_pass=True,scope='Changed pages visually inspected; unchanged pages matched to prior inspected render by RGB pixel hash.')
    write(OUT/'page_visual_qa.json',qa)
    now=datetime.now(timezone.utc);hist=WORK/'delivery_history'/now.strftime('%Y%m%dT%H%M%S%fZ');hist.mkdir(parents=True)
    for lang in ['en','zh']:
        shutil.copy2(inv[lang]['pdf'],OUT/f'manuscript_revised_{lang}.pdf')
    manifest=read(PREV/'delivery_manifest.json')
    notes=f'''# 本轮两处定点修改

1. 结果表1的“最大距离（mm）”改为“最大轮廓偏差（mm）”；英文为Maximum contour deviation (mm)，表下注释同步术语。所有表格数值不变。
2. 附录B新增图B.1：当前冻结J-NNC模型在既定90°/C02角点的三项动作请求与实际航向、速度、曲率对应。所有子图共用时间轴，动作按实际20 ms更新、执行量由1 ms位置记录重建，无信号平滑。与正文图6使用同一角点、同一模型和同一执行记录。

正文九幅图、研究方法、结果数值、标题、章节逻辑及其余文字保持；附录原完整执行图C.1保留。上一轮图件说明作为历史说明保留，本次只有以上增补。

英文{inv['en']['pages']}页，中文{inv['zh']['pages']}页。图B.1位于英文第{read(OUT/'figure_table_locations.json')['en']['fig:action_correspondence']['page']}页、中文第{read(OUT/'figure_table_locations.json')['zh']['fig:action_correspondence']['page']}页。全部变动页已视觉检查，其余页与先前已检查页面进行像素核对。

未重新训练或仿真；历史稿件、模型和数据保留。图B.1提供PDF/SVG/PNG和两份CSV，来源绑定见appendix_action_binding.json。
'''
    (OUT/'本轮两处修改说明.md').write_text(notes,encoding='utf-8')
    (OUT/'README.md').write_text(notes+'\n编译：TeX Live + XeLaTeX + BibTeX。中英文源码分别位于latex_source_en和latex_source_zh，文本为UTF-8无BOM。\n'
        '图1/3完整修订图为PDF；editable_labels.svg仅为新增标签图层。其余数据图提供完整SVG。\n',encoding='utf-8')
    previous_binding=read(PREV/'final_figure_binding.json')
    previous_binding.update(main_figure_count=9,supporting_figure_count=2)
    for name,entry in previous_binding['figures'].items():
        entry['source']=str(FIG/name)
    previous_binding['figures']['Figure_B1.pdf']=dict(source=str(FIG/'Figure_B1.pdf'),sha256=sha(FIG/'Figure_B1.pdf'),
        data_binding=str(OUT/'appendix_action_binding.json'))
    write(OUT/'final_figure_binding.json',previous_binding)
    archives=[]
    for name in ['latex_sources_en_zh.zip','figures_editable.zip']:
        archive=OUT/name;assert not archive.exists()
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            if name.startswith('latex'):
                for lang in ['en','zh']:
                    for p in sorted((OUT/f'latex_source_{lang}').rglob('*')):
                        if p.is_file():z.write(p,p.relative_to(OUT))
                for audit in ['content_audit.json','page_visual_qa.json','changes_en.diff','changes_zh.diff']:
                    z.write(OUT/audit,audit)
            for p in FIG.iterdir():
                if p.is_file() and (name.startswith('figures') or p.suffix in ['.svg','.csv','.json']):z.write(p,'figures/'+p.name)
            for f in ['README.md','本轮两处修改说明.md','appendix_action_binding.json','appendix_figure_contract.json','final_figure_binding.json']:
                z.write(OUT/f,f)
            for f in ['figure_data_binding.json','original_vector_edit_audit.json','utilization_statistics.json','utilization_statistics.csv','图件优化与非实验修改说明.pdf']:
                z.write(PREV/f,'prior_revision_binding/'+f)
            for f in ['paper_result_binding.json','nnc_evidence_binding.json','citation_audit.json']:z.write(OLD/f,'evidence_binding/'+f)
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
        archives.append(archive)
    src=Path('D:/PPO_article_work/论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous')
    for rel in ['manuscript.tex','figures_lowdim_revision/Figure_B1.pdf']:
        dest=src/rel
        if dest.exists():
            bk=hist/'working_source'/rel;bk.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,bk)
        shutil.copy2(OUT/'latex_source_en'/rel,dest)
    for label,p in [('selected',OLD/'delivery_manifest.json'),('root',OLD.parent/'delivery_manifest.json')]:
        shutil.copy2(p,hist/(label+'_delivery_manifest.json'))
    manifest.update(updated_utc=now.isoformat(),editorial_revision='Compact Appendix B.1 and maximum contour deviation table label',
        pdf=str(OUT/'manuscript_revised_en.pdf'),pdf_sha256=inv['en']['sha256'],pages=inv['en']['pages'],
        chinese_pdf=str(OUT/'manuscript_revised_zh.pdf'),chinese_pdf_sha256=inv['zh']['sha256'],chinese_pages=inv['zh']['pages'],
        source_archive=str(archives[0]),source_archive_sha256=sha(archives[0]),editable_figure_archive=str(archives[1]),editable_figure_archive_sha256=sha(archives[1]),
        source_directory=str(OUT/'latex_source_en'),chinese_source_directory=str(OUT/'latex_source_zh'),
        revision_notes_addendum=str(OUT/'本轮两处修改说明.md'),visual_qa=str(OUT/'page_visual_qa.json'),content_audit=str(OUT/'content_audit.json'),
        figure_binding=str(OUT/'final_figure_binding.json'),supporting_figure_count=2,
        appendix_action_binding=str(OUT/'appendix_action_binding.json'),latest_version_manifest=str(OUT/'delivery_manifest.json'),
        prior_delivery_manifest_history=str(hist/'selected_delivery_manifest.json'),manuscript_review_ready=True,
        remaining_author_action=None,remaining_author_action_summary='The two requested final edits are delivered. No mandatory figure task remains.',
        next_step='Await new author instructions. Preserve this revision and the frozen method; no further edits, training, simulation or repeated checks without new evidence or user request.')
    for p in [OUT/'delivery_manifest.json',OLD/'delivery_manifest.json',OLD.parent/'delivery_manifest.json']:write(p,manifest)
    summary=('User requested only a compact action-correspondence appendix figure and clearer maximum contour deviation table label. '
        'Both bilingual edits delivered. Figure B.1 uses recorded seed923202 C02 requests and XY-derived motion; '
        'all nine main figures, numeric results, method and other wording unchanged. No research reopened; no training/simulation. Await new user instructions.')
    for i,p in enumerate([BASE/'当前执行状态.json',ROOT/'current_progress.json',ROOT.parent/'live_research_summary.json',
                         ROOT.parent/'all34_dominance_research/active_research_checkpoint.json']):
        state=read(p);shutil.copy2(p,hist/f'state_{i}_{p.name}')
        state.update(updated_utc=now.isoformat(),current=summary,next_step=manifest['next_step'],active_stages=[],
            final_pdf=manifest['pdf'],review_pdf=manifest['pdf'],review_pdf_sha256=manifest['pdf_sha256'],manuscript_pages=manifest['pages'],
            chinese_review_pdf=manifest['chinese_pdf'],chinese_review_pdf_sha256=manifest['chinese_pdf_sha256'],chinese_manuscript_pages=manifest['chinese_pages'],
            manuscript_source_archive=manifest['source_archive'],manuscript_delivery_manifest=str(OUT/'delivery_manifest.json'),
            manuscript_delivery_manifest_sha256=sha(OUT/'delivery_manifest.json'),remaining_author_action=manifest['remaining_author_action_summary'],
            revision_notes_addendum=manifest['revision_notes_addendum'])
        write(p,state)
    with (ROOT/'continuation_round49_20260923.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+now.isoformat()+' TWO REQUESTED FINAL EDITS DELIVERED\n'+summary+'\n'
            f"English {manifest['pages']}pages / Chinese {manifest['chinese_pages']}pages. Main tables/figures and all prior results unchanged. "
            'Preserve current research-evidence scope. No branch switch, cleanup, deletion, commit, push or subagent.\n'
            'Manifest: '+str(OUT/'delivery_manifest.json')+'\n')
    print(json.dumps(dict(pdf=manifest['pdf'],chinese_pdf=manifest['chinese_pdf'],pages=[manifest['pages'],manifest['chinese_pages']],
        appendix_figure=str(FIG/'Figure_B1.pdf'),source_archive=manifest['source_archive']),ensure_ascii=False))


if __name__=='__main__':
    if sys.argv[-1]=='finish':finish()
    else:inspect()

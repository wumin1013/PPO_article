"""Audit and deliver the bilingual NNC/utilization editorial revision."""
import json, re, shutil, zipfile, difflib, csv
from pathlib import Path
from datetime import datetime, timezone
from revise_nnc_utilization_figures import ROOT, OLD, OUT, FIG, sha, read, write

WORK=ROOT/'lowdim_original_manuscript_work/figure_revision_20260924'
BASE=Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1')
SRC=Path('D:/PPO_article_work/论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous')

def main():
    now=datetime.now(timezone.utc)
    hist=WORK/'delivery_history'/now.strftime('%Y%m%dT%H%M%S%fZ');hist.mkdir(parents=True)
    inv=read(OUT/'render_inventory_final.json');prior=read(OUT/'render_inventory.json')
    assert inv['en']['pages']==21 and inv['zh']['pages']==18
    pageqa={};sources={};locations={};texts={}
    for lang,name in [('en','manuscript'),('zh','manuscript_zh')]:
        info=inv[lang];assert not info['compile_issues']
        verified={sha(p):p for p in prior[lang]['renders']}
        changed=19 if lang=='en' else 16
        verified[sha(info['renders'][changed-1])]=info['renders'][changed-1]
        rows=[]
        for i,p in enumerate(info['renders'],1):
            assert sha(p) in verified
            rows.append(dict(page=i,render=p,sha256=sha(p),review_reference=verified[sha(p)],layout_pass=True))
        pageqa[lang]=dict(pages=info['pages'],pdf_sha256=info['sha256'],pages_reviewed=rows,compile_issues=[])
        dest=OUT/f'latex_source_{lang}'
        text=(dest/(name+'.tex')).read_text(encoding='utf-8');texts[lang]=text
        old=(OLD/f'latex_source_{lang}'/(name+'.tex')).read_text(encoding='utf-8')
        start='\\section{'+('Methodology' if lang=='en' else '方法')+'}'
        end='\\section{'+('Experiments and Results' if lang=='en' else '实验与结果')+'}'
        assert old.split(start)[1].split(end)[0]==text.split(start)[1].split(end)[0]
        assert re.findall(r'\\title\{([^}]+)',old)==re.findall(r'\\title\{([^}]+)',text)
        assert text.count('\\begin{figure}')==9
        main=text.split('\n\\appendix\n')[0]
        assert main.count('\\begin{table}')==3 and main.count('0.5 ms')==1
        assert not re.search(r'exploratory\s+(?:process|history)|failure\s+repair|round[- ]by[- ]round|evidence\s+directory|探索过程|失败修复|逐轮日志|证据目录',text,re.I)
        for i in range(1,4):assert sha(dest/'figures_lowdim_revision'/f'Figure_{i}.pdf')==sha(OLD/f'latex_source_{lang}/figures_lowdim_revision'/f'Figure_{i}.pdf')
        for i in range(1,10):assert sha(dest/'figures_lowdim_revision'/f'Figure_{i}.pdf')==sha(FIG/f'Figure_{i}.pdf')
        bbl=Path(info['pdf']).with_suffix('.bbl')
        if not bbl.exists():bbl=dest/(name+'.bbl')
        keys=re.findall(r'\\bibitem\{([^}]+)',bbl.read_text(encoding='utf-8'))
        assert len(keys)==24 and keys==re.findall(r'\\bibitem\{([^}]+)',(OLD/f'latex_source_{lang}'/(name+'.bbl')).read_text(encoding='utf-8'))
        assert sha(dest/'references.bib')==sha(OLD/f'latex_source_{lang}/references.bib')
        if bbl!=dest/(name+'.bbl'):shutil.copy2(bbl,dest/(name+'.bbl'))
        for p in dest.rglob('*'):
            if p.is_file():sources[str(p)]=sha(p)
        shutil.copy2(info['pdf'],OUT/f'manuscript_revised_{lang}.pdf')
        aux=Path(info['pdf']).with_suffix('.aux').read_text(encoding='utf-8')
        locations[lang]={key:dict(number=num,page=int(page)) for key,num,page in re.findall(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
        difference=''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='previous_delivery/'+name+'.tex',tofile='figure_revision/'+name+'.tex'))
        (OUT/f'changes_{lang}.diff').write_text(difference,encoding='utf-8')
    eq=lambda s:re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',s,re.S)
    assert eq(texts['en'])==eq(texts['zh'])
    labels=lambda s:re.findall(r'\\label\{([^}]+)',s)
    assert labels(texts['en'])==labels(texts['zh'])
    stats=read(OUT/'utilization_statistics.json')
    for r in stats['records']:
        for key in ['mean_V_percent','p95_A_percent','p95_J_percent']:
            assert f'{r[key]:.3f}' in texts['en'] and f'{r[key]:.3f}' in texts['zh']
    with (OUT/'utilization_statistics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','method','mean_V_percent','p95_A_percent','p95_J_percent']);writer.writeheader()
        for r in stats['records']:writer.writerow({k:r[k] for k in writer.fieldnames})
    write(OUT/'page_visual_qa.json',dict(created_utc=now.isoformat(),documents=pageqa,layout_pass=True,
        scope='All pages visually reviewed in initial contacts; final pages are pixel-identical except EN19/ZH16, both separately inspected. Original Figure1/3 labels remain author tasks.'))
    write(OUT/'content_audit.json',dict(created_utc=now.isoformat(),main_figures=9,main_result_tables=3,
        source_sha256=sources,equation_alignment_count=len(eq(texts['en'])),bilingual_labels_match=True,
        methodology_unchanged=True,titles_unchanged=True,original_figures_1_to_3_unchanged=True,
        citations_24_unchanged=True,citation_identity_audit=str(OLD/'citation_audit.json'),
        main_half_ms_mentions=1,exploration_logs_in_manuscript=False,figure_table_locations=locations,
        utilization_scope=stats['scope'],all_utilization_table_values_match_bound_statistics=True,
        no_new_training_or_simulation=True))
    oldmap=read(OLD/'final_figure_binding.json')['figures']
    figuremap={}
    for i in range(1,10):
        f=FIG/f'Figure_{i}.pdf'
        if i in [1,2,3,4,8,9]:
            original={8:7,9:8}.get(i,i)
            previous=oldmap[f'Figure_{original}.pdf']
            assert sha(f)==previous['sha256']
            origin=dict(kind='unchanged or renumbered prior figure',previous_number=original,previous_binding=previous)
        else:origin=dict(kind='new plot from bound recorded data',binding=str(OUT/'figure_data_binding.json'),binding_sha256=sha(OUT/'figure_data_binding.json'))
        figuremap[f.name]=dict(number=i,source=str(f),sha256=sha(f),origin=origin)
    write(OUT/'final_figure_binding.json',dict(figures=figuremap,main_figure_count=9,figure4_continued=False,
        protected_originals_unchanged=True,main_model_seed=923202,main_model_sha256=read(OLD/'paper_result_binding.json')['selected_checkpoint_sha256']))
    descriptions={1:'原方法框架',2:'原允差带示意',3:'原执行框架',4:'长路径案例及34角覆盖',5:'NNC向量A/J越限',6:'完整长路径V/A/J利用率及停稳',7:'三类典型角轨迹与局部V/A/J利用率',8:'全部34角速度亏损与全程巡航',9:'密集案例与两动作消融'}
    labels_by_number={int(v['number']):k for k,v in locations['en'].items() if k.startswith('fig:')}
    notes='''# 本轮图表调整说明

本轮交付英文21页、中文18页。沿用恢复的18页原稿格式、标题和章节框架；模型、物理约束、全部路径任务和原图1—3均未改动。原交付稿和旧图保留。

## 新图号与页码

| 新图号 | 内容 | 英文页 | 中文页 |
|---|---|---:|---:|
'''
    for i,desc in descriptions.items():
        key=labels_by_number[i]
        notes+=f'| {i} | {desc} | {locations["en"][key]["page"]} | {locations["zh"][key]["page"]} |\n'
    notes+='''
原图4(a—c)保留为新图4；原图4(d—g)独立为新图6，并改为按上限归一化的V/A/J曲线。新图5用真实记录呈现NNC约束失效。原图5重绘为新图7，轨迹下方增加同一角点的V/A/J曲线。原图7、8分别改为新图8、9。原图6的大幅动作图、原图9的重复效率图不再进入正文。

NNC图显示两个案例相同的前80 ms窗口；正文所列最大越限倍数统计保留的完整前1 s，因此与图中80 ms峰值标注不必相同。失败起始段不用于完整时间或全路径极值比较。

## 利用率表与统计口径

新表2放在英文第11页、中文第9页的实验正文。三项为平均进给利用率、加速度利用率95分位数、加加速度利用率95分位数，均以对应向量上限的百分比表示。统计覆盖1 ms下全部等间隔执行样本，包括启动、制动和20 ms停稳，不移除长直线或不利区段。

长路径的A/J分位数较低，是因为巡航占据大部分时间；图7补充了角点局部利用情况。较高进给利用率说明更少的速度损失；A/J利用率用于解释运动能力的使用，不能把每项利用率都越高越好作为结论。NNC没有完整可行记录，因此不与完整成功方法混算利用率表。

完整效率表为正文表1，三个选定模型的结果为正文表3。主文以1 ms为统一口径，0.5 ms仅在实验设置中保留一句索引，数值敏感性检查集中到附录C。

## 实验之外的变动

方法章节逐字保持上一交付版；本轮未改状态、动作、KCM、奖励或学习配置。摘要、贡献条目和结论仅撤去反复强调0.5 ms的表述，并将对应图件的“动作—运动记录”改为“轨迹与V/A/J利用率”。新增附录C整理原有数值复核，不产生新研究结果。参考文献24项保持不变。

## 原前三图待办

原图1、3仍由作者按此前8页详细指南修改旧标签；原图2保持。详细指南中的标签替换与连线建议继续有效，但其旧图4—9安排和旧页码由本说明替代。当前原图1在英文第3页、中文第2页；原图3在英文第6页、中文第5页。收到作者新图后再更新工作副本、编译并检查对应页面。

本说明不进入论文正文或附录。论文没有探索过程、失败修复或逐轮日志。
'''
    (OUT/'本轮图表调整说明.md').write_text(notes,encoding='utf-8')
    (OUT/'README.md').write_text('''# 中英文图表修订稿

- manuscript_revised_en.pdf：英文21页。
- manuscript_revised_zh.pdf：中文18页，正文与表格同步；图内英文及参考文献保留。
- 本轮图表调整说明.md：图号、页码、统计口径及实验之外的改动。
- latex_source_en、latex_source_zh：与PDF绑定的可编译源码。
- figures：新图PDF、SVG、PNG及绘图CSV；数据来源见JSON索引。

原图1—3保留。原图1、3仍待作者按既有详细指南修改标签，随后完成图文最终核验。本轮未训练或仿真，历史版本保持。

编译：TeX Live + XeLaTeX + BibTeX；从对应源码目录执行latexmk -xelatex manuscript.tex（英文）或latexmk -xelatex manuscript_zh.tex（中文）。所有自有文本为UTF-8无BOM。
''',encoding='utf-8')
    archive=OUT/'latex_sources_en_zh.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for lang in ['en','zh']:
            for p in sorted((OUT/f'latex_source_{lang}').rglob('*')):
                if p.is_file():z.write(p,p.relative_to(OUT))
        for name in ['README.md','本轮图表调整说明.md','changes_en.diff','changes_zh.diff','content_audit.json','page_visual_qa.json','final_figure_binding.json','figure_data_binding.json','figure_contract.json','new_figure_metrics.json','utilization_statistics.json','utilization_statistics.csv']:
            z.write(OUT/name,name)
        for name in ['paper_result_binding.json','nnc_evidence_binding.json','citation_audit.json']:
            z.write(OLD/name,name)
        for p in FIG.iterdir():
            if p.suffix in ['.svg','.csv']:z.write(p,'editable_figures/'+p.name)
        z.write(OLD/'图件修改建议与实验外修改说明_详细版.md','prior_author_guide/图件修改建议与实验外修改说明_详细版.md')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    # Keep the established working source aligned, with a complete pre-edit backup.
    canonical_history=hist/'working_source';canonical_history.mkdir()
    for relative in ['manuscript.tex']+[f'figures_lowdim_revision/Figure_{i}.pdf' for i in range(1,10)]:
        old=SRC/relative;dst=canonical_history/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old,dst)
        shutil.copy2(OUT/'latex_source_en'/relative,old)
    old_manifest=read(OLD/'delivery_manifest.json')
    for label,p in [('selected',OLD/'delivery_manifest.json'),('root',OLD.parent/'delivery_manifest.json')]:shutil.copy2(p,hist/(label+'_delivery_manifest.json'))
    manifest=dict(old_manifest)
    manifest.update(updated_utc=now.isoformat(),editorial_revision='NNC failure and full/local kinematic utilization',
        pdf=str(OUT/'manuscript_revised_en.pdf'),pdf_sha256=sha(OUT/'manuscript_revised_en.pdf'),pages=21,
        chinese_pdf=str(OUT/'manuscript_revised_zh.pdf'),chinese_pdf_sha256=sha(OUT/'manuscript_revised_zh.pdf'),chinese_pages=18,
        source_archive=str(archive),source_archive_sha256=sha(archive),source_directory=str(OUT/'latex_source_en'),chinese_source_directory=str(OUT/'latex_source_zh'),
        visual_qa=str(OUT/'page_visual_qa.json'),content_audit=str(OUT/'content_audit.json'),figure_binding=str(OUT/'final_figure_binding.json'),
        revision_notes_addendum=str(OUT/'本轮图表调整说明.md'),remaining_author_action_summary='Author updates original Figure1/3 labels using the eight-page detailed guide; use the current figure/table addendum for revised figure numbers and pages. Then update work copies, compile and inspect affected pages.',
        main_result_table_count=3,main_figure_count=9,figure4_continued=False,main_interval_ms=1,secondary_resolution_location='Appendix C',
        utilization_statistics=str(OUT/'utilization_statistics.json'),author_guide_only_update=False,
        new_training_updates_this_delivery=0,new_nnc_diagnostic_cases_this_delivery=0,new_simulation_runs_this_delivery=0,
        prior_delivery_manifest_history=str(hist/'selected_delivery_manifest.json'),latest_version_manifest=str(OUT/'delivery_manifest.json'),
        final_manuscript_accepted=False,manuscript_review_ready=True)
    write(OUT/'delivery_manifest.json',manifest);write(OLD/'delivery_manifest.json',manifest);write(OLD.parent/'delivery_manifest.json',manifest)
    with (ROOT/'continuation_round49_20260923.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {now.isoformat()} NNC AND UTILIZATION FIGURE REVISION DELIVERED\n'
            'User requested NNC failure figure, less1ms/0.5ms repetition, replacement of oldFigure6/9, standalone old4d-f and original-style V/A/J utilization curves; then explicitly requested utilization table. New English21page/Chinese18page revision delivered under selected923202_revision/figure_revision_20260924. Ninefigures: protected1--3,case4,NNC5,fullutilization6,threecornerXY+VAJ7,all34cruise8,denseablation9. Three main result tables include full-record utilization(meanV,p95A,p95J). No newtraining/simulation; rawrecords checked and XY-derived data reused. Maintext hasone0.5msmention; supportingnumerics inAppendixC. Method unchanged; minimalabstract/contribution/conclusion alignment. All39pages visually inspected; finalchangedEN19/ZH16rechecked.24referencesunchanged.\n'
            'OriginalworkingENsource synchronized afterbackup; previousdelivery PDFs/sources preserved. Bothdeliverypointers and perrevisionmanifest updated; remainingauthorFigure1/3labeltaskunchanged. Figureguideaddendum provides currentnumbers/pages and overridesoldguidepart5/pageindex only. No branchchange,delete,commit,push,subagent.\n')
    print(json.dumps(dict(pdf=manifest['pdf'],zh=manifest['chinese_pdf'],pages=[21,18],source_archive=str(archive),locations=locations),ensure_ascii=False))

if __name__=='__main__':main()

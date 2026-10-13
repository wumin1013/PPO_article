"""Package the reviewed English/Chinese revision and retain all historical files."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re, shutil, zipfile, difflib
from PIL import Image

ROOT=Path(r'D:\PPO_article_joint_runs\natural_closed_resume\compact_dense_research')
WORK=ROOT/'lowdim_original_manuscript_work'
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
DELIVERY=BASE/'paper_evidence_review/verified_results/lowdim_original_revision'
NEW=DELIVERY/'selected923202_revision'
SRC=Path(r'D:\PPO_article_work\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous')
SUP=ROOT/'experimental_supplement_20260923'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    now=datetime.now(timezone.utc)
    history=WORK/'delivery_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
    history.mkdir(parents=True)
    def preserve(p):
        if Path(p).exists():shutil.copy2(p,history/Path(p).name)
    inv=read(WORK/'final_render_inventory.json')
    assert inv['en']['pages']==22 and inv['zh']['pages']==18 and inv['notes']['pages']==3
    # All v3 pages were visually inspected. Later changed pages were inspected separately.
    reviewed=[*sorted((WORK/'final_visual_en').glob('page-*.png')),
              *sorted((WORK/'final_visual_zh').glob('page-*.png')),
              *[WORK/f'final_visual_v4_en/page-{i:02d}.png' for i in [21,22]],
              WORK/'final_visual_v4_zh/page-01.png',
              *[WORK/f'final_visual_v6_zh/page-{i:02d}.png' for i in [17,18]],
              *sorted((WORK/'final_visual_v6_notes').glob('page-*.png'))]
    inspected={sha(p):str(p) for p in reviewed}
    qa={}
    for key,info in inv.items():
        rows=[]
        for i,p in enumerate(info['renders'],1):
            h=sha(p)
            assert h in inspected,(key,i,'not visually reviewed or identical to a reviewed page')
            rows.append(dict(page=i,render=p,render_sha256=h,visual_review_reference=inspected[h],layout_pass=True))
        logfile=Path(info['pdf']).with_suffix('.log')
        log=logfile.read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'Overfull|undefined references|undefined citations|Missing character:',log,re.I)
        qa[key]=dict(pdf=info['pdf'],pdf_sha256=sha(info['pdf']),pages=info['pages'],pages_reviewed=rows,compile_log=str(logfile),compile_log_sha256=sha(logfile))
    write(NEW/'page_visual_qa.json',dict(created_utc=now.isoformat(),documents=qa,scope='All final pages visually inspected, or pixel-identical to inspected pages; changed pages rechecked. Original Figure1/3 label mismatch remains an author task.',layout_pass=True,final_manuscript_accepted=False))
    destinations={'en':'manuscript_revised_en.pdf','zh':'manuscript_revised_zh.pdf','notes':'图件修改建议与实验外修改说明.pdf'}
    for key,name in destinations.items():
        preserve(NEW/name)
        shutil.copy2(inv[key]['pdf'],NEW/name)
    en=(SRC/'manuscript.tex').read_text(encoding='utf-8')
    origin=read(DELIVERY/'revision_origin.json')
    original=Path(origin['base_source']).read_text(encoding='utf-8')
    def diff(a,b):return ''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='restored_18page/manuscript.tex',tofile='selected923202_revision/manuscript.tex'))
    (NEW/'original_to_revised.diff').write_text(diff(original,en),encoding='utf-8')
    omit=lambda s:re.sub(r'\\section\{Experiments and Results\}.*?(?=\\section\{Conclusion\})','% Experimental section omitted from this focused comparison.\n',s,flags=re.S)
    assert omit(original)!=original and omit(en)!=en
    (NEW/'non_experimental_changes.diff').write_text(diff(omit(original),omit(en)),encoding='utf-8')
    shutil.copy2(DELIVERY/'revision_origin.json',NEW/'base_manuscript_origin.json')
    figure_map={}
    figures=re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)',en)
    for f in figures:
        k=int(re.search(r'Figure_(\d+)',f).group(1))
        folder=SRC/'figures_lowdim_revision' if k<4 else NEW/('dense_ablation' if k==8 else 'student_efficiency' if k==9 else 'figures_publication')
        assert sha(folder/f)==sha(SRC/'figures_lowdim_revision'/f)
        source_binding=DELIVERY/'revision_origin.json' if k<4 else NEW/('dense_ablation/Figure_8_scope.json' if k==8 else 'student_efficiency/figure_data_binding.json' if k==9 else 'figures_publication/figure_data_binding.json')
        figure_map[f]=dict(number=k,source=str(folder/f),sha256=sha(folder/f),evidence_binding=str(source_binding),evidence_binding_sha256=sha(source_binding),main_result_binding=str(NEW/'paper_result_binding.json') if k>=4 else None)
    write(NEW/'final_figure_binding.json',dict(figures=figure_map,protected_originals_unchanged=True,main_model_seed=923202,selected_model_sha256=read(NEW/'paper_result_binding.json')['selected_checkpoint_sha256'],dense_total_speed_deficit_mm=dict(three_action=132.50249970170424,two_action=213.8666335138402),main_figure_count=9,figure4_continued=True))
    # Snapshot exactly the source dependencies of this PDF, without compiler intermediates.
    en_snapshot=NEW/'latex_source_en'
    en_snapshot.mkdir(exist_ok=True)
    (en_snapshot/'figures_lowdim_revision').mkdir(exist_ok=True)
    for name in ['manuscript.tex','references.bib','elsarticle.cls','unsrt.bst']:
        shutil.copy2(SRC/name,en_snapshot/name)
    english_bbl=Path(inv['en']['pdf']).with_suffix('.bbl')
    if not english_bbl.exists():english_bbl=SRC/'manuscript.bbl'
    expected_cites=list(dict.fromkeys(','.join(re.findall(r'\\cite\{([^}]+)',en)).split(',')))
    assert re.findall(r'\\bibitem\{([^}]+)',english_bbl.read_text(encoding='utf-8'))==expected_cites
    shutil.copy2(english_bbl,en_snapshot/'manuscript.bbl')
    for f in figures:shutil.copy2(SRC/'figures_lowdim_revision'/f,en_snapshot/'figures_lowdim_revision'/f)
    zh_snapshot=NEW/'latex_source_zh'
    for name in ['elsarticle.cls','unsrt.bst']:
        shutil.copy2(SRC/name,zh_snapshot/name)
    shutil.copy2(Path(inv['zh']['pdf']).with_suffix('.bbl'),zh_snapshot/'manuscript_zh.bbl')
    readme='''# PPO_article 修订稿交付\n\n本目录以恢复的18页匿名投稿稿为唯一基础，沿用原标题、elsarticle格式及原五章结构。英文22页，中文全文18页；原前三图保留，图4—9和实验正文已更新。\n\n- manuscript_revised_en.pdf：英文修订稿。\n- manuscript_revised_zh.pdf：自然学术中文全文，图内英文及参考文献题名保留。\n- 图件修改建议与实验外修改说明.pdf / .md：原图1、3逐项替换建议，以及实验之外的必要修改。\n- original_to_revised.diff：相对唯一18页源码的完整差异。\n- non_experimental_changes.diff：略去实验章节后的差异。\n- latex_source_en / latex_source_zh：两份PDF对应的可编辑源码。\n- paper_result_binding.json / nnc_evidence_binding.json / final_figure_binding.json：结果、NNC可行性记录和图件的路径与哈希。\n\n英文正文保留两张核心结果表；KCM、策略表示和必要学习参数放在附录。研究日志及失败修复过程未放入论文。三个选定模型的成功结果与同一固定流程的独立训练重复有区别；正文和附录保留必要说明。\n\n当前版本可供审阅。原图1、3仍有旧标签，需作者按修改建议更新后再完成投稿前图文验收。原图2无需修改。所有历史稿件与模型均保留。\n\n## 编译\n\n使用TeX Live、XeLaTeX、BibTeX。英文目录运行 `latexmk -xelatex manuscript.tex`，中文目录运行 `latexmk -xelatex manuscript_zh.tex`。字体采用Windows宋体；中文斜体采用楷体。所有自有文本源文件均为UTF-8无BOM。\n'''
    (NEW/'README.md').write_text(readme,encoding='utf-8')
    archive=NEW/'latex_sources_en_zh.zip'
    preserve(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in [en_snapshot,zh_snapshot]:
            for name in ([('manuscript.tex' if folder==en_snapshot else 'manuscript_zh.tex'),'references.bib','elsarticle.cls','unsrt.bst',('manuscript.bbl' if folder==en_snapshot else 'manuscript_zh.bbl')]):
                z.write(folder/name,(folder.name+'/'+name))
            for f in figures:z.write(folder/'figures_lowdim_revision'/f,folder.name+'/figures_lowdim_revision/'+f)
        for name in ['README.md','图件修改建议与实验外修改说明.md','修改说明.tex','original_to_revised.diff','non_experimental_changes.diff','paper_result_binding.json','nnc_evidence_binding.json','final_figure_binding.json','all34_corner_statistics.csv','content_audit.json','citation_audit.json','translation_alignment.json','page_visual_qa.json','base_manuscript_origin.json']:
            z.write(NEW/name,name)
        for f,info in figure_map.items():
            if info['number']<4:continue
            p=Path(info['source'])
            for ext in ['.svg']:
                if p.with_suffix(ext).exists():z.write(p.with_suffix(ext),'editable_figures/'+p.with_suffix(ext).name)
        for folder in ['figures_publication','dense_ablation','student_efficiency']:
            for p in sorted((NEW/folder).glob('*source_data.csv')):z.write(p,'figure_data/'+folder+'/'+p.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    manifest=dict(created_utc=now.isoformat(),version='selected923202_revision',pdf=str(NEW/destinations['en']),pdf_sha256=sha(NEW/destinations['en']),pages=inv['en']['pages'],chinese_pdf=str(NEW/destinations['zh']),chinese_pdf_sha256=sha(NEW/destinations['zh']),chinese_pages=inv['zh']['pages'],revision_notes_pdf=str(NEW/destinations['notes']),revision_notes_pdf_sha256=sha(NEW/destinations['notes']),revision_notes_pages=inv['notes']['pages'],source_archive=str(archive),source_archive_sha256=sha(archive),source_directory=str(en_snapshot),chinese_source_directory=str(zh_snapshot),original_to_revised_diff=str(NEW/'original_to_revised.diff'),non_experimental_changes_diff=str(NEW/'non_experimental_changes.diff'),evidence_binding=str(NEW/'paper_result_binding.json'),evidence_binding_sha256=sha(NEW/'paper_result_binding.json'),nnc_evidence_binding=str(NEW/'nnc_evidence_binding.json'),visual_qa=str(NEW/'page_visual_qa.json'),content_audit=str(NEW/'content_audit.json'),citation_audit=str(NEW/'citation_audit.json'),figure_binding=str(NEW/'final_figure_binding.json'),selected_checkpoint=read(NEW/'paper_result_binding.json')['selected_checkpoint'],selected_checkpoint_sha256=read(NEW/'paper_result_binding.json')['selected_checkpoint_sha256'],method_frozen=True,research_acceptance_passed=True,manuscript_review_ready=True,final_manuscript_accepted=False,remaining_author_action=str(NEW/'图件修改建议与实验外修改说明.md'),remaining_author_action_summary='Author updates original Figure1/3 labels; then replace work copies, compile and check affected pages. Original Figure2 remains unchanged.',current_case_nnc_comparison_complete=True,three_selected_models_pass=True,uniform_full_pipeline_independent_repetitions=False,paired_action_only_five_percent_demonstrated=False,original_long_case_included=True,original_long_case_reference_length_m=44.420514,original_long_case_corner_count=34,main_figure_count=9,new_training_updates_this_delivery=0,new_nnc_diagnostic_cases_this_delivery=2,prior_delivery_manifest_history=str(history/'delivery_manifest.json'),history_preserved=True)
    preserve(NEW/'delivery_manifest.json')
    write(NEW/'delivery_manifest.json',manifest)
    shutil.copy2(DELIVERY/'delivery_manifest.json',history/'delivery_manifest.json')
    manifest['latest_version_manifest']=str(NEW/'delivery_manifest.json')
    write(DELIVERY/'delivery_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['pdf','pages','chinese_pdf','chinese_pages','revision_notes_pdf','revision_notes_pages','source_archive']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

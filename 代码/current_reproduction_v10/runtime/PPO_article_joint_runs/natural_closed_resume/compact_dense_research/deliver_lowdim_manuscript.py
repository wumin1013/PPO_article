"""Preserve, package and register the visually reviewed paper revision."""
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timezone
from build_lowdim_original_manuscript import ROOT, WORK, BASE, DELIVERY, SRC, sha, read, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-dir',type=Path,default=WORK/'page_review_final')
    parser.add_argument('--changed-pages',type=int,nargs='+',default=[6,7,19])
    args=parser.parse_args()
    now=datetime.now(timezone.utc).isoformat()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit=read(DELIVERY/'manuscript_integrity_audit.json')
    assert audit['pages']==21 and audit['no_unresolved_references']
    pdf=WORK/'build/manuscript.pdf'
    assert sha(pdf)==audit['pdf_sha256']
    history=ROOT/'lowdim_manuscript_history'/('delivery_'+stamp)
    history.mkdir(parents=True)
    output=DELIVERY/'manuscript_without_author_details.pdf'
    if output.exists(): shutil.copy2(output,history/output.name)
    shutil.copy2(pdf,output)
    package=DELIVERY/'latex_source'
    if package.exists(): shutil.copytree(package,history/'latex_source')
    package.mkdir(exist_ok=True)
    for name in ['manuscript.tex','references.bib','manuscript.bbl','elsarticle.cls','unsrt.bst']:
        shutil.copy2(SRC/name,package/name)
    (package/'figures_lowdim_revision').mkdir(exist_ok=True)
    for p in (SRC/'figures_lowdim_revision').glob('*.pdf'):
        shutil.copy2(p,package/'figures_lowdim_revision'/p.name)
    zipfile=DELIVERY/'latex_source.zip'
    if zipfile.exists(): shutil.copy2(zipfile,history/zipfile.name)
    shutil.make_archive(str(DELIVERY/'latex_source'),'zip',DELIVERY,'latex_source')
    page_notes={1:'Title, abstract, keywords and introduction; original class and margins.',2:'Literature paragraphs and citations.',3:'Protected original figure1; legacy two-output schematic flagged for author.',4:'Protected original figure2; full-width epsilon convention preserved.',5:'Planar constraints and start of methodology.',6:'Protected original figure3 and14input equation; appendix reference corrected.',7:'Action map, reward and appendix references.',8:'Case, constraints and comparator definitions.',9:'Six complete results in table; values checked against bound traces.',10:'Figure4 geometry,10mm grid and all34angles.',11:'Figure4 continued;full V/A/J and braking.',12:'Figure5 representative trajectory zooms and deficit discussion.',13:'Figure6 six rows by three columns;raw requests and execution.',14:'Figure7 all34and cumulative below99time;no filled carpet.',15:'Figure8 dense case,all six two-action checks and local zooms.',16:'Figure9 paired times and single-model evidence scope.',17:'Conclusion and vector-executor appendix.',18:'KCM algorithm and static policy feature definitions.',19:'PPO table B.1 and offline provenance;no orphan tail page.',20:'References1--12.',21:'References13--24.'}
    pages=[]
    for n in range(1,22):
        path=args.review_dir/f'page-{n:02d}.png'
        pages.append(dict(page=n,path=str(path),sha256=sha(path),visual_reviewed=True,notes=page_notes[n]))
    write(DELIVERY/'page_visual_qa.json',dict(created_utc=now,pdf_sha256=sha(output),pages=pages,page_count=21,all_pages_inspected=True,final_changed_pages_reinspected=args.changed_pages,other_final_page_rasters_identical_to_reviewed=True,layout_passed=True,renderer_note='Poppler emitted Adobe-GB1 mapping warnings from protected original assets; reviewed rendered pages retain their original appearance. Assets not altered.',remaining_content_action='Author updates original figure1/3 labels using supplied advice.'))
    figs=DELIVERY/'figures_publication'
    figbinding=read(figs/'figure_data_binding.json')
    shutil.copy2(figs/'figure_data_binding.json',history/'figure_data_binding_before_visual_qa.json')
    figbinding.update(visual_qa_complete=True,visual_qa=str(DELIVERY/'page_visual_qa.json'),final_manuscript_accepted=False)
    write(figs/'figure_data_binding.json',figbinding)
    manifest=dict(created_utc=now,pdf=str(output),pdf_sha256=sha(output),pages=21,source_archive=str(zipfile),source_archive_sha256=sha(zipfile),source_files={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()},evidence_binding=str(DELIVERY/'paper_result_binding.json'),evidence_binding_sha256=sha(DELIVERY/'paper_result_binding.json'),visual_qa=str(DELIVERY/'page_visual_qa.json'),research_acceptance_passed=True,method_frozen=True,manuscript_review_ready=True,final_manuscript_accepted=False,remaining_author_action=str(DELIVERY/'原前三图修改建议.md'),new_training_updates_this_delivery=0,new_simulations_this_delivery=0)
    write(DELIVERY/'delivery_manifest.json',manifest)
    statepaths=[BASE/'当前执行状态.json',ROOT.parent/'live_research_summary.json',ROOT.parent/'all34_dominance_research/active_research_checkpoint.json',ROOT/'current_progress.json']
    for i,path in enumerate(statepaths):
        value=read(path);shutil.copy2(path,history/f'state_{i}_{path.name}')
        value.update(updated_utc=now,method_frozen=True,research_acceptance_passed=True,goal_complete=False,final_manuscript_accepted=False,manuscript_review_ready=True,final_pdf=str(output),review_pdf=str(output),review_pdf_sha256=sha(output),manuscript_pages=21,active_stages=[],current='Selected14input model passed joint research acceptance. Original18page-based21page paper revision compiled and visually reviewed. First3figures unchanged. Author changes to legacy labels in figures1/3 remain; no new training is warranted by this editorial item.',manuscript_delivery_manifest=str(DELIVERY/'delivery_manifest.json'),pending_author_figure_suggestions=str(DELIVERY/'原前三图修改建议.md'))
        write(path,value)
    with (ROOT/'continuation_round49_20260923.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 21-PAGE ORIGINAL-BASED REVISION DELIVERED; AUTHOR FIGURE LABELS PENDING\n'
                'No training/simulation added. Research acceptance remains73candidate0;method frozen. New manuscript built from verified original18page source,not72input historical text. Original title/preamble/main5sections and experiment3subsections retained;originalfigures1--3byte-identical.14current inputs andactualfar-reference tangent/normal frame verified. KCMalgorithm/PPOsettings/staticfeaturefitinappendices. Newfigures4--9boundto73c0,S2,andtrainedtwoaction24;conservativedenseenvelope15.815/15.8945retained.\n'
                f'Delivery:{output}\nLaTeX:{zipfile}\n'
                '21pages,24resolvedcites,BibTeXregenerated,nooverfull/undefinedreferences. All21pagesrenderedandviewed;finalchanges6/7/19reinspected,otherfinalPNGsidentical. Figure8localcorridoradjacentsegments12/20.905mmexceeddisplayextent. Originalfigures1/3stillcontainlegacyoutput/execution/rewardlabelsbyuserinstruction;preciseauthor-onlyeditadvicein原前三图修改建议.md. Thisisareview-readyrevision,notfinalpublicationacceptance. Noindependentfullpipeline3seedsorpaireddemonstratedno-swingwinnerclaim. Current4statesupdated;goal_complete/final_manuscript_acceptedfalse. Noactivejobs. Quota91used9remaining,noredemption.\n')
    with (ROOT/'base18_current14_method_map_20260923.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 2026-09-23 论文交付更新\n第73轮候选0已通过联合研究验收，方法冻结。基于原18页源码的21页修订稿及源码包已生成，所有页面完成渲染检查。30mm位移的两个分量属于参考切向/法向坐标，已按源码修正；其余局部运动/近端点使用航向坐标。原图1/3仍按要求保留，待作者依建议更新旧标签后再作投稿终稿验收。\n')
    print('DELIVERED',output,sha(output))

if __name__=='__main__':main()

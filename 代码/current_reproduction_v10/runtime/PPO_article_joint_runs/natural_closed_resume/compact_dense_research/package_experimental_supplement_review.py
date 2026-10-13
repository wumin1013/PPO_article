"""Publish an explicitly interim paper packet without replacing stable delivery."""
import json
import argparse
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageChops
from build_lowdim_original_manuscript import ROOT, WORK, DELIVERY, SRC, sha, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--body-tables', action='store_true')
    args = parser.parse_args()
    build = WORK/'build_supplement'
    audit = read(build/'manuscript_integrity_audit.json')
    assert audit['pdf_sha256'] == sha(build/'manuscript.pdf')
    assert audit['source_sha256'] == sha(SRC/'manuscript.tex')
    assert audit['pages'] == 23
    stable = read(DELIVERY/'delivery_manifest.json')
    prior_qa = read(DELIVERY/'page_visual_qa.json')
    assert prior_qa['all_pages_inspected'] and prior_qa['pdf_sha256'] == stable['pdf_sha256']
    reviewed = {6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23}
    review_dir = WORK/('page_review_only_result_tables' if args.body_tables else 'page_review_supplement_v2')
    pages = []
    for i in range(1,24):
        path = review_dir/f'page-{i:02d}.png'
        same = False
        if i <= 21:
            before = WORK/'page_review_published_comparison'/f'page-{i:02d}.png'
            same = ImageChops.difference(Image.open(path).convert('RGB'), Image.open(before).convert('RGB')).getbbox() is None
        assert i in reviewed or same, f'Unreviewed changed page {i}'
        pages.append(dict(page=i, path=str(path), sha256=sha(path),
            visually_reviewed_this_revision=i in reviewed,
            pixel_identical_to_published_reviewed_page=same))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    packet = DELIVERY/'experimental_supplement_review'
    if packet.exists():
        shutil.copytree(packet, ROOT/'lowdim_manuscript_history'/f'supplement_review_{stamp}')
    packet.mkdir(exist_ok=True)
    output = packet/'manuscript_without_author_details.pdf'
    shutil.copy2(build/'manuscript.pdf', output)
    names = ['revision_origin.json', 'original_to_revised.diff', 'citation_audit.json', 'manuscript_integrity_audit.json']
    for name in names:
        shutil.copy2(build/name, packet/name)
    inputs = {name:SRC/name for name in ['manuscript.tex','references.bib','elsarticle.cls','unsrt.bst']}
    inputs['manuscript.bbl'] = build/'manuscript.bbl'
    if not inputs['manuscript.bbl'].exists():
        inputs['manuscript.bbl'] = SRC/'manuscript.bbl'
    assert sha(inputs['manuscript.bbl']) == read(build/'citation_audit.json')['bbl_sha256']
    inputs.update({f'figures_lowdim_revision/{p.name}':p for p in (SRC/'figures_lowdim_revision').glob('*.pdf')})
    archive = packet/'latex_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, path in inputs.items():
            bundle.write(path, 'latex_source/'+name)
    write(packet/'page_visual_qa.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        pdf_sha256=sha(output), all_pages_reviewed_or_pixel_identical=True, pages=pages,
        inherited_qa=str(DELIVERY/'page_visual_qa.json'), inherited_qa_sha256=sha(DELIVERY/'page_visual_qa.json'),
        scope='Changed pages reviewed individually; unchanged rasters bound to previously inspected publication packet.'))
    source = ROOT/'experimental_supplement_20260923'
    for name in ['window_sensitivity.json','window_sensitivity.csv','training_budget_audit.json','refinement_snapshot_audit.json','paper_table_audit.json','bias_inheritance_audit.json']:
        shutil.copy2(source/name, packet/name)
    note = '''# 补充实验阶段修订稿

本包基于唯一18页原稿对应源码，保留标题、主体结构与九幅主图；原图1--3字节未改。

本次加入已核实的角点窗口敏感性、历史训练预算谱系、累计历史阶段贡献。23页；24条引用；修改页面已逐页检查，未改页面与上一已检查交付逐像素相同。

三个独立初始化的完整训练流程及预算匹配的两动作对照尚未完成。本PDF继续按单个已冻结模型报告结果，没有加入未完成种子的误差条，也不把中间检查点算成完整重复。

这是补充实验的阶段审阅稿，不是最终投稿验收。原21页交付稿和历史材料保留。作者仍按原前三图修改建议更新图1/3标签。
'''
    (packet/'修订说明.md').write_text(note, encoding='utf-8', newline='\n')
    if args.body_tables:
        with (packet/'修订说明.md').open('a', encoding='utf-8') as handle:
            handle.write('\n按作者最新要求，正文只放实验结果表：表1完整效率（第9页）、表2窗口敏感性（第15页）、表3历史阶段完整时间（第17页）。PPO参数表B.1（第20页）及训练谱系/预算表B.2（第21页）留在附录。KCM及实现细节也在附录；进给输出偏置的离线参数选择已披露。\n')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), pdf=str(output), pdf_sha256=sha(output),
        pages=23, source_archive=str(archive), source_archive_sha256=sha(archive),
        source_files={name:sha(path) for name,path in inputs.items()},
        stable_delivery_preserved=str(DELIVERY/'delivery_manifest.json'),
        stable_delivery_manifest_sha256=sha(DELIVERY/'delivery_manifest.json'),
        paper_result_binding=str(DELIVERY/'paper_result_binding.json'),
        paper_result_binding_sha256=sha(DELIVERY/'paper_result_binding.json'),
        supplement_progress=str(source/'progress.json'), selected_model_frozen=True,
        independent_final_pipeline_repetitions_established=0,
        working_training_results_included=False, review_ready=True, final_manuscript_accepted=False)
    manifest['experimental_tables_in_main_text'] = args.body_tables
    manifest['experimental_result_tables_in_main_text'] = args.body_tables
    manifest['training_parameter_tables_in_appendix'] = args.body_tables
    manifest['training_budget_table_in_appendix'] = args.body_tables
    write(packet/'delivery_manifest.json', manifest)
    print('INTERIM_REVIEW_PACKET', output, sha(output))


if __name__ == '__main__':
    main()

"""Verify the bounded move and record the final release locations without changing research results."""
import sys,os,shutil,json,zipfile,subprocess
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from prepare_eswa_release import ROOT,WORK,BASE,CURRENT,OLD,SYNC,read,write,sha

ARCH=Path('D:/PPO_article_cleanup_archive/20260924_before_five_folders')
PAPER=SYNC/'论文';PKG=SYNC/'新EWSA发布包';DATA=SYNC/'论文用到的数据和模型';CODE=SYNC/'代码'
QA=PAPER/'核验与修改说明'
now=datetime.now(timezone.utc).isoformat()

def verify():
    before=read(WORK/'sync_full_inventory_hashed.json')
    overwritten={r['target']:r for r in read(WORK/'original_figure_overwrite_receipt.json')['records']}
    root_kept={'.git','.gitignore','AGENTS.md','PPO_存储与清理规则.md'}
    rows=[]
    for r in before['files']:
        old=SYNC/r['relative'];p=old if r['relative'] in root_kept else ARCH/r['relative']
        extended=Path('\\\\?\\'+str(p))
        expected=overwritten.get(str(old),{}).get('after_sha256',r['sha256'])
        assert extended.is_file(),p
        assert sha(extended)==expected,p
        if str(old) in overwritten:
            assert sha(overwritten[str(old)]['backup'])==r['sha256']
        rows.append(dict(original=str(old),current=str(p),original_sha256=r['sha256'],current_sha256=expected))
    original=read(WORK/'original_EAAI_hashes.json')
    assert {str(p.relative_to(SYNC/'EAAI发布包')):sha(p) for p in (SYNC/'EAAI发布包').rglob('*') if p.is_file()}==original
    evidence=read(DATA/'来源与整理路径.json')
    for r in evidence['records']:assert sha(DATA/r['relative'])==r['sha256'],r['relative']
    release=read(PKG/'release_manifest.json')
    for r in release['files']:assert sha(PKG/r['relative'])==r['sha256'],r['relative']
    categories={'代码','论文用到的数据和模型','论文','EAAI发布包','新EWSA发布包'}
    assert {p.name for p in SYNC.iterdir() if p.is_dir()}==categories
    write(WORK/'verified_path_migration.json',dict(created_utc=now,archive=str(ARCH),records=rows,
        old_EAAI_hashes_unchanged=True,evidence_records_verified=len(evidence['records']),new_release_files_verified=len(release['files'])))
    print('VERIFIED old files',len(rows),'evidence records',len(evidence['records']),'new package',len(release['files']),flush=True)

def record():
    # Copy only existing paper-facing revision guides, not exploration logs.
    for name in ['图件优化与非实验修改说明.pdf','图件优化与非实验修改说明.md']:
        shutil.copy2(OLD/'academic_figures_20260924'/name,QA/name)
    shutil.copy2(OLD/'original_to_revised.diff',QA/'原18页稿至修订稿.diff')
    shutil.copy2(OLD/'non_experimental_changes.diff',QA/'非实验部分修改.diff')
    for name in ['citation_audit.json','paper_result_binding.json','nnc_evidence_binding.json']:
        shutil.copy2(OLD/name,QA/name)
    tools=CODE/'release_tools';tools.mkdir(exist_ok=True)
    for p in [ROOT/'prepare_eswa_release.py',ROOT/'finalize_eswa_release.py']+list(WORK.glob('*.py'))+list(WORK.glob('*.ps1'))+list(WORK.glob('*.cjs')):
        shutil.copy2(p,tools/p.name)
    (tools/'README.md').write_text('发布整理脚本归档，仅用于理解本次操作；含原绝对路径和一次性移动步骤，请勿直接重复运行。完整操作收据在非同步D盘release_cleanup_20260924。\n',encoding='utf-8')
    (SYNC/'目录说明.md').write_text(f'''# 当前目录

- 代码：PPO代码、配置及论文研究/绘图脚本的同步副本。
- 论文用到的数据和模型：论文绑定数据、选定模型、必要训练依据及核验摘要。
- 论文：英文23页、中文21页PDF、图件、中文源码和给辅导老师的4页改进清单。
- EAAI发布包：原18页稿及原发布文件，内容保持不变。
- 新EWSA发布包：面向Expert Systems with Applications（ESWA）的新投稿文件。

实际工作目录仍为D:/PPO_article_work；研究运行目录仍为D:/PPO_article_joint_runs，分支仍为codex/complex-path-adaptive-preview。
根目录.git、.gitignore、AGENTS.md和存储规则是必要管理文件。

2026-09-24整理前历史材料已移至：{ARCH.as_posix()}。
没有删除原D盘训练历史。旧路径与归档路径对应关系：{(WORK/'verified_path_migration.json').as_posix()}。
唯一18页基稿的新位置为EAAI发布包/01_manuscript_without_author_details/manuscript_without_author_details.pdf。
''',encoding='utf-8')
    storage=SYNC/'PPO_存储与清理规则.md'
    text=storage.read_text('utf-8')
    update='''

## 2026-09-24 用户授权整理补充

坚果云根目录现在按“代码、论文用到的数据和模型、论文、EAAI发布包、新EWSA发布包”五类保存。原EAAI包内容未改；新包目标为ESWA，文件夹拼写按用户要求。
旧同步残留已移至D:/PPO_article_cleanup_archive/20260924_before_five_folders，并保留SHA-256和路径映射。原D盘研究历史不在本次清理范围。后续不得根据本次授权再次自动清理其他目录。
'''
    storage.write_text(text+update,encoding='utf-8')
    manifest=read(CURRENT/'delivery_manifest.json')
    manifest.update(updated_utc=now,version='eswa_release_20260924',
        pdf=str(PAPER/'英文稿_ESWA.pdf'),pdf_sha256=sha(PAPER/'英文稿_ESWA.pdf'),
        chinese_pdf=str(PAPER/'中文稿.pdf'),chinese_pdf_sha256=sha(PAPER/'中文稿.pdf'),
        source_archive=str(PKG/'06_latex_source_files/latex_source_anonymous.zip'),source_archive_sha256=sha(PKG/'06_latex_source_files/latex_source_anonymous.zip'),
        source_directory=str(PKG/'06_latex_source_files/latex_source_anonymous'),chinese_source_directory=str(PAPER/'中文源码'),
        revision_notes_pdf=str(QA/'图件优化与非实验修改说明.pdf'),revision_notes_pdf_sha256=sha(QA/'图件优化与非实验修改说明.pdf'),
        teacher_comparison_pdf=str(PAPER/'给辅导老师/新旧稿改进清单.pdf'),teacher_comparison_sha256=sha(PAPER/'给辅导老师/新旧稿改进清单.pdf'),teacher_comparison_pages=4,
        eswa_package=str(PKG),old_EAAI_package=str(SYNC/'EAAI发布包'),old_EAAI_content_unchanged=True,
        original_18_page_pdf=str(SYNC/'EAAI发布包/01_manuscript_without_author_details/manuscript_without_author_details.pdf'),
        original_master_figures_overwritten=True,full_svg_1_to_3='PDF-derived outlined vectors; Illustrator layer history unavailable',
        cleanup_archive=str(ARCH),migration_audit=str(WORK/'verified_path_migration.json'),
        paper_ready_for_author_submission_review=True,release_packaging_complete=True,teacher_comparison_complete=True,
        remaining_author_action='Corresponding author to confirm declaration before actual submission.',
        remaining_author_action_summary='No figure or experiment task remains in this release. Author must confirm declaration and current submission status before uploading.',
        next_step='Await author feedback or new instructions; preserve frozen method. Do not repeat training, compilation or cleanup without a new request.',
        figure_master_directory=str(PAPER/'图件'),latest_version_manifest=str(QA/'当前交付清单.json'))
    write(QA/'当前交付清单.json',manifest)
    history=WORK/'status_before_release';history.mkdir(exist_ok=True)
    # Mutable entry-point manifests get a backup; the immutable previous version directory stays untouched.
    for index,p in enumerate([OLD/'delivery_manifest.json',OLD.parent/'delivery_manifest.json']):
        if p.is_file():shutil.copy2(p,history/f'manifest_{index}.json')
        write(p,manifest)
    state_paths=[BASE/'当前执行状态.json',ROOT/'current_progress.json',ROOT.parent/'live_research_summary.json',ROOT.parent/'all34_dominance_research/active_research_checkpoint.json']
    for index,p in enumerate(state_paths):
        state=read(p);shutil.copy2(p,history/f'state_{index}.json')
        state.update(updated_utc=now,active_stages=[],method_frozen=True,research_acceptance_passed=True,
            current='ESWA release and bilingual PDFs delivered; original figure masters overwritten with backups; old EAAI package unchanged; sync root organized into five folders. Four-page teacher comparison included. No new training or simulation.',
            final_pdf=manifest['pdf'],review_pdf=manifest['pdf'],review_pdf_sha256=manifest['pdf_sha256'],
            chinese_review_pdf=manifest['chinese_pdf'],chinese_review_pdf_sha256=manifest['chinese_pdf_sha256'],
            manuscript_delivery_manifest=str(QA/'当前交付清单.json'),manuscript_delivery_manifest_sha256=sha(QA/'当前交付清单.json'),
            manuscript_source_archive=manifest['source_archive'],revision_notes_pdf=manifest['revision_notes_pdf'],
            teacher_comparison_pdf=manifest['teacher_comparison_pdf'],eswa_release_package=str(PKG),
            release_packaging_complete=True,cleanup_complete=True,cleanup_archive=str(ARCH),
            remaining_author_action=manifest['remaining_author_action_summary'],next_step=manifest['next_step'])
        write(p,state)
    with (ROOT/'continuation_round49_20260923.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {now} — ESWA release and bounded sync cleanup\n\nUser-authorized original master PDF overwrite completed with backup; complete SVGs exported as outlines. Old EAAI package hashes unchanged. English 23-page and Chinese 21-page PDFs compile and are pixel-identical to the prior accepted-layout revision. New target journal is ESWA; folder name is 新EWSA发布包. Four-page teacher change list delivered. Five-category sync layout installed; all previous sync files verified in archive or retained root. No training/simulation or branch change. Current manifest: {QA/"当前交付清单.json"}. Archive: {ARCH}. Author declaration confirmation remains for actual submission.\n')
    # Repackage only the final submission directory, outside that directory to avoid recursion.
    archive_dir=PAPER/'交付归档';archive_dir.mkdir(exist_ok=True)
    zip_path=archive_dir/'新EWSA发布包.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in PKG.rglob('*'):
            if p.is_file():z.write(p,Path(PKG.name)/p.relative_to(PKG))
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
    stats={}
    for d in SYNC.iterdir():
        if d.is_dir():
            files=[p for p in d.rglob('*') if p.is_file()]
            stats[d.name]=dict(files=len(files),bytes=sum(p.stat().st_size for p in files))
    write(WORK/'completion_receipt.json',dict(created_utc=now,folders=stats,total_bytes=sum(r['bytes'] for r in stats.values()),
        prior_sync_bytes=read(WORK/'sync_full_inventory_hashed.json')['bytes'],submission_zip=str(zip_path),submission_zip_sha256=sha(zip_path),
        archive=str(ARCH),manifest=str(QA/'当前交付清单.json'),all_requested_packaging_tasks_complete=True))
    shutil.copy2(WORK/'completion_receipt.json',QA/'整理完成收据.json')
    print(json.dumps(stats,ensure_ascii=False),flush=True)

if __name__=='__main__':
    verify();record()

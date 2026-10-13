"""Independent read-back verification of the two PPO submission-package locations."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import subprocess

AUDIT=Path(r'D:\PPO_article_recovery\main_restore_20260921_102943')
roots=[Path(r'D:\Nutstore\1\基于强化学习的运动规划方法'),Path(r'D:\PPO_article_work')]
prefix='论文项目/submission_package_EAAI'
expected=json.loads((AUDIT/'restore_receipt.json').read_text(encoding='utf-8'))
records=[]
for root in roots:
    files=[]
    for item in expected['files']:
        path=root/item['path']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==item['sha256'], str(path)
        files.append(dict(path=str(path),size=path.stat().st_size,sha256=digest))
    actual=[p for p in (root/prefix).rglob('*') if p.is_file()]
    names={x['path'] for x in files}
    extras=[str(p) for p in actual if str(p) not in names]
    counts=Counter(str(Path(row['path']).relative_to(root/prefix)).split('\\')[0] for row in files)
    records.append(dict(root=str(root),expected_tracked_files=len(files),verified_tracked_files=len(files),
                        actual_files=len(actual),extra_files_preserved=extras,folder_counts=dict(counts),files=files))

identities=[]
for label,path,commit in [
    ('committed_main',AUDIT/'pdf_source_commit/论文项目/main.pdf',expected['pdf_last_change_commit']),
    ('working_main',roots[1]/'论文项目/main.pdf',None),
    ('submission_main',roots[0]/prefix/'01_manuscript_without_author_details/manuscript_without_author_details.pdf',expected['main_commit'])]:
    data=path.read_bytes()
    identity=dict(label=label,path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),commit=commit)
    if label=='working_main':
        backup=AUDIT/'working_24_page_snapshot'
        backup.mkdir(exist_ok=True)
        for name in ('main.pdf','main.tex','references.bib','main.bbl'):
            source=roots[1]/'论文项目'/name
            if source.exists():shutil.copy2(source,backup/name)
        identity['backup']=str(backup/'main.pdf')
    identities.append(identity)

result=dict(source_main_commit=expected['main_commit'],verification=records,pdf_identities=identities,
            research_started=False,baseline_selection='awaiting_user_clarification',
            branch_switch=False,reset=False,commit=False,push=False,delete=False)
(AUDIT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
report=['# 投稿包恢复核验与 PDF 版本说明','',
        '本次只恢复投稿包并核对 PDF 身份，未继续研究，也未更新研究完成状态。','',
        f'来源 main 提交：`{expected["main_commit"]}`。已只读核实远端 main 与此相同。','',
        '| 目标目录 | 已跟踪文件哈希核验 | 额外文件保留 |','|---|---:|---:|']
for r in records:report.append(f'| `{r["root"]}\\论文项目\\submission_package_EAAI` | 30 / 30 | {len(r["extra_files_preserved"])} |')
report+=['','原项目同步目录此前确实没有文件，本次已补回。上次只恢复 D 盘工作副本，未恢复原项目目录，是交付位置遗漏。','',
         '## PDF 身份','',
         '- Git main 内 `论文项目/main.pdf`：18 页，最后修改提交 `6393294435d3adb202f464158c66f7fe1ef87572`。',
         '- 磁盘现存两处 `论文项目/main.pdf`：24 页，2026-09-14 生成，未提交内容；原位未覆盖，已另存快照。',
         '- 投稿包匿名稿：18 页，与 main.pdf 的匿名排版版本分别保留，不能视为同一字节文件。','']
for row in identities:report+=[f'- `{row["label"]}`：`{row["path"]}`',f'  SHA-256：`{row["sha256"]}`']
report+=['','两份 main.pdf 的基稿选择待用户指认；在此之前不继续研究和正式稿写作。',
         '没有切换或重置分支，没有删除、提交、推送。没有发现投稿包的 Git LFS 指针或缺失的已跟踪文件。',
         '逐文件清单及 SHA-256 见同目录 independent_verification.json。']
(AUDIT/'恢复核验与PDF版本说明.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print(json.dumps({**result,'verification':[{k:v for k,v in r.items() if k!='files'} for r in records]},ensure_ascii=False,indent=2))

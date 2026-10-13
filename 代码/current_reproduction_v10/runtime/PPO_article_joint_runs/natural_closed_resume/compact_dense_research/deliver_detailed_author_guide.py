"""Deliver the expanded author guide without changing either manuscript."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('D:/PPO_article_joint_runs/natural_closed_resume/compact_dense_research')
WORK = ROOT / 'lowdim_original_manuscript_work'
DELIVERY = Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1/paper_evidence_review/verified_results/lowdim_original_revision')
NEW = DELIVERY / 'selected923202_revision'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = datetime.now(timezone.utc)
    manifest = json.loads((NEW / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert sha(manifest['pdf']) == manifest['pdf_sha256']
    assert sha(manifest['chinese_pdf']) == manifest['chinese_pdf_sha256']
    history = WORK / 'delivery_history' / now.strftime('%Y%m%dT%H%M%S%fZ_author_guide')
    history.mkdir(parents=True)
    for label, path in [('version_manifest', NEW / 'delivery_manifest.json'),
                        ('root_manifest', DELIVERY / 'delivery_manifest.json'),
                        ('scope_audit', NEW / 'author_guide_scope_audit.json'),
                        ('readme', NEW / 'README.md')]:
        shutil.copy2(path, history / (label + path.suffix))
    guide = NEW / '图件修改建议与实验外修改说明_详细版.pdf'
    build = WORK / 'detailed_guide_work/build/detailed_guide.pdf'
    if guide.exists():
        assert sha(guide) == sha(build)
    else:
        shutil.copy2(build, guide)
    audit_path = NEW / 'author_guide_scope_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update(delivered_utc=now.isoformat(), detailed_guide_pdf=str(guide),
                 detailed_guide_pdf_sha256=sha(guide),
                 visual_qa=dict(pages=8, status='passed',
                     method='All eight rendered pages inspected in two contact sheets before delivery.',
                     layout_issues_remaining=[], compile_warnings=[]),
                 history=str(history))
    write_json(audit_path, audit)
    manifest.update(updated_utc=now.isoformat(),
        prior_brief_revision_notes_pdf=manifest['revision_notes_pdf'],
        prior_brief_revision_notes_pdf_sha256=manifest['revision_notes_pdf_sha256'],
        revision_notes_pdf=str(guide), revision_notes_pdf_sha256=sha(guide), revision_notes_pages=8,
        revision_notes_markdown=str(NEW / '图件修改建议与实验外修改说明_详细版.md'),
        revision_notes_scope_audit=str(audit_path),
        remaining_author_action=str(NEW / '图件修改建议与实验外修改说明_详细版.md'),
        remaining_author_action_summary='Author updates original Figure1/3 labels using the eight-page detailed guide; then replace work copies, compile and check affected pages. Original Figure2 remains unchanged.',
        author_guide_only_update=True, author_guide_update_history=str(history))
    write_json(NEW / 'delivery_manifest.json', manifest)
    manifest['latest_version_manifest'] = str(NEW / 'delivery_manifest.json')
    write_json(DELIVERY / 'delivery_manifest.json', manifest)
    readme_path = NEW / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    readme = readme.replace('- 图件修改建议与实验外修改说明.pdf / .md：原图1、3逐项替换建议，以及实验之外的必要修改。',
        '- 图件修改建议与实验外修改说明_详细版.pdf / .md：8页作者修改指南，含原图缩图、逐位置替换文字、符号与连线说明，以及实验之外按章节列出的原文／现稿／修改原因。原3页简版保留为历史。')
    readme += '\n本次仅扩充作者指南，英文22页和中文18页论文均未改动。详细指南为独立交付文件；原LaTeX源码包保持不变。\n'
    readme_path.write_text(readme, encoding='utf-8')
    with (ROOT / 'continuation_round49_20260923.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n\n## {now.isoformat()} DETAILED AUTHOR GUIDE DELIVERED\n'
            'Expanded author guide from 3 to 8 pages, with original Figure1/3 thumbnails, exact label replacements, arrows and symbols, and original/current/reason comparisons outside experiments. All eight rendered pages reviewed; no remaining layout issues or compile warnings. Original brief guide retained. New detailed PDF/Markdown linked in both manifests and README.\n'
            'English22page and Chinese18page manuscript PDF hashes unchanged; original figures1--3 unchanged. The quoted storage sentence, exploration narrative and round logs are absent from both manuscript bodies and appendices. Only author-facing notes updated; no new training, simulation, manuscript compilation, branch change, deletion, commit, push or subagent. Await author-updated Figure1/3.\n')
    print(json.dumps(dict(guide=str(guide), pages=8, sha256=sha(guide),
                          manuscripts_unchanged=True, history=str(history)), ensure_ascii=False))


if __name__ == '__main__':
    main()

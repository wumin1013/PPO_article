"""Fingerprint the delivered files in place; do not duplicate raw experiments."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import preview_event_study as study
from verify_preview_event_preservation import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing delivery inventory')
    revision = study.OUT.parent
    audit = study.read(revision / 'final_delivery_audit.json')
    manuscript = audit.get('manuscript', {})
    if (audit.get('mismatch_count') != 0 or len(audit.get('trajectories', [])) != 204 or
            any(manuscript.get(key, ['missing']) for key in
                ('missing_citations', 'missing_inputs', 'undefined_reference_warnings', 'overfull_boxes'))):
        raise ValueError('Complete trajectory/manuscript audit required first')
    if manuscript.get('tex_sha256') != sha(study.ROOT / '论文项目/main.tex') or manuscript.get('pdf_sha256') != sha(study.ROOT / '论文项目/main.pdf'):
        raise ValueError('Manuscript changed since final audit')
    preservation = study.read(revision / 'preservation_all_artwork_verification.json')
    if not preservation['success']:
        raise ValueError('Archive protection audit failed')
    state = study.source_state()
    if state['source_sha256'] != study.read(study.OUT / 'freeze.json')['source_sha256']:
        raise ValueError('Frozen numerical source has changed')
    files = {study.ROOT / name for name in state['source_files']}
    for directory in (revision, study.DATA, study.ROOT / 'PPO_project/tests'):
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    tools = study.ROOT / 'PPO_project/tools'
    for pattern in ('preview_event_*.py', 'verify_preview_event_*.py'):
        files.update(tools.glob(pattern))
    paper = study.ROOT / '论文项目'
    files.update(paper / name for name in ('main.tex', 'main.pdf', 'main.log', 'main.bbl', 'references.bib',
                                          '修改说明.md', '实验复现说明.md', '固定前瞻与自适应前瞻原因分析.md'))
    files.update((paper / 'generated').glob('*.tex'))
    files.update(study.ROOT / row['path'] for row in preservation['active_figures'])
    # The output is not part of its own inventory. Existing runs are only read.
    rows = {p.relative_to(study.ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(files) if p.resolve() != output}
    git = lambda *parts: subprocess.check_output(['git', *parts], cwd=study.ROOT, text=True).strip()
    payload = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), branch=git('branch', '--show-current'),
        reference_commit=git('rev-parse', 'main'), head=git('rev-parse', 'HEAD'),
        numerical_source_sha256=state['source_sha256'], files=rows, file_count=len(rows),
        total_bytes=sum(row['bytes'] for row in rows.values()),
        scope='In-place fingerprints of event experiments, test data, current numerical/report tools, tests, manuscript and preserved active artwork',
        exclusions=['Git objects, caches and unrelated project files',
                    'Predecessor raw experiments are covered by before_event_manifest.json protection records'],
        source_control_note='Uncommitted work retained on the existing revision branch; no remote push',
        figures_note='Original active artwork preserved; event reference drafts are not active manuscript figures')
    study.write(output, payload)
    print(json.dumps({k: payload[k] for k in ('branch', 'numerical_source_sha256', 'file_count', 'total_bytes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

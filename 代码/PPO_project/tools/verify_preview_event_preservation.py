"""Verify the pre-revision archive and retained historical artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / '论文项目/revision_20260909_event_preview'


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous verification')
    record = json.loads((REVISION / 'before_event_manifest.json').read_text(encoding='utf-8'))
    archive = REVISION / 'before_event_revision.zip'
    mismatches = []
    if sha(archive) != record['archive_sha256']:
        mismatches.append('before_event_archive_sha256')
    with zipfile.ZipFile(archive) as saved:
        for name, expected in record['archived_files'].items():
            if hashlib.sha256(saved.read(name)).hexdigest() != expected:
                mismatches.append('archive_member:' + name)
        old = saved.read('论文项目/main.tex').decode('utf-8')
    for name, expected in record['protected_original_formal_files'].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected:
            mismatches.append('protected_historical_file:' + name)
    archived_artwork = {name: expected for name, expected in record['archived_files'].items()
                        if name.startswith('论文项目/figures/') and Path(name).suffix.lower() in ('.pdf', '.png', '.svg')}
    for name, expected in archived_artwork.items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected:
            mismatches.append('archived_artwork:' + name)
    paper = ROOT / '论文项目'
    active = (paper / 'main.tex').read_text(encoding='utf-8')
    pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    old_figures, figures = re.findall(pattern, old), re.findall(pattern, active)
    if old_figures != figures:
        mismatches.append('active_includegraphics_sequence')
    figure_rows = []
    for name in figures:
        candidates = [paper / name, paper / 'figures' / name]
        path = next((p for p in candidates if p.is_file()), None)
        key = path.relative_to(ROOT).as_posix() if path else name
        expected = record['archived_files'].get(key)
        actual = sha(path) if path else None
        # Every active artwork must have an archived counterpart; unknown is not a pass.
        if expected is None or actual != expected:
            mismatches.append('active_figure_bytes:' + key)
        figure_rows.append(dict(path=key, expected_sha256=expected, actual_sha256=actual))
    git = lambda *parts: subprocess.check_output(['git', *parts], cwd=ROOT, text=True).strip()
    result = dict(archive_sha256=sha(archive), archived_members_checked=len(record['archived_files']),
        retained_historical_files_checked=len(record['protected_original_formal_files']),
        archived_artwork_files_checked=len(archived_artwork),
        includegraphics_sequence_unchanged=old_figures == figures, active_figures=figure_rows,
        branch=git('branch', '--show-current'), head=git('rev-parse', 'HEAD'),
        original_branch=record['branch'], original_head=record['head'],
        mismatches=mismatches, success=not mismatches,
        scope='Pre-revision archive integrity, retained predecessor formal results, active figure sequence and artwork bytes',
        script_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('success', 'archived_members_checked', 'retained_historical_files_checked', 'mismatches')}, ensure_ascii=False))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

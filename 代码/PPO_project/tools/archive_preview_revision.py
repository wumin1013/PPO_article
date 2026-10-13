"""Protect the current dirty implementation before the event-preview revision."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '论文项目/revision_20260909_event_preview'


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT / 'before_event_revision.zip'
    if target.exists():
        raise FileExistsError('Preserve the previous archive')
    files = set()
    for folder in ['PPO_project/src', 'PPO_project/tools', 'PPO_project/tests', 'PPO_project/configs', 'PPO_project/data']:
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(ROOT / name for name in ['AGENTS.md', 'PPO_project/main.py', 'trajectory_autoresearch/paper_sync.py'])
    files.update(p for p in (ROOT / '论文项目').iterdir() if p.is_file() and p.suffix in ('.tex', '.bib', '.pdf', '.md'))
    for folder in ['论文项目/generated', '论文项目/figures', '论文项目/revision_20260909_cause_analysis']:
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    # The formal model/source/config snapshots are copied too. All original raw
    # trajectories remain in their immutable run folders with a hash inventory.
    formal = ROOT / '论文项目/revision_20260908/preview_direct_explore/formal'
    for run in formal.iterdir():
        files.update(p for p in run.iterdir() if p.is_file() and p.name in ('checkpoint.pt', 'source.zip', 'config.yaml', 'provenance.json'))
    hashes = {}
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(files):
            raw = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            archive.writestr(name, raw)
    preserved = {}
    for path in formal.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            preserved[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), archived_files=hashes,
        protected_original_formal_files=preserved, original_results_copied=False,
        original_results_policy='Retained in original run folders; never written by the new study')
    (OUT / 'before_event_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(archive=str(target), archive_files=len(files), protected_original_files=len(preserved), sha256=record['archive_sha256'])))


if __name__ == '__main__':
    main()

"""Restore main's tracked submission package without changing Git index or refs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(r'D:\PPO_article_work')
STAMP = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
AUDIT = Path(r'D:\PPO_article_recovery') / ('main_restore_' + STAMP)
PACKAGE = '论文项目/submission_package_EAAI'

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.quotepath=false', *args], cwd=ROOT)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def tree(ref, prefix=None):
    args = ['ls-tree', '-r', '-z', ref]
    if prefix:
        args += ['--', prefix]
    entries = []
    for entry in git(*args).split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        entries.append(dict(path=name.decode('utf-8'), mode=mode, kind=kind, blob=oid))
    return entries

def main():
    AUDIT.mkdir(parents=True, exist_ok=False)
    head = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    main_commit = git('rev-parse', 'main').decode().strip()
    pdf_commit = git('log', '-1', '--format=%H', 'main', '--', '论文项目/main.pdf').decode().strip()
    for name, args in [('status_before.bin', ['status', '--porcelain=v1', '-z', '--untracked-files=all']),
                       ('tracked_names_before.txt', ['diff', '--name-status', 'HEAD']),
                       ('staged_diff_before.patch', ['diff', '--cached', '--binary'])]:
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True)
        (AUDIT/name).write_bytes(result.stdout)
        (AUDIT/(name+'.stderr.txt')).write_bytes(result.stderr)
    index_path = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index_path.is_absolute():
        index_path = ROOT/index_path
    index_before = index_path.read_bytes()
    (AUDIT/'git_index_before').write_bytes(index_before)
    all_entries = tree(main_commit)
    write_json(AUDIT/'main_tree.json', all_entries)
    entries = [r for r in all_entries if r['path'].startswith(PACKAGE+'/')]
    assert entries, 'Submission package not found in main'
    assert all(r['kind']=='blob' and r['mode'] in ('100644','100755') for r in entries), 'Non-regular tracked item requires explicit handling'
    source_names = {r['path'] for r in entries}
    before = []
    target_dir = ROOT/PACKAGE
    if target_dir.exists():
        for folder, dirs, files in os.walk(target_dir, followlinks=False):
            for name in files:
                p = Path(folder)/name
                assert not p.is_symlink(), str(p)
                before.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p.read_bytes()), size=p.stat().st_size))
    write_json(AUDIT/'target_inventory_before.json', before)
    for row in entries:
        data = git('cat-file', 'blob', row['blob'])
        row.update(sha256=sha(data), size=len(data), lfs_pointer=data.startswith(b'version https://git-lfs.github.com/spec/v1'))
        source = AUDIT/'main_package_snapshot'/row['path']
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        target = ROOT/row['path']
        if target.exists() and target.read_bytes() != data:
            backup = AUDIT/'conflict_backup'/row['path']
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            assert backup.read_bytes() == target.read_bytes()
            row['conflict_backup'] = str(backup)
        if not target.exists() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        row['verified_sha256'] = sha(target.read_bytes())
        assert row['verified_sha256'] == row['sha256']
    extras = [r for r in before if r['path'] not in source_names]
    for row in extras:
        assert sha((ROOT/row['path']).read_bytes()) == row['sha256']
    pdf_entries = tree(pdf_commit)
    write_json(AUDIT/'pdf_commit_tree.json', pdf_entries)
    snapshots = []
    for row in pdf_entries:
        if row['path'] in ('论文项目/main.pdf', '论文项目/main.tex') or (row['path'].startswith('论文项目/') and row['path'].endswith(('.bib', '.bbl', '.cls', '.bst'))):
            data = git('cat-file', 'blob', row['blob'])
            p = AUDIT/'pdf_source_commit'/row['path']
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            snapshots.append(dict(**row, sha256=sha(data), size=len(data), snapshot=str(p)))
    assert git('rev-parse', 'HEAD').decode().strip() == head
    assert git('branch', '--show-current').decode().strip() == branch
    assert index_path.read_bytes() == index_before
    receipt = dict(timestamp=STAMP, repository=str(ROOT), branch=branch, head=head, main_commit=main_commit,
                   pdf_last_change_commit=pdf_commit, package=PACKAGE, tracked_count=len(entries),
                   conflicts=sum('conflict_backup' in r for r in entries), branch_extras_preserved=extras,
                   lfs_pointers=[r['path'] for r in entries if r['lfs_pointer']], files=entries, snapshots=snapshots,
                   index_unchanged=True, head_unchanged=True, audit_directory=str(AUDIT))
    write_json(AUDIT/'restore_receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','snapshots','branch_extras_preserved')}, ensure_ascii=False, indent=2))
    print('PDF:', json.dumps([r for r in snapshots if r['path'].endswith('main.pdf')],ensure_ascii=False))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', default=str(ROOT))
    args = parser.parse_args()
    ROOT = Path(args.repository).resolve()
    allowed = [Path(r'D:\PPO_article_work'), Path(r'D:\Nutstore\1\基于强化学习的运动规划方法')]
    assert ROOT in allowed, 'Only the two existing PPO project working trees are allowed'
    main()

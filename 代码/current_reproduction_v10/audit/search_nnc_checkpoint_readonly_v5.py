"""Hash narrowly named NNC checkpoint candidates from the existing D-drive inventory."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import sha, utc, write_json

EXPECTED = '24d263a0e07fe471e7995ebae2d0a0ee7a2ee03e2feec80ea50aa5b94b0cfe6b'


def wanted(p):
    return (p.suffix.lower() in ('.pt', '.pth') and
            ('nnc' in str(p).lower() or 'baseline_policy' in str(p).lower()
             or p.name.lower().startswith('best')))


def main():
    started = time.perf_counter()
    index = ROOT / 'audit/historical_asset_recovery_search_v6_inventory.json'
    old = json.loads(index.read_text('utf-8'))
    candidates = {Path(s) for scan in old for s in scan['files'] if wanted(Path(s))}
    scans = []
    # The earlier v5 file inventory included .pt but omitted .pth. Read these
    # two original roots only, constrained to the registered checkpoint names.
    for directory in (Path('D:/PPO_article_work'), Path('D:/Nutstore/1/基于强化学习的运动规划方法')):
        command = ['rg', '--files', '--hidden', '--no-ignore',
                   '-g', '*.pth', '-g', '*.pt', str(directory)]
        proc = subprocess.run(command, capture_output=True)
        names = proc.stdout.decode('utf-8').splitlines()
        selected = [Path(s) for s in names if wanted(Path(s))]
        candidates.update(selected)
        scans.append(dict(root=str(directory), command=command,
                          returncode=proc.returncode, model_filename_count=len(names),
                          selected_filename_count=len(selected), stderr=proc.stderr.decode('utf-8')))
    rows = []
    for p in sorted(candidates, key=str):
        row = dict(path=str(p), exists=p.is_file())
        if p.is_file():
            row.update(bytes=p.stat().st_size, sha256=sha(p))
            row['matches_registration'] = row['sha256'] == EXPECTED
            if row['matches_registration']:
                print('EXACT_NNC_CHECKPOINT', p, flush=True)
        rows.append(row)
    out = dict(created_utc=utc(), operation='Read-only filename-filtered file hashing; no model unpickling, training or rollout.',
               target_sha256=EXPECTED, target_registered_basename='best_model.pth',
               reused_inventory=str(index), reused_inventory_sha256=sha(index),
               name_filter='PT/PTH with NNC or baseline_policy in path, or best-prefixed basename',
               additional_scans=scans, candidates=rows,
               exact_matches=[r for r in rows if r.get('matches_registration')],
               candidates_hashed=sum(r['exists'] for r in rows), wall_s=time.perf_counter()-started,
               scope_limit='No assertion that a differently named file anywhere on D: cannot hold the target. Archive model entries are covered only by prior named searches, not this file hash pass.')
    write_json(ROOT / 'audit/nnc_checkpoint_targeted_search_v5.json', out)
    print('HASHED', out['candidates_hashed'], 'EXACT_MATCHES', len(out['exact_matches']), flush=True)


if __name__ == '__main__':
    main()

"""Re-run frozen evaluation in a new directory without touching retained evidence."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, sha, write_json, utc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Run the entire 100-cell evaluation and regenerate figures')
    args=parser.parse_args()
    target=args.output.resolve()
    if target==ROOT or ROOT in target.parents:
        raise ValueError('Use a new directory outside the source evidence tree')
    if target.exists():raise FileExistsError(target)
    frozen=json.loads((ROOT/'audit'/'test_freeze.json').read_text('utf-8'))
    for name,digest in frozen['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source mismatch: '+name)
    files={ROOT/name for name in frozen['sha256']}
    files.update((ROOT/'src').rglob('*.py'))
    files.update(p for p in (ROOT/'results'/'paired').rglob('*') if p.is_file() and p.suffix in ('.pt','.json','.csv','.npz'))
    for folder in (ROOT/'results').glob('teacher_selection*'):
        files.update(folder.glob('*.json'))
    files.add(ROOT/'audit'/'test_freeze.json')
    for source in sorted(files):
        destination=target/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    log=[]
    commands=[['src/test_contract.py']]
    if args.execute:
        commands += [['src/run_queue.py','test'],['src/summarize_results.py'],['src/plot_results.py']]
    for command in commands:
        completed=subprocess.run([sys.executable,'-B',*command],cwd=target)
        log.append(dict(command=command,returncode=completed.returncode))
        if completed.returncode:
            write_json(target/'audit'/'reproduction_status.json',dict(complete=False,commands=log))
            raise RuntimeError('Reproduction command failed: '+' '.join(command))
    comparisons=[]
    if args.execute:
        expected=json.loads((ROOT/'results'/'summary'/'all_results.json').read_text('utf-8'))['all_grid_records']
        actual=json.loads((target/'results'/'summary'/'all_results.json').read_text('utf-8'))['all_grid_records']
        lookup={r['result']:r for r in actual}
        for row in expected:
            other=lookup[row['result']]
            same=row['success']==other['success'] and row['time_s']==other['time_s']
            comparisons.append(dict(result=row['result'],success_and_completion_time_identical=same))
        if not all(r['success_and_completion_time_identical'] for r in comparisons):
            write_json(target/'audit'/'reproduction_status.json',dict(complete=False,commands=log,comparisons=comparisons))
            raise RuntimeError('Evaluation differs from the retained reference; inspect comparison log')
    write_json(target/'audit'/'reproduction_status.json',dict(complete=True,created_utc=utc(),
        source_root=str(ROOT),python=sys.executable,full_grid_executed=args.execute,commands=log,comparisons=comparisons))
    print('Independent reproduction directory:',target)


if __name__=='__main__':main()

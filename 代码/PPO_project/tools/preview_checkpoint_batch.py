"""Run registered development checkpoints in separate, logged processes.

Each invocation records its literal argv, complete process wall time, exit
code and console output. Validation cannot consume a later training RNG state.
Existing snapshots and completed validations are preserved by their runners.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDIES = {'full_batch': 'preview_full_batch_study.py',
           'target_tracking': 'preview_target_tracking_study.py'}
MILESTONES = (0, 256, 512, 1024, 2048, 4096)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(argv, directory, index, source):
    log_path = directory / f'{index:03d}_stdout.log'
    started = time.perf_counter()
    utc = datetime.now(timezone.utc).isoformat()
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='strict')
        for line in process.stdout:
            log.write(line); log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    record = dict(argv=argv, cwd=str(ROOT), started_utc=utc,
        process_wall_s=time.perf_counter()-started, returncode=code,
        runner_sha256=source, output_file=log_path.name, output_sha256=sha(log_path))
    (directory / f'{index:03d}_command.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return record


def run(study, method, budget):
    if budget not in MILESTONES[1:]:
        raise ValueError('Choose a registered trained milestone')
    runner = ROOT / 'PPO_project/tools' / STUDIES[study]
    source = sha(runner)
    base = ROOT / '论文项目/revision_20260910_continued' / f'{study}_v1'
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = base / 'invocations' / f'{method}_{name}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records = []
    for step in MILESTONES:
        if step > budget: break
        if sha(runner) != source:
            raise ValueError('Runner changed during a registered execution')
        argv = [sys.executable, '-u', str(runner), 'train', '--method', method,
                '--seed', '101', '--decisions', str(step), '--stage', 'development']
        records.append(execute(argv, directory, len(records), source))
        snapshot = base / 'development' / f'preview_time_{method}_seed101/checkpoints/step{step}'
        if not (snapshot / 'validation/summary.json').exists():
            records.append(execute([sys.executable, '-u', str(runner), 'evaluate', '--run', str(snapshot)],
                                   directory, len(records), source))
    summary = dict(study=study, method=method, seed=101, available_budget=budget,
        commands=records, command_process_wall_s=sum(r['process_wall_s'] for r in records),
        scope='Process wall times include imports, preparation, training/evaluation and checkpoint storage; concurrent process totals are not elapsed wall time',
        old_completed_validation='Skipped and retained; its original cost is not included again')
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(directory, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES, required=True)
    parser.add_argument('--method', choices=('fixed1p2', 'fixed3', 'fixed4', 'adaptive'), required=True)
    parser.add_argument('--budget', type=int, choices=MILESTONES[1:], default=1024)
    args = parser.parse_args()
    run(args.study, args.method, args.budget)

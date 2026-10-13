"""Log complete subprocess costs for the registered independent formal study.

Training completes all ten chains before freezing. Held-out evaluation is a
separate explicit invocation. Numerical methods and selection live in their
existing runners; this module only schedules and records their exact commands.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '论文项目/revision_20260910_continued/formal_v1'
SEEDS = (11, 22, 33, 44, 55)
METHODS = ('fixed3', 'adaptive')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def execute(argv, directory, name):
    """Preserve even launch errors or interrupted subprocesses as invocations."""
    log_path = directory / (name + '.log')
    record_path = directory / (name + '.json')
    record = dict(argv=argv, cwd=str(ROOT), started_utc=datetime.now(timezone.utc).isoformat(),
                  status='running', output_file=log_path.name)
    write(record_path, record)
    print(json.dumps(dict(command=name, status='running'), ensure_ascii=False), flush=True)
    started = time.perf_counter()
    try:
        with log_path.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            record['pid'] = process.pid
            write(record_path, record)
            record['returncode'] = process.wait()
            record['status'] = 'complete' if record['returncode'] == 0 else 'failed'
    except BaseException as error:
        record.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        record.update(process_wall_s=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        if log_path.exists():
            record['output_sha256'] = sha(log_path)
        write(record_path, record)
        print(json.dumps(dict(command=name, status=record['status'],
                              process_wall_s=record['process_wall_s']), ensure_ascii=False), flush=True)
    return record


def run(phase, output, workers):
    if phase not in ('train', 'holdout') or not 1 <= workers <= 4:
        raise ValueError('Use train/holdout and one to four workers')
    output = Path(output).resolve()
    output.relative_to(ROOT / '论文项目/revision_20260910_continued')
    if phase == 'holdout' and not (output / 'freeze.json').is_file():
        raise ValueError('Complete and freeze all ten models before holdout evaluation')
    train_runner = ROOT / 'PPO_project/tools/preview_formal_continuation.py'
    eval_runner = ROOT / 'PPO_project/tools/preview_formal_holdout.py'
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = output / 'batch_invocations' / (phase + '_' + name)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    source = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), train_runner, eval_runner)}
    summary = dict(phase=phase, status='running', workers=workers, source_files=source,
        started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        commands=[], scope='Process costs include imports, training/evaluation and storage. Concurrent process time sums differ from batch elapsed wall time.')
    started = time.perf_counter()
    write(directory / 'summary.json', summary)

    def call(argv, label):
        if any(sha(ROOT / name) != digest for name, digest in source.items()):
            raise ValueError('Registered runner changed during batch')
        return execute(argv, directory, label)

    training_base = [sys.executable, '-u', str(train_runner), '--output-root', str(output)]
    try:
        if phase == 'train':
            prepared = call(training_base + ['prepare'], 'prepare')
            summary['commands'].append(prepared)
            if prepared['returncode']:
                raise RuntimeError('Formal preparation failed; inspect prepare.log')
        jobs = []
        for seed in SEEDS:
            for method in METHODS:
                if phase == 'train':
                    argv = training_base + ['chain', '--method', method, '--seed', str(seed)]
                else:
                    argv = [sys.executable, '-u', str(eval_runner), 'evaluate', '--freeze',
                            str(output / 'freeze.json'), '--method', method, '--seed', str(seed)]
                jobs.append((argv, f'{method}_seed{seed}'))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(call, argv, label) for argv, label in jobs]
            for future in as_completed(futures):
                summary['commands'].append(future.result())
                write(directory / 'summary.json', summary)
        if any(record['returncode'] for record in summary['commands']):
            raise RuntimeError('One or more formal jobs failed; retain logs and resume the same batch command')
        if phase == 'train':
            frozen = call(training_base + ['freeze'], 'freeze')
            summary['commands'].append(frozen)
            if frozen['returncode']:
                raise RuntimeError('Freeze failed; no holdout evaluation has been started')
        summary['status'] = 'complete'
    except BaseException as error:
        summary.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        summary.update(batch_wall_s=time.perf_counter() - started,
            command_process_wall_s=sum(item.get('process_wall_s', 0) for item in summary['commands']),
            finished_utc=datetime.now(timezone.utc).isoformat())
        write(directory / 'summary.json', summary)
    print(str(directory), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'holdout'), required=True)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    run(args.phase, args.output_root, args.workers)

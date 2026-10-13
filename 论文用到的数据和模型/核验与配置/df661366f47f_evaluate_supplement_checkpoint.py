"""Unconditional two-case/two-step evaluation of a fixed supplemental actor."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import multiprocessing as mp
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from train_localized_two_action_ppo import physics_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    verify_bound_inputs()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source').mkdir()
    started = time.perf_counter()
    deadline = started+1680.
    sources = [Path(__file__)] + [ROOT/name for name in (
        'train_mixed_resolution_ppo.py', 'train_localized_two_action_ppo.py',
        'research_policy_loader.py', 'localized_two_action_policy.py',
        'localized_joint_correction.py', 'localized_preview_head.py')]
    hashes = {str(p):sha(p) for p in [args.checkpoint, *sources]}
    for path in sources:
        shutil.copy2(path, args.output/'source'/path.name)
    write(args.output/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        source_sha256=hashes, wall_budget_s=1800, worker_deadline_s=1680,
        evidence_gap='Fixed supplemental checkpoint has no complete pair of cases at both physical steps.',
        procedure='Evaluate all four registered combinations from physical rest to full coverage and terminal rest. Record every failed prefix; no local-screen gate and no training.',
        adoption='Full success/physics and the unchanged all34/time requirements; independent-seed outcomes reported without replacing failures.',
        stop='All four evaluations, fatal worker error, or registered deadline.',
        new_ppo_updates=0, no_checkpoint_selection=True))
    ctx = mp.get_context('spawn')
    connections, processes, records = [], [], []
    error = None
    worker_args = SimpleNamespace(seed=args.seed, initial=args.checkpoint)

    def receive(connection):
        while time.perf_counter()<deadline:
            if connection.poll(1.):
                value = connection.recv()
                if 'success' not in value and isinstance(value.get('error'), str):
                    raise RuntimeError(value['error'])
                return value
        raise TimeoutError('Supplement evaluation deadline')

    try:
        for dt in [.001, .0005]:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=physics_worker, args=(child, dt, worker_args, args.output, deadline))
            proc.start()
            child.close()
            connections.append(parent)
            processes.append(proc)
        for connection in connections:
            receive(connection)
        for case in ['dense', 'original']:
            for connection in connections:
                connection.send(dict(kind='evaluate', case=case, checkpoint=str(args.checkpoint)))
            results = [receive(connection) for connection in connections]
            records.append(dict(case=case, results=results))
            write(args.output/'evaluations.json', records)
            print('SUPPLEMENT_EVALUATION', args.seed, case,
                  [(r['dt_s'],r['success'],r['complete_time_s'],r['reason']) for r in results], flush=True)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        for connection in connections:
            try:
                connection.send(dict(kind='stop'))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in processes:
            proc.join(timeout=5.)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.)
        for connection in connections:
            connection.close()
        complete = len(records)==2 and error is None
        write(args.output/'replay_receipt.json', dict(status='finished' if complete else 'error',
            exit_code=0 if complete else 1, wall_s=time.perf_counter()-started,
            all_four_evaluations_recorded=complete,
            all_four_complete_success=complete and all(r['success'] for item in records for r in item['results']),
            source_unchanged=all(sha(p)==h for p,h in hashes.items()), error=error,
            full_pipeline_complete=False, numerical_and_corner_audit_pending=True))


if __name__=='__main__':
    main()

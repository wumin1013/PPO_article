"""Resume missing physical checks, preserving and binding completed executions."""
import argparse
import multiprocessing as mp
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from fit_retained_basis_repair import read
from localized_joint_correction import physics_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', required=True)
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    verify_bound_inputs()
    stage = ROOT / args.name
    stage.mkdir(exist_ok=False)
    (stage / 'source').mkdir()
    started = time.perf_counter()
    deadline = started + 1680.
    prior = args.source.resolve()
    imported = {}
    checkpoints = {}
    for count in args.counts:
        path = prior / f'checkpoint_{count:04d}.pt'
        checkpoints[str(path)] = sha(path)
        shutil.copy2(path, stage / path.name)
        for case in ['dense', 'original']:
            for key in ['1ms', 'half_ms']:
                result_path = prior / f'eval_{count:04d}_{case}_{key}.json'
                if not result_path.exists():
                    continue
                result = read(result_path)
                trace = result_path.with_suffix('.npz')
                assert result['checkpoint_sha256'] == sha(path)
                assert trace.is_file() and result['trace_sha256'] == sha(trace)
                for suffix in ['.json', '.npz', '_gates.json', '_metrics.json']:
                    bound = result_path.with_name(result_path.stem + suffix)
                    if bound.exists():
                        shutil.copy2(bound, stage / bound.name)
                        imported[str(bound)] = sha(bound)
        for key in ['1ms', 'half_ms']:
            local = prior / f'local_check_{count:04d}_{key}.json'
            if local.exists():
                shutil.copy2(local, stage / local.name)
                imported[str(local)] = sha(local)
    sources = [Path(__file__)] + [ROOT / name for name in [
        'train_mixed_resolution_ppo.py', 'localized_joint_correction.py',
        'localized_preview_head.py', 'research_policy_loader.py',
        'fit_retained_basis_repair.py']]
    hashes = {str(path): sha(path) for path in sources}
    for path in sources:
        shutil.copy2(path, stage / 'source' / path.name)
    write(stage / 'registration.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Resume interrupted checkpoint selection; no new training',
        source_directory=str(prior), source_registration_sha256=sha(prior / 'registration.json'),
        imported_source_sha256=imported, checkpoints=checkpoints, source_sha256=hashes,
        wall_budget_s=1800, deadline_s=1680,
        evidence_gap='Prior process ended during user interruption; nominal candidate0 is complete while half-step and remaining checkpoints are missing.',
        procedure='Hash-verify and copy existing completed checks, retaining their original checkpoint/trace provenance. Simulate only missing checks with the same physics worker and screening order.',
        adoption='Unchanged complete paired protocol and extended all34 comparisons; an imported execution is not a new repeat.',
        stop='Registered candidates exhausted or deadline; terminate only owned workers.',
        new_ppo_updates=0, goal_complete=False, method_frozen=False))
    ctx = mp.get_context('spawn')
    connections, processes, records = [], [], []
    error = None
    worker_args = SimpleNamespace(seed=922302, initial=stage / f'checkpoint_{args.counts[0]:04d}.pt')

    def receive(connection):
        while time.perf_counter() < deadline:
            if connection.poll(1.):
                message = connection.recv()
                if 'success' not in message and isinstance(message.get('error'), str):
                    raise RuntimeError(message['error'])
                return message
        raise TimeoutError('Registered bounded resume deadline')

    def paired(count, kind, case=None):
        results = [None, None]
        missing = []
        for i, key in enumerate(['1ms', 'half_ms']):
            path = stage / (f'local_check_{count:04d}_{key}.json' if kind == 'local_check'
                            else f'eval_{count:04d}_{case}_{key}.json')
            if path.exists():
                results[i] = read(path)
            else:
                command = dict(kind=kind, checkpoint=str(stage / f'checkpoint_{count:04d}.pt'))
                if case is not None:
                    command['case'] = case
                connections[i].send(command)
                missing.append(i)
        for i in missing:
            results[i] = receive(connections[i])
        return results

    try:
        for dt in [.001, .0005]:
            parent, child = ctx.Pipe()
            process = ctx.Process(target=physics_worker, args=(child, dt, worker_args, stage, deadline))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        for connection in connections:
            receive(connection)
        for count in args.counts:
            if time.perf_counter() > deadline - 80.:
                break
            dense = paired(count, 'evaluate', 'dense')
            record = dict(update=count, dense=dense)
            if all(r['success'] for r in dense):
                local = paired(count, 'local_check')
                record['local'] = local
                if all(r['valid_count'] == 34 and r['budget_count'] == 34 for r in local) and time.perf_counter() < deadline - 400.:
                    record['original'] = paired(count, 'evaluate', 'original')
                    print('RESUMED_ORIGINAL', count, [(r['success'], r['complete_time_s']) for r in record['original']], flush=True)
            records.append(record)
            write(stage / 'evaluations.json', records)
            print('CHECKPOINT_FINISHED', count, flush=True)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        for connection in connections:
            try:
                connection.send(dict(kind='stop'))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=5.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.)
        for connection in connections:
            connection.close()
        write(stage / 'result.json', dict(evaluations=records, error=error, wall_s=time.perf_counter()-started,
            new_ppo_updates=0, goal_complete=False, method_frozen=False))
        write(stage / 'receipt.json', dict(status='finished' if error is None else 'error',
            sources_unchanged=all(sha(p) == h for p, h in hashes.items()),
            imported_sources_unchanged=all(sha(p) == h for p, h in imported.items()),
            wall_s=time.perf_counter()-started))


if __name__ == '__main__':
    main()

"""Bounded physical screening of saved PPO checkpoints; no new training."""
import argparse
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from train_mixed_resolution_ppo import physics_worker


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--counts',type=int,nargs='+',required=True)
    parser.add_argument('--localized-head',action='store_true')
    parser.add_argument('--localized-joint-head',action='store_true')
    parser.add_argument('--localized-two-head',action='store_true')
    parser.add_argument('--full-original-after-dense',action='store_true')
    args=parser.parse_args()
    assert Path(args.name).name==args.name
    verify_bound_inputs()
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1680.
    bindings={}
    for count in args.counts:
        path=args.source/f'checkpoint_{count:04d}.pt'
        bindings[str(path.resolve())]=sha(path)
        shutil.copy2(path,stage/path.name)
    sources=[Path(__file__),ROOT/'train_mixed_resolution_ppo.py',ROOT/'gated_head_policy.py']
    if args.localized_head:sources.append(ROOT/'localized_preview_head.py')
    if args.localized_joint_head:sources.extend([ROOT/'localized_joint_correction.py',ROOT/'localized_preview_head.py'])
    if args.localized_two_head:sources.extend([ROOT/'localized_two_action_policy.py',ROOT/'train_localized_two_action_ppo.py',ROOT/'localized_preview_head.py',ROOT/'research_policy_loader.py'])
    source_hashes={str(p):sha(p) for p in sources}
    for path in sources:shutil.copy2(path,stage/'source'/path.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Supplementary checkpoint selection using full physical executions',
        checkpoints=bindings,source_sha256=source_hashes,wall_budget_s=1800,deadline_s=1680,
        evidence_gap='Explicit derived local-support candidates require complete paired physical verification.' if args.localized_head or args.localized_joint_head else 'Scheduled checks8and16failed at half-ms;other saved policies not checked.',
        procedure=('Both dense resolutions then unconditional full original pair;baseline completion is not gated on beating main-method corner budgets.'
                   if args.full_original_after_dense else 'Both dense resolutions,then both all34local screens,then full original pair only if local screens pass.'),
        adoption='Original joint protocol including time numerical margin and all34comparison;no prefix or local success adoption.',
        stop='Registered checkpoints exhausted or wall deadline;do not alter policy or physical limits.',
        new_ppo_updates=0,goal_complete=False,method_frozen=False))
    worker_args=SimpleNamespace(seed=922302,initial=stage/f'checkpoint_{args.counts[0]:04d}.pt')
    worker=physics_worker
    if args.localized_head:
        from localized_preview_head import physics_worker as worker
    if args.localized_joint_head:
        from localized_joint_correction import physics_worker as worker
    if args.localized_two_head:
        from train_localized_two_action_ppo import physics_worker as worker
    ctx=mp.get_context('spawn');connections=[];processes=[];records=[];error=None
    def receive(connection):
        while time.perf_counter()<deadline:
            if connection.poll(1.):
                message=connection.recv()
                if 'success' not in message and isinstance(message.get('error'),str):raise RuntimeError(message['error'])
                return message
        raise TimeoutError('Registered30minute limit')
    try:
        for dt in [.001,.0005]:
            parent,child=ctx.Pipe();process=ctx.Process(target=worker,args=(child,dt,worker_args,stage,deadline))
            process.start();child.close();connections.append(parent);processes.append(process)
        for c in connections:receive(c)
        for count in args.counts:
            if time.perf_counter()>deadline-80.:break
            checkpoint=stage/f'checkpoint_{count:04d}.pt'
            for c in connections:c.send(dict(kind='evaluate',case='dense',checkpoint=str(checkpoint)))
            dense=[receive(c) for c in connections];record=dict(update=count,dense=dense)
            print('DENSE_SCREEN',count,[(r['success'],r['complete_time_s'],r['elapsed_prefix_s']) for r in dense],flush=True)
            if all(r['success'] for r in dense):
                if not args.full_original_after_dense:
                    for c in connections:c.send(dict(kind='local_check',checkpoint=str(checkpoint)))
                    record['local']=[receive(c) for c in connections]
                    print('LOCAL_SCREEN',count,[(r['valid_count'],r['budget_count']) for r in record['local']],flush=True)
                eligible=args.full_original_after_dense or all(r['valid_count']==34 and r['budget_count']==34 for r in record.get('local',[]))
                if eligible and time.perf_counter()<deadline-400.:
                    for c in connections:c.send(dict(kind='evaluate',case='original',checkpoint=str(checkpoint)))
                    record['original']=[receive(c) for c in connections]
                    print('ORIGINAL_SCREEN',count,[(r['success'],r['complete_time_s']) for r in record['original']],flush=True)
            records.append(record);write(stage/'evaluations.json',records)
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:
        for c in connections:
            try:c.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError,OSError):pass
        for p in processes:
            p.join(timeout=5.)
            if p.is_alive():p.terminate();p.join(timeout=5.)
        for c in connections:c.close()
        write(stage/'result.json',dict(evaluations=records,error=error,wall_s=time.perf_counter()-started,
            new_ppo_updates=0,goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(sources_unchanged=all(sha(p)==h for p,h in source_hashes.items()),
            wall_s=time.perf_counter()-started,status='finished' if error is None else 'error'))


if __name__=='__main__':main()

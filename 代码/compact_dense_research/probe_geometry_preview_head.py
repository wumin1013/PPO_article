"""Registered geometry-head parameter probe; zero new PPO updates."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import itertools
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from probe_paired_local_repair import INITIAL
from geometry_preview_head import GeometryPreviewPolicy,physics_worker
from train_mixed_resolution_ppo import policy_from


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    assert Path(args.name).name==args.name;verify_bound_inputs()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1680.
    variants=list(itertools.product([131.5,133.],[0.,.1,.2,.3]))
    sources=[Path(__file__),ROOT/'geometry_preview_head.py',ROOT/'train_mixed_resolution_ppo.py',ROOT/'gated_head_policy.py']
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Observable geometry preview-head finite parameter probe;notPPO',
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),source_sha256=hashes,wall_budget_s=1800,deadline_s=1680,
        variants=[dict(acute_mid_degrees=a,gains=[.1,.2,b]) for a,b in variants],inputs=14,outputs=3,policy_memory_scalars=0,
        evidence_gap='Uniform1.1fails06andnegative14/29;uniform1.3fails28. Need distinguish observable turn geometry and entry/exit state.',
        procedure='Frozen learned base steering/feed and preview plus explicit smooth three-coefficient preview head. No route identifier. Both dense resolutions then all34local,then original pair.',
        adoption='Joint protocol only;parameter probes are not trainedPPO or full evidence. Keep all failures.',
        stop='Eight registered variants or wall deadline;no bounds or baseline changes.',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False);base=policy_from(saved,922302)
    with np.load(INITIAL.parent/'dense.npz') as data:x=torch.tensor(data['observations'])
    for i,(angle,post) in enumerate(variants):
        policy=GeometryPreviewPolicy();policy.initialize(base)
        with torch.no_grad():
            policy.actor.gains.copy_(torch.tensor([.1,.2,post]));policy.actor.acute_mid_degrees.fill_(angle)
            assert torch.equal(policy.actor(x)[:,:2],base.actor(x)[:,:2])
        payload=dict(saved);payload.update(policy=policy.state_dict(),updates=i,interface=[14,3,922302],
            policy_class='GeometryPreviewPolicy',runtime_class='FarVectorRuntime',new_ppo_updates=0,
            checkpoint_index_kind='finite_parameter_probe_index',training_kind='Explicit parameter probe,notPPO',
            geometry_gains=[.1,.2,post],acute_mid_degrees=angle,base_checkpoint_sha256=sha(INITIAL))
        torch.save(payload,stage/f'checkpoint_{i:04d}.pt')
    ctx=mp.get_context('spawn');worker_args=SimpleNamespace(seed=922302,initial=stage/'checkpoint_0000.pt')
    connections=[];processes=[];records=[];error=None
    def receive(c):
        while time.perf_counter()<deadline:
            if c.poll(1.):
                r=c.recv()
                if 'success' not in r and isinstance(r.get('error'),str):raise RuntimeError(r['error'])
                return r
        raise TimeoutError('Registered30minute limit')
    try:
        for dt in [.001,.0005]:
            parent,child=ctx.Pipe();proc=ctx.Process(target=physics_worker,args=(child,dt,worker_args,stage,deadline))
            proc.start();child.close();processes.append(proc);connections.append(parent)
        for c in connections:receive(c)
        for i in range(len(variants)):
            if time.perf_counter()>deadline-90.:break
            checkpoint=stage/f'checkpoint_{i:04d}.pt'
            for c in connections:c.send(dict(kind='evaluate',case='dense',checkpoint=str(checkpoint)))
            dense=[receive(c) for c in connections];record=dict(update=i,dense=dense)
            print('GEOMETRY_DENSE',i,variants[i],[(r['success'],r['complete_time_s'],r['elapsed_prefix_s']) for r in dense],flush=True)
            if all(r['success'] for r in dense):
                for c in connections:c.send(dict(kind='local_check',checkpoint=str(checkpoint)))
                record['local']=[receive(c) for c in connections]
                print('GEOMETRY_LOCAL',i,[(r['valid_count'],r['budget_count']) for r in record['local']],flush=True)
                if all(r['valid_count']==34 and r['budget_count']==34 for r in record['local']) and time.perf_counter()<deadline-400.:
                    for c in connections:c.send(dict(kind='evaluate',case='original',checkpoint=str(checkpoint)))
                    record['original']=[receive(c) for c in connections]
                    print('GEOMETRY_ORIGINAL',i,[(r['success'],r['complete_time_s']) for r in record['original']],flush=True)
            records.append(record);write(stage/'evaluations.json',records)
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:
        for c in connections:
            try:c.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError,OSError):pass
        for proc in processes:
            proc.join(timeout=5.)
            if proc.is_alive():proc.terminate();proc.join(timeout=5.)
        for c in connections:c.close()
        write(stage/'result.json',dict(evaluations=records,error=error,new_ppo_updates=0,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(status='finished' if error is None else 'error',wall_s=time.perf_counter()-started,sources_unchanged=all(sha(p)==h for p,h in hashes.items())))


if __name__=='__main__':main()

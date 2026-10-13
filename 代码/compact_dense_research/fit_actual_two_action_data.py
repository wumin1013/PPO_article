"""Offline initialization from actual two-output successful actions; not PPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import json
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from train_mixed_resolution_ppo import physics_worker,TWO


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=922302)
    args=p.parse_args();assert Path(args.name).name==args.name
    import numpy as np
    import torch
    from direct_two_action_control import DirectTwoPolicy
    from learning_update_coherent_anchor import equivalent_loss
    verify_bound_inputs();torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1680.
    dataset=ROOT/'direct14_actual_two_action_dataset_round44/supervision.npz'
    audit=dataset.parent/'audit.json';assert json.loads(audit.read_text())['dataset_sha256']==sha(dataset)
    sources=[Path(__file__),ROOT/'train_mixed_resolution_ppo.py',ROOT/'direct_two_action_control.py']
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    checks=[1000,4000,12000,30000]
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Offline fitting to actual successful two-action requests',
        wall_budget_s=1800,deadline_s=1680,seed=args.seed,initial_checkpoint=str(TWO),initial_sha256=sha(TWO),
        dataset_sha256=sha(dataset),dataset_audit_sha256=sha(audit),source_sha256=hashes,fit_checks=checks,
        evidence_gap='Initial-heading conversion did not preserve20ms guide evolution;actual successful two-action requests now re-encoded exactly.',
        procedure='14-64-64-2actor,own fixed3mm guide selected previously. Fit executed steering/feed with saturation-equivalent targets. Full dense at both dt for all checkpoints.',
        adoption='Initialization only;best paired successful candidate gets fresh PPO. No claimed5percent advantage against failed comparison.',
        stop='Four checkpoints or30minute deadline;change strategy if no closed-loop success.',
        new_ppo_updates=0,policy_inputs=14,actor_outputs=2,policy_memory_scalars=0,
        independence='Conditional initializer from shared weights and historical demonstrations;not an independent PPO repetition.',goal_complete=False,method_frozen=False))
    saved=torch.load(TWO,map_location='cpu',weights_only=False);policy=DirectTwoPolicy(args.seed);policy.load_state_dict(saved['policy'])
    with np.load(dataset) as d:x=torch.tensor(d['observations']);y=torch.tensor(d['targets']);w=torch.tensor(d['weights']);groups=d['case']
    center=(x*w[:,None]).sum(0);scale=torch.sqrt(((x-center).square()*w[:,None]).sum(0)).clamp_min(.03)
    actor=copy.deepcopy(policy.actor)
    with torch.no_grad():
        raw=actor[0].weight.clone();actor[0].weight.mul_(scale);actor[0].bias.add_(raw@center)
        assert float(abs(actor((x-center)/scale)-policy.actor(x)).max())<3e-5
    optimizer=torch.optim.Adam(actor.parameters(),lr=2e-4)
    ctx=mp.get_context('spawn');worker_args=SimpleNamespace(seed=args.seed,initial=TWO)
    connections=[];processes=[];evaluations=[];log=[];error=None;step=0
    def receive(c):
        while time.perf_counter()<deadline:
            if c.poll(1.):
                r=c.recv()
                if 'success' not in r and isinstance(r.get('error'),str):raise RuntimeError(r['error'])
                return r
        raise TimeoutError('Registered30minute limit')
    try:
        for dt in [.001,.0005]:
            parent,child=ctx.Pipe();process=ctx.Process(target=physics_worker,args=(child,dt,worker_args,stage,deadline))
            process.start();child.close();processes.append(process);connections.append(parent)
        for c in connections:receive(c)
        normalized=(x-center)/scale;prob=w.numpy()/float(w.sum())
        for step in range(1,checks[-1]+1):
            if time.perf_counter()>deadline-90.:break
            ids=torch.from_numpy(rng.choice(len(x),size=512,replace=True,p=prob))
            loss=equivalent_loss(actor(normalized[ids]),y[ids],torch.tensor([.015,.15]))
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
            if step%500==0:log.append(dict(step=step,loss=float(loss.detach())))
            if step not in checks:continue
            exported=copy.deepcopy(actor)
            with torch.no_grad():
                exported[0].weight.div_(scale);exported[0].bias.sub_(exported[0].weight@center)
                export_error=float(abs(exported(x)-actor(normalized)).max())
            assert export_error<3e-5
            policy.actor.load_state_dict(exported.state_dict())
            checkpoint=stage/f'checkpoint_{step:04d}.pt';payload=dict(saved)
            payload.update(policy=policy.state_dict(),interface=[14,2,args.seed],updates=step,offline_iterations=step,
                checkpoint_index_kind='offline_fit_iteration',new_ppo_updates=0,training_kind='Offline actual two-action request fitting',
                initializer_sha256=sha(TWO),dataset_sha256=sha(dataset),export_equivalence_error=export_error,
                fit_optimizer=optimizer.state_dict(),normalized_actor=actor.state_dict(),center=center,scale=scale)
            torch.save(payload,checkpoint)
            for c in connections:c.send(dict(kind='evaluate',case='dense',checkpoint=str(checkpoint)))
            pair=[receive(c) for c in connections];record=dict(update=step,dense=pair)
            print('ACTUAL_TWO_FIT',step,[(r['success'],r['complete_time_s'],r['elapsed_prefix_s']) for r in pair],flush=True)
            evaluations.append(record);write(stage/'evaluations.json',evaluations);write(stage/'fit_log.json',log)
            if step==4000:
                for group in optimizer.param_groups:group['lr']=8e-5
            if step==12000:
                for group in optimizer.param_groups:group['lr']=3e-5
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:
        for c in connections:
            try:c.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError,OSError):pass
        for proc in processes:
            proc.join(timeout=5.)
            if proc.is_alive():proc.terminate();proc.join(timeout=5.)
        for c in connections:c.close()
        successful=[r for r in evaluations if all(v['success'] for v in r['dense'])]
        best=min(successful,key=lambda r:max(v['complete_time_s'] for v in r['dense'])) if successful else None
        write(stage/'result.json',dict(selected=best,evaluations=evaluations,offline_fit_steps=step,new_ppo_updates=0,
            error=error,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(status='finished' if error is None else 'error',wall_s=time.perf_counter()-started,
            sources_unchanged=all(sha(p)==h for p,h in hashes.items())))


if __name__=='__main__':main()

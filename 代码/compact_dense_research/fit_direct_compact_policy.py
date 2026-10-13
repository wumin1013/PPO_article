"""Bounded14-input direct-action compression of saved successful development traces."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,CASE,sha,write,verify_bound_inputs

STAGE=ROOT/'direct14_saved_motion_fit_seed922302_bound'
TEACHERS=[ROOT/'historical_fitted_residual_ppo_seed922302_original_full/full_trace.npz',
          ROOT/'paired_three_action_guarded_seed922302_round1/eval_0024.npz']


def worker():
    import numpy as np
    import torch
    from compact_control import CompactPolicy
    from direct_compact_runtime import DirectCompactRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    torch.set_num_threads(1);torch.manual_seed(922302);verify_bound_inputs()
    start=time.perf_counter();deadline=start+1080.
    observations=[];targets=[];weights=[];datasets=[]
    for index,path in enumerate(TEACHERS):
        d=np.load(path);x=d['observations'][:,:14].astype(np.float32)
        a=d['actions'];assert len(x)==len(a) and np.isfinite(x).all() and np.isfinite(a).all()
        y=(a/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])).astype(np.float32)
        # Preserve every observed transition and a fixed chronological subsample
        # of near-straight cruise. This changes only offline sample weighting.
        if index==0:
            transition=(np.max(abs(x[:,[6,8,10,12]]),axis=1)>.002)|(a[:,1]<99.)|(abs(a[:,0])>.001)
            transition[:100]=True;transition[-100:]=True
            cruise=np.flatnonzero(~transition)
            chosen=cruise[np.unique(np.linspace(0,len(cruise)-1,min(512,len(cruise))).astype(int))] if len(cruise) else cruise
            mask=transition.copy();mask[chosen]=True
            ids=np.flatnonzero(mask)
        else:ids=np.arange(len(x))
        observations.append(x[ids]);targets.append(y[ids]);weights.append(np.full(len(ids),.5/len(ids),dtype=np.float32))
        datasets.append(dict(trace=str(path),trace_sha256=sha(path),total_decisions=len(x),
            training_decisions=len(ids),retained_indices=ids.tolist(),weight_sum=.5))
    xx=np.vstack(observations);yy=np.vstack(targets);ww=np.concatenate(weights)
    np.savez_compressed(STAGE/'supervision.npz',observation=xx,direct_normalized_action=yy,sample_weight=ww)
    write(STAGE/'dataset.json',dict(sources=datasets,observation_dimension=14,actor_outputs=3,
        inherited_controller_at_execution=False,policy_memory_scalars=0,
        source_information='First14physical/geometric inputs; historical nominal3inputs excluded. Recorded actual held actions are direct targets.'))
    x,y,w=torch.tensor(xx),torch.tensor(yy),torch.tensor(ww)
    scale=torch.tensor([.015,.15,.3])
    policy=CompactPolicy(14,3,922302)
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=1e-3)
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    evaluations=[];fit_log=[];best=None
    for epoch in range(1,1601):
        if time.perf_counter()>deadline-45.:break
        prediction=policy.actor(x);loss=((((prediction-y)/scale)**2).mean(-1)*w).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
        if epoch%100==0:fit_log.append(dict(epoch=epoch,weighted_scaled_mse=float(loss.detach()),wall_s=time.perf_counter()-start))
        if epoch not in [200,600,1600]:continue
        checkpoint=STAGE/f'fit_{epoch:04d}.pt'
        torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,seed=922302,
            observation_dimension=14,action_dim=3,policy_memory_scalars=0,teacher_trace_sha256=[sha(p) for p in TEACHERS],
            training_kind='Offline direct-action compression, zero PPO updates'),checkpoint)
        result=evaluate(policy,DirectCompactRuntime(reference),STAGE/f'eval_{epoch:04d}.npz',deadline)
        if result['success']:
            d=np.load(STAGE/f'eval_{epoch:04d}.npz')
            result['analytic']=analytic_command_audit(reference[0],d['rows'],geometry)
            if not result['analytic']['analytic_corridor_pass']:result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        result.update(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            training_kind='Offline direct-action compression',ppo_updates=0,goal_complete=False,method_frozen=False)
        write(STAGE/f'eval_{epoch:04d}.json',result);evaluations.append(result)
        if result['success'] and (best is None or result['complete_time_s']<best['complete_time_s']):
            best=dict(checkpoint=str(checkpoint),complete_time_s=result['complete_time_s'])
        write(STAGE/'evaluations.json',evaluations);write(STAGE/'fit_log.json',fit_log)
        print(json.dumps(dict(epoch=epoch,success=result['success'],time_s=result['complete_time_s'],gates=result['geometry']['gates_passed'])),flush=True)
    write(STAGE/'result.json',dict(status='fit_completed' if epoch==1600 else 'wall_budget_completed',epochs=epoch,
        best=best,evaluations=[dict(epoch=e['epoch'],success=e['success'],time_s=e['complete_time_s'],gates=e['geometry']['gates_passed']) for e in evaluations],
        observation_dimension=14,actor_outputs=3,policy_memory_scalars=0,inherited_controller_at_execution=False,
        ppo_updates=0,training_kind='Offline compression from two developed teacher traces',
        wall_s=time.perf_counter()-start,goal_complete=False,method_frozen=False,
        scope='New compact candidate only; cannot replace historical method before complete two-case and comparative verification.'))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',ROOT.parent/'turn_aware_projector.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='14-input direct policy offline compression pilot',
        wall_budget_s=1200,child_timeout_s=1150,execution_deadline_s=1080,
        source_sha256=hashes,teacher_traces={str(p):sha(p) for p in TEACHERS},inherited_manifest=manifest,
        seed=922302,epochs=1600,evaluation_epochs=[200,600,1600],ppo_updates=0,
        actor='14inputs,2x64tanh,3direct outputs; no inherited model or nominal-action inputs at execution',
        objective='Weighted normalized action MSE with scales[.015,.15,.3]; each source case has equal weight; all original transition records plus512chronological cruise records retained.',
        adoption='Fit error is insufficient; require full unchanged dense physical/geometry success before continuation, then original-path and relative-efficiency checks before replacement.',
        stop='At fixed1600epochs or deadline; only3scheduled development evaluations, no model/seed/checkpoint expansion.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1150)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-2500:])


if __name__=='__main__':main()

"""Bounded offline imitation of observed forecast corrections, not PPO training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,HIST,CASE,sha,write,verify_bound_inputs

STAGE=ROOT/'historical_predictive_residual_fit_seed922302'
TEACHER=ROOT/'historical_dense_predictive_action_diagnostic'


def dataset():
    import numpy as np
    from turn_aware_projector import environment_factory
    from compact_control import CompactRuntime,local_progress
    data=np.load(TEACHER/'trace.npz')
    reference,rows,actions=data['reference'],data['rows'],data['actions']
    decisions=json.loads((TEACHER/'action_search.json').read_text(encoding='utf-8'))
    assert len(decisions)==len(actions)
    env=environment_factory(reference);env.dimension=14
    features=[];targets=[];maximum=0.
    norm=np.array([np.pi/2,50.,5.6]);offset=np.array([0.,1.,6.4/5.6])
    for i,record in enumerate(decisions):
        assert record['decision']==i
        if i:
            for row in rows[(i-1)*20:i*20]:env.tracker.update(row[:2])
            row=rows[i*20-1]
            env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
            env.heading=float(np.arctan2(row[9],row[8]));env.held=actions[i-1].copy()
        env.ticks=i*20;env.decisions=i
        difference=float(np.max(abs(env.observation()-data['observations'][i])))
        maximum=max(maximum,difference)
        local=CompactRuntime.observation(env)
        remaining=max(0.,env.corridor.arcs[-1]-local_progress(env))
        local[-1]=math.log1p(remaining/30.)/8.
        nominal=np.array(record['candidates'][0]['action'])/norm-offset
        features.append(np.r_[local,nominal])
        targets.append(actions[i]/norm-offset-nominal)
    assert maximum<1e-5,maximum
    x=np.asarray(features,dtype=np.float32);y=np.asarray(targets,dtype=np.float32)
    np.savez_compressed(STAGE/'supervision.npz',features=x,residual_targets=y)
    write(STAGE/'dataset_audit.json',dict(decisions=len(x),dimension=17,
        maximum_reconstructed_historical_observation_error=maximum,
        teacher_trace_sha256=sha(TEACHER/'trace.npz'),teacher_search_sha256=sha(TEACHER/'action_search.json'),
        nonzero_correction_rows=int(np.any(abs(y)>1e-7,axis=1).sum()),
        scope='Full recorded teacher trajectory; no withheld-validation or generalization claim.17inputs apply only to the new residual; whole method still inherits72inputs.'))
    return x,y,reference


def worker():
    import numpy as np
    import torch
    from historical_residual_ppo import HistoricalResidualRuntime,ResidualPolicy
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze
    verify_bound_inputs();torch.set_num_threads(1)
    seed=922302;torch.manual_seed(seed)
    start=time.perf_counter()
    xx,yy,reference=dataset()
    x=torch.tensor(xx);y=torch.tensor(yy)
    scale=torch.tensor([.05,.4,.4])
    policy=ResidualPolicy(seed=seed)
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=.001)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    evaluations=[];fit_log=[];best=None
    checkpoints=[200,600,1600]
    for epoch in range(1,1601):
        if time.perf_counter()>start+750.:break
        optimizer.zero_grad()
        prediction=policy.actor(x)
        loss=(((prediction-y)/scale)**2).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
        optimizer.step()
        if epoch%100==0:
            fit_log.append(dict(epoch=epoch,scaled_training_mse=float(loss),wall_s=time.perf_counter()-start))
        if epoch not in checkpoints:continue
        file=STAGE/f'fit_{epoch:04d}.pt'
        torch.save(dict(policy=policy.state_dict(),epoch=epoch,seed=seed,dimension=17,action_dim=3,
            training_kind='offline_supervised_residual_fit',ppo_updates=0,
            optimizer=optimizer.state_dict(),teacher_trace_sha256=sha(TEACHER/'trace.npz')),file)
        env=HistoricalResidualRuntime(reference)
        trace=STAGE/f'eval_{epoch:04d}.npz'
        result=evaluate(policy,env,trace,min(start+750.,time.perf_counter()+150.))
        rows=np.load(trace)['rows']
        result['analytic']=analytic_command_audit(reference[0],rows,geometry)
        if not result['analytic']['analytic_corridor_pass']:
            result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        result.update(training_kind='offline_supervised_residual_fit',ppo_updates=0,epoch=epoch,
            goal_complete=False,method_frozen=False,checkpoint=str(file))
        write(trace.with_suffix('.json'),result)
        if result['success']:
            write(STAGE/f'metrics_{epoch:04d}.json',analyze(trace,trace.with_suffix('.json')))
            if best is None or result['complete_time_s']<best['complete_time_s']:
                best=dict(checkpoint=str(file),complete_time_s=result['complete_time_s'],evaluation=str(trace.with_suffix('.json')))
        evaluations.append(result)
        write(STAGE/'evaluations.json',evaluations);write(STAGE/'fit_log.json',fit_log)
        print('FIT_EVALUATION',epoch,'success',result['success'],'gates',result['geometry']['gates_passed'],'time',result['complete_time_s'],flush=True)
    write(STAGE/'result.json',dict(kind='Offline residual imitation pilot',seed=seed,epochs=epoch,
        residual_observation_dimension=17,inherited_observation_dimension=72,
        independent_full_method_training=False,ppo_updates=0,best=best,
        evaluations=[dict(epoch=r['epoch'],success=r['success'],gates=r['geometry']['gates_passed'],complete_time_s=r['complete_time_s']) for r in evaluations],
        goal_complete=False,method_frozen=False,wall_s=time.perf_counter()-start,
        scope='Offline fitting on a single complete diagnostic trace. No PPO updates, no independent full-method repetition and no final dimensional-reduction acceptance.'))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();assert json.loads((TEACHER/'result.json').read_text(encoding='utf-8'))['success']
    STAGE.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'historical_residual_ppo.py',ROOT/'compact_control.py',ROOT/'local_arc_support.py']
    hashes={str(p):sha(p) for p in files};(STAGE/'source').mkdir()
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Single-seed offline residual imitation pilot',
        wall_budget_s=900,child_timeout_s=850,execution_deadline_s=750,ppo_updates=0,
        source_sha256=hashes,seed=922302,epochs=1600,evaluation_epochs=[200,600,1600],
        objective='Mean squared normalized action residual error, scales[.05,.4,.4], full recorded trajectory,Adam1e-3.',
        inherited_manifest=manifest,teacher_trace_sha256=sha(TEACHER/'trace.npz'),
        evidence_gap='Successful forecast corrections are computed online; a small learned correction has not reproduced the complete behavior.',
        adoption='Only complete physical/geometry/braking success qualifies a checkpoint. Training fit alone cannot qualify. Preserve all three scheduled evaluations.',
        stop='At1600 epochs or750s worker deadline; no seed or checkpoint expansion.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=850)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':main()

"""Condition existing14features during fitting; export an ordinary memoryless actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,HIST,CASE,sha,write,verify_bound_inputs

STAGE=ROOT/'direct14_geometry_conditioned_fit_seed922302_round3'
INITIAL=ROOT/'direct14_multicase_balanced_seed922302_round2/checkpoint_0016.pt'
TEACHERS={'original':ROOT/'historical_fitted_residual_ppo_seed922302_original_full/full_trace.npz',
          'dense':ROOT/'direct14_multicase_balanced_seed922302_round2/eval_0024_dense.npz'}
CASES={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
LIMITS={'original':444.531,'dense':15.213}


def worker():
    import numpy as np
    import torch
    from compact_control import CompactPolicy
    from direct_compact_runtime import DirectCompactRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    from analyze_dense_trace import analyze
    torch.set_num_threads(1);torch.manual_seed(922302)
    started=time.perf_counter();deadline=started+1380.
    xs=[];ys=[];weights=[];dataset=[]
    for case,path in TEACHERS.items():
        with np.load(path) as data:x=data['observations'][:,:14].astype(np.float32);a=data['actions']
        y=(a/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])).astype(np.float32)
        if case=='original':
            mask=(np.max(abs(x[:,[6,8,10,12]]),axis=1)>.002)|(a[:,1]<99.)|(abs(a[:,0])>.001)
            mask[:100]=True;mask[-100:]=True
            cruise=np.flatnonzero(~mask)
            if len(cruise):mask[cruise[np.unique(np.linspace(0,len(cruise)-1,min(512,len(cruise))).astype(int))]]=True
            ids=np.flatnonzero(mask)
        else:ids=np.arange(len(x))
        xs.append(x[ids]);ys.append(y[ids]);weights.append(np.full(len(ids),.5/len(ids),np.float32))
        dataset.append(dict(case=case,source=str(path),sha256=sha(path),indices=ids.tolist(),samples=len(ids),case_weight=.5))
    x=torch.tensor(np.vstack(xs));y=torch.tensor(np.vstack(ys));w=torch.tensor(np.concatenate(weights))
    center=(x*w[:,None]).sum(0);scale=torch.sqrt(((x-center).square()*w[:,None]).sum(0)).clamp_min(.02)
    normalized=(x-center)/scale
    policy=CompactPolicy(14,3,922302);saved=torch.load(INITIAL,weights_only=False);policy.load_state_dict(saved['policy'])
    actor=copy.deepcopy(policy.actor)
    with torch.no_grad():
        raw=actor[0].weight.clone();actor[0].weight.mul_(scale);actor[0].bias.add_(raw@center)
        initial_error=float((actor(normalized)-policy.actor(x)).abs().max())
    assert initial_error<2e-5
    optimizer=torch.optim.Adam(actor.parameters(),lr=2e-4)
    np.savez_compressed(STAGE/'supervision.npz',observation=x.numpy(),target=y.numpy(),weight=w.numpy(),center=center.numpy(),scale=scale.numpy())
    write(STAGE/'dataset.json',dict(sources=dataset,center=center.tolist(),scale=scale.tolist(),initial_coordinate_equivalence_error=initial_error))
    references={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in CASES.items()}
    geometries={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in CASES.items()}
    historical=json.loads((HIST/'full_metrics.json').read_text(encoding='utf-8'))
    old={c['id']:c for c in historical['corners']}
    fitting=[];evaluations=[];eligible=[]
    action_scale=torch.tensor([.015,.15,.3])
    for epoch in range(1,1801):
        if time.perf_counter()>deadline-35.:break
        loss=((((actor(normalized)-y)/action_scale).square()).mean(-1)*w).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if epoch%100==0:fitting.append(dict(epoch=epoch,loss=float(loss.detach()),wall_s=time.perf_counter()-started))
        if epoch not in [300,900,1800]:continue
        exported=copy.deepcopy(actor)
        with torch.no_grad():
            exported[0].weight.div_(scale);exported[0].bias.sub_(exported[0].weight@center)
            error=float((exported(x)-actor(normalized)).abs().max())
        assert error<3e-5
        policy.actor.load_state_dict(exported.state_dict())
        model=STAGE/f'fit_{epoch:04d}.pt'
        torch.save(dict(policy=policy.state_dict(),normalized_actor=actor.state_dict(),fit_optimizer=optimizer.state_dict(),
            center=center,scale=scale,epoch=epoch,seed=922302,observation_dimension=14,policy_memory_scalars=0,
            ppo_updates_this_stage=0,inherited_ppo_updates=16,training_kind='Offline geometry-conditioned action fitting after PPO initialization',
            initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),export_equivalence_error=error),model)
        results={};metric_original=None
        for case in ['dense','original']:
            env=DirectCompactRuntime(references[case]);trace=STAGE/f'eval_{epoch:04d}_{case}.npz'
            result=evaluate(policy,env,trace,deadline)
            with np.load(trace) as data:rows=data['rows']
            if len(rows):
                result['analytic']=analytic_command_audit(references[case][0],rows,geometries[case])
                if not result['analytic']['analytic_corridor_pass']:result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
            result.update(checkpoint=str(model),checkpoint_sha256=sha(model),observation_dimension=14,policy_memory_scalars=0,
                inherited_controller_at_execution=False,ppo_updates_this_stage=0,inherited_ppo_updates=16,fit_epoch=epoch,
                goal_complete=False,method_frozen=False)
            write(trace.with_suffix('.json'),result);write(STAGE/f'eval_{epoch:04d}_{case}_gates.json',env.tracker.events)
            if result['success']:
                m=metrics(references[case],rows,geometries[case]) if case=='original' else analyze(trace,trace.with_suffix('.json'))
                write(STAGE/f'eval_{epoch:04d}_{case}_metrics.json',m)
                if case=='original':metric_original=m
            results[case]=result
            print('FULL_CHECK',epoch,case,result['success'],result['complete_time_s'],result['reason'],flush=True)
        candidate_ok=all(results[k]['success'] and results[k]['complete_time_s']<=LIMITS[k]+1e-9 for k in CASES)
        candidate_ok=bool(candidate_ok and any(LIMITS[k]-results[k]['complete_time_s']>.02+1e-9 for k in CASES))
        deficit_change=None;not_worse=None
        if metric_original:
            delta=[c['speed_deficit_mm']-old[c['id']]['speed_deficit_mm'] for c in metric_original['corners']]
            deficit_change=sum(delta);not_worse=sum(d<=1e-7 for d in delta)
            candidate_ok=candidate_ok and deficit_change<11.148579193301657 and not_worse>=3
        record=dict(update=epoch,fit_epoch=epoch,checkpoint=str(model),results=results,eligible_for_half_ms=bool(candidate_ok),
            original_total_deficit_excess_over_historical_mm=deficit_change,original_not_worse_than_historical_count=not_worse)
        evaluations.append(record);write(STAGE/'evaluations.json',evaluations);write(STAGE/'fit_log.json',fitting)
        if candidate_ok:eligible.append(record)
    best=min(eligible,key=lambda e:(e['results']['original']['complete_time_s'],e['results']['dense']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='fit_completed' if epoch==1800 else 'wall_budget_completed',epochs=epoch,
        ppo_updates_this_stage=0,inherited_ppo_updates=16,observation_dimension=14,policy_memory_scalars=0,
        training_kind='Offline action fitting with invertible training-only conditioning; exported standard14-input2x64tanh actor',
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),selected_for_half_ms=best,adopted=False,
        retained_checkpoint=str(INITIAL),evaluations=evaluations,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',BASE/'paper_complete_evaluation.py',
        BASE/'natural_case_audit.py',BASE/'natural_case_metrics.py',ROOT/'analyze_dense_trace.py',ROOT/'refine_fitted_residual_ppo.py']
    hashes={str(p):sha(p) for p in files}
    for i,p in enumerate(files):shutil.copy2(p,STAGE/'source'/f'{i:02d}_{p.name}')
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Geometry-conditioned compact offline refit',
        wall_budget_s=1800,worker_deadline_s=1380,child_timeout_s=1450,optional_refined_timeout_s=290,
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),source_sha256=hashes,historical_manifest=manifest,
        teacher_trace_sha256={str(p):sha(p) for p in TEACHERS.values()},case_sha256={k:sha(p/'reference.csv') for k,p in CASES.items()},
        epochs=1800,evaluation_epochs=[300,900,1800],ppo_updates_this_stage=0,inherited_ppo_updates=16,
        objective='Same weighted normalized action MSE scales[.015,.15,.3],case weights0.5; each14feature centered/scaled with floor0.02 in training only; exact first-layer export conversion.',
        actor='Unchanged14input2x64tanh3output architecture; no teacher/memory at execution; Adam2e-4.',
        evidence_gap='PPO improved long cruise; remaining0.229s versus historical splits into0.098s path length,0.111s corner deficit and0.020s outside deficit. Faster checkpoint fails45degree corner6. Features vary in numerical scale.',
        adoption='Both full times<=444.531/15.213 and one gain>20ms; original summed corner deficit better than current candidate and >=3corners not worse than historical. Select shortest original then dense; require successful nonregressing dense0.5ms<=15.2315. This is a candidate screen only; full joint protocol remains required.',
        stop='Three predetermined fit checkpoints or wall deadline, all failures preserved; no extra fit nodes or architecture search.',
        joint_protocol=str(ROOT/'joint_lowdim_acceptance_20260922.json'),goal_complete=False))
    started=time.perf_counter();code=None;refined=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1450)
            code=run.returncode;status='finished'
        except subprocess.TimeoutExpired:status='timeout'
    if code==0:
        result=json.loads((STAGE/'result.json').read_text(encoding='utf-8'));best=result['selected_for_half_ms']
        if best and time.perf_counter()-started<1480.:
            name=STAGE.name+'_half_ms'
            command=[sys.executable,'-B',str(ROOT/'refine_fitted_residual_ppo.py'),'--name',name,'--checkpoint',best['checkpoint'],
                '--nominal',str(STAGE/f"eval_{best['fit_epoch']:04d}_dense.json"),'--actions','3','--direct-policy']
            with (STAGE/'refined_stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'refined_stderr.txt').open('w',encoding='utf-8') as err:
                try:refine=subprocess.run(command,stdout=out,stderr=err,timeout=290)
                except subprocess.TimeoutExpired:refine=None
            refined=ROOT/name/'result.json'
            if refine is not None and refine.returncode==0 and refined.exists():
                r=json.loads(refined.read_text(encoding='utf-8'));result['half_ms_result']=str(refined)
                result['adopted']=bool(r['success'] and r['complete_time_s']<=15.2315+1e-9)
                if result['adopted']:result['retained_checkpoint']=best['checkpoint']
            write(STAGE/'result.json',result)
    verify_bound_inputs()
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),half_ms_result=str(refined) if refined else None,goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()

"""Fit14features with30mm geometric horizon using audited saved-state reconstruction."""
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

STAGE=ROOT/'direct14_longview_fit_seed922302_round10'
INITIAL=ROOT/'direct14_speed_deficit_ppo_seed922302_round5/checkpoint_0024.pt'
TEACHERS={'original':ROOT/'longview_saved_data/original_teacher.npz',
          'dense':ROOT/'longview_saved_data/dense_retained.npz'}
CASES={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
LIMITS={'original':444.296,'dense':15.075}


def worker():
    import numpy as np
    import torch
    from compact_control import CompactPolicy
    from compact_longview_runtime import LongViewRuntime as DirectCompactRuntime
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
    original_target=y.clone();saturated=y[:,1]>=1.-1e-6;y[saturated,1]=1.04
    assert torch.allclose(y.clamp(-1,1),original_target.clamp(-1,1),atol=1e-6,rtol=0)
    center=(x*w[:,None]).sum(0);scale=torch.sqrt(((x-center).square()*w[:,None]).sum(0)).clamp_min(.02)
    normalized=(x-center)/scale
    policy=CompactPolicy(14,3,922302);saved=torch.load(INITIAL,weights_only=False);policy.load_state_dict(saved['policy'])
    actor=copy.deepcopy(policy.actor)
    with torch.no_grad():
        raw=actor[0].weight.clone();actor[0].weight.mul_(scale);actor[0].bias.add_(raw@center)
        initial_error=float((actor(normalized)-policy.actor(x)).abs().max())
    assert initial_error<2e-5
    optimizer=torch.optim.Adam(actor.parameters(),lr=2e-4)
    np.savez_compressed(STAGE/'supervision.npz',observation=x.numpy(),target=y.numpy(),original_target=original_target.numpy(),weight=w.numpy(),center=center.numpy(),scale=scale.numpy())
    write(STAGE/'dataset.json',dict(sources=dataset,center=center.tolist(),scale=scale.tolist(),initial_coordinate_equivalence_error=initial_error))
    references={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in CASES.items()}
    geometries={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in CASES.items()}
    historical=json.loads((HIST/'full_metrics.json').read_text(encoding='utf-8'))
    old={c['id']:c for c in historical['corners']}
    fitting=[];evaluations=[];eligible=[]
    action_scale=torch.tensor([.015,.15,.3])
    for epoch in range(1,12001):
        if time.perf_counter()>deadline-35.:break
        loss=((((actor(normalized)-y)/action_scale).square()).mean(-1)*w).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if epoch%100==0:fitting.append(dict(epoch=epoch,loss=float(loss.detach()),wall_s=time.perf_counter()-started))
        if epoch not in [3000,6000,12000]:continue
        exported=copy.deepcopy(actor)
        with torch.no_grad():
            exported[0].weight.div_(scale);exported[0].bias.sub_(exported[0].weight@center)
            error=float((exported(x)-actor(normalized)).abs().max())
        assert error<3e-5
        policy.actor.load_state_dict(exported.state_dict())
        model=STAGE/f'fit_{epoch:04d}.pt'
        torch.save(dict(policy=policy.state_dict(),normalized_actor=actor.state_dict(),fit_optimizer=optimizer.state_dict(),
            center=center,scale=scale,epoch=epoch,seed=922302,observation_dimension=14,policy_memory_scalars=0,
            ppo_updates_this_stage=0,inherited_ppo_updates=40,observation_features=DirectCompactRuntime(references['original']).feature_names(),far_sensor_mm=30,training_kind='Offline fitting of reconstructed30mm geometry after40inheritedPPOupdates; current stage has0PPOupdates',
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
                inherited_controller_at_execution=False,ppo_updates_this_stage=0,inherited_ppo_updates=40,fit_epoch=epoch,
                goal_complete=False,method_frozen=False,far_sensor_mm=30)
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
            candidate_ok=candidate_ok and deficit_change<=-2.826765315886917+1e-7 and not_worse>=30
        record=dict(update=epoch,fit_epoch=epoch,checkpoint=str(model),results=results,eligible_for_half_ms=bool(candidate_ok),
            original_total_deficit_excess_over_historical_mm=deficit_change,original_not_worse_than_historical_count=not_worse)
        evaluations.append(record);write(STAGE/'evaluations.json',evaluations);write(STAGE/'fit_log.json',fitting)
        if candidate_ok:eligible.append(record)
    best=min(eligible,key=lambda e:(e['results']['original']['complete_time_s'],e['results']['dense']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='fit_completed' if epoch==12000 else 'wall_budget_completed',epochs=epoch,
        ppo_updates_this_stage=0,inherited_ppo_updates=40,observation_dimension=14,policy_memory_scalars=0,
        training_kind='Offline30mm-far-point action fitting; exported standard14-input2x64tanh actor; existing near2/6mm points unchanged; zeroPPOupdates this stage',
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),selected_for_half_ms=best,adopted=False,
        retained_checkpoint=str(INITIAL),retained_far_sensor_mm=15,candidate_far_sensor_mm=30,
        evaluations=evaluations,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'compact_longview_runtime.py',ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',BASE/'paper_complete_evaluation.py',
        BASE/'natural_case_audit.py',BASE/'natural_case_metrics.py',ROOT/'analyze_dense_trace.py',ROOT/'refine_longview_compact_policy.py']
    hashes={str(p):sha(p) for p in files}
    for i,p in enumerate(files):shutil.copy2(p,STAGE/'source'/f'{i:02d}_{p.name}')
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='14input horizon reconstruction and offline policy fit',
        wall_budget_s=1800,worker_deadline_s=1380,child_timeout_s=1450,optional_refined_timeout_s=290,
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),source_sha256=hashes,historical_manifest=manifest,
        teacher_trace_sha256={str(p):sha(p) for p in TEACHERS.values()},case_sha256={k:sha(p/'reference.csv') for k,p in CASES.items()},
        reconstructed_data_audit=str(ROOT/'longview_saved_data/audit.json'),
        reconstructed_data_audit_sha256=sha(ROOT/'longview_saved_data/audit.json'),
        observation_dimension=14,policy_memory_scalars=0,far_sensor_mm=30,
        epochs=12000,evaluation_epochs=[3000,6000,12000],ppo_updates_this_stage=0,inherited_ppo_updates=40,
        objective='Weighted action MSE scales[.015,.15,.3],case weights0.5; saturated feed1mapped to equivalent1.04training target. Far sensor replaced by30mm point normalized30mm, no added input. Other12observations preserved and verified from raw saved states. Original source is successful historical full trace; dense source is successful round8gain1.25trace. Offline fit, notPPO.',
        actor='Unchanged14input2x64tanh3output architecture; no teacher/memory at execution; Adam2e-4.',
        evidence_gap='Current14input15mm-view candidate initiates turning about13mm before four deficient corners; historical teacher initiates near29mm. Earlier geometric cue may allow lower-curvature coordinated motion without increasing dimension. Reconstruction matches saved observations within2e-6and preserves12of14features. Test the30mmview by offline fitting and complete execution, no claim of successful control from reconstruction alone.',
        adoption='Both complete times<=444.296/15.075and one gain>20ms; original historical deficit change<=-2.826765mm and>=30nonworse corners. Select shortest original then dense. Require dense0.5ms full success and<=15.0745s. Full joint all34, credible historical time gain,5percent dense advantage, matched two-action control and repeatability remain mandatory.',
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
            command=[sys.executable,'-B',str(ROOT/'refine_longview_compact_policy.py'),'--name',name,'--checkpoint',best['checkpoint'],
                '--nominal',str(STAGE/f"eval_{best['fit_epoch']:04d}_dense.json"),'--feed-gain','1']
            with (STAGE/'refined_stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'refined_stderr.txt').open('w',encoding='utf-8') as err:
                try:refine=subprocess.run(command,stdout=out,stderr=err,timeout=290)
                except subprocess.TimeoutExpired:refine=None
            refined=ROOT/name/'result.json'
            if refine is not None and refine.returncode==0 and refined.exists():
                r=json.loads(refined.read_text(encoding='utf-8'));result['half_ms_result']=str(refined)
                result['adopted']=bool(r['success'] and r['complete_time_s']<=15.0745+1e-9)
                if result['adopted']:
                    result['retained_checkpoint']=best['checkpoint'];result['retained_far_sensor_mm']=30
            write(STAGE/'result.json',result)
    verify_bound_inputs()
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),half_ms_result=str(refined) if refined else None,goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()

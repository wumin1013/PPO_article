"""Declared distance/initialization opportunity for a genuine14input two-output arm."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha,verify_bound_inputs

STAGE=ROOT/'direct14_two_action_initialization_round27'
SOURCE=ROOT/'direct14_farvector_local_feedback_fit_round21'
INITIAL=SOURCE/'fit_08000.pt'
DISTANCES=[1.5,3.,4.5,6.,8.,12.]
SEED=922302


def dataset(distance,folder):
    import numpy as np
    from direct_two_action_control import DirectTwoRuntime
    from reconstruct_farvector_observations import set_physical
    from probe_joint_target_coordinates import guide_angle
    from joint_runtime import wrap
    pieces=[];targets=[];weights=[];reports=[]
    specs=[('dense',SOURCE/'eval_8000_dense.npz',SOURCE/'eval_8000_dense_gates.json',20,.30),
        ('dense_half',ROOT/'direct14_farvector_local_feedback_fit_round21_dense_half_ms/trace.npz',
         ROOT/'direct14_farvector_local_feedback_fit_round21_dense_half_ms/gate_events.json',40,.30),
        ('original',SOURCE/'eval_8000_original.npz',SOURCE/'eval_8000_original_gates.json',20,.40)]
    for case,path,gates,stride,weight in specs:
        with np.load(path) as data:
            ref=data['reference'];rows=data['rows'];x=data['observations'];actions=data['actions']
        events=json.loads(gates.read_text(encoding='utf-8'))
        crossing=np.array([e['command_segment'] for e in events])
        ids=np.arange(len(x))
        if case=='original':
            local=np.flatnonzero((abs(x[:,6])>.025)|(abs(x[:,13])>.015)|(np.linalg.norm(x[:,1:3],axis=1)<.995))
            anchors=np.linspace(0,len(x)-1,512).astype(int)
            ids=np.unique(np.r_[local,anchors,np.arange(100),np.arange(len(x)-100,len(x))])
        env=DirectTwoRuntime(ref,preview_rule_mm=distance)
        target=[];features=[];clips=0;max_error=0.;max_feature_error=0.
        for i in ids:
            tick=int(i*stride)
            if tick:
                set_physical(env.env,rows[tick-1])
            env.env.tracker.frontier=int(np.searchsorted(crossing,tick,side='left'))
            observed=env.observation()
            max_feature_error=max(max_feature_error,float(np.max(abs(observed-x[i]))))
            original=actions[i].copy();fixed=original.copy();fixed[2]=distance
            desired=wrap(guide_angle(env.env,original)+original[0])
            angle=wrap(desired-guide_angle(env.env,fixed))
            clips+=int(abs(angle)>=np.pi/2)
            fixed[0]=np.clip(angle,-np.pi/2+1e-9,np.pi/2-1e-9)
            max_error=max(max_error,abs(wrap(guide_angle(env.env,fixed)+fixed[0]-desired)))
            features.append(observed);target.append(fixed[:2]/np.array([np.pi/2,50.])-np.array([0.,1.]))
        assert max_feature_error<2e-6,(case,max_feature_error)
        pieces.append(np.asarray(features));targets.append(np.asarray(target));weights.extend([weight/len(ids)]*len(ids))
        reports.append(dict(case=case,samples=len(ids),weight=weight,source=str(path),source_sha256=sha(path),
            gates_sha256=sha(gates),maximum_feature_error=max_feature_error,
            steering_targets_outside_range=clips,maximum_initial_heading_target_error_rad=max_error))
    x=np.vstack(pieces).astype(np.float32);y=np.vstack(targets).astype(np.float32);w=np.asarray(weights,dtype=np.float32)
    np.savez_compressed(folder/'supervision.npz',observations=x,targets=y,weights=w)
    write(folder/'dataset_audit.json',dict(sources=reports,preview_mm=distance,observations=14,actor_outputs=2,
        scope='Offline initial-heading-equivalent targets on exposed complete teacher trajectories;not an actual two-action rollout and not PPO.'))
    return x,y,w


def worker():
    import numpy as np
    import torch
    from direct_two_action_control import DirectTwoPolicy,DirectTwoRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    torch.set_num_threads(1);torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    started=time.perf_counter();deadline=started+1640.
    geometries={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in
        [('dense',CASE),('original',BASE/'paper_evidence_review/natural_closed_case')]}
    references={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in
        [('dense',CASE),('original',BASE/'paper_evidence_review/natural_closed_case')]}
    common=torch.load(INITIAL,map_location='cpu',weights_only=False)['policy']
    records=[]

    def check(policy,distance,checkpoint,case,trace):
        env=DirectTwoRuntime(references[case],preview_rule_mm=distance)
        result=evaluate(policy,env,trace,deadline)
        with np.load(trace) as data:
            rows=data['rows'];assert data['latents'].shape[1]==2
            assert np.all(data['actions'][:,2]==distance)
        result['analytic']=analytic_command_audit(references[case][0],rows,geometries[case])
        if not result['analytic']['analytic_corridor_pass']:
            result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            observation_dimension=14,policy_memory_scalars=0,actor_outputs=2,global_preview_mm=distance,
            new_ppo_updates=0,initialization_kind='Shared14input starting network;two-output offline fit',
            inherited_ppo_updates=88,independent_full_method_training=False,goal_complete=False,method_frozen=False)
        write(trace.with_suffix('.json'),result)
        write(trace.parent/(trace.stem+'_gates.json'),env.tracker.events)
        if case=='original' and result['success']:
            write(trace.parent/(trace.stem+'_metrics.json'),metrics(references[case],rows,geometries[case]))
        return result

    for distance in DISTANCES:
        if time.perf_counter()>deadline-300.:break
        folder=STAGE/f'd_{distance:g}';folder.mkdir()
        xx,yy,ww=dataset(distance,folder)
        x=torch.tensor(xx);y=torch.tensor(yy);w=torch.tensor(ww)
        center=(x*w[:,None]).sum(0);scale=torch.sqrt(((x-center).square()*w[:,None]).sum(0)).clamp_min(.03)
        policy=DirectTwoPolicy(SEED);policy.initialize_common_layers(common)
        actor=copy.deepcopy(policy.actor)
        with torch.no_grad():
            raw=actor[0].weight.clone();actor[0].weight.mul_(scale);actor[0].bias.add_(raw@center)
            assert float(abs(actor((x-center)/scale)-policy.actor(x)).max())<3e-5
        optimizer=torch.optim.Adam(actor.parameters(),lr=2e-4)
        action_scale=torch.tensor([.015,.15]);log=[]
        for step in range(1,12001):
            if time.perf_counter()>deadline-240.:break
            ids=torch.from_numpy(rng.choice(len(x),size=512,replace=True,p=ww/ww.sum()))
            prediction=actor((x[ids]-center)/scale);target=y[ids]
            error=prediction-target
            error=torch.where(target>=1.-1e-6,-torch.relu(1.02-prediction),error)
            error=torch.where(target<=-1.+1e-6,torch.relu(prediction+1.02),error)
            loss=(error/action_scale).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
            if step%500==0:log.append(dict(step=step,loss=float(loss.detach())))
            if step not in (1500,5000,12000):continue
            exported=copy.deepcopy(actor)
            with torch.no_grad():
                exported[0].weight.div_(scale);exported[0].bias.sub_(exported[0].weight@center)
                export_error=float(abs(exported(x)-actor((x-center)/scale)).max())
            assert export_error<3e-5
            policy.actor.load_state_dict(exported.state_dict())
            checkpoint=folder/f'fit_{step:05d}.pt'
            torch.save(dict(policy=policy.state_dict(),offline_iterations=step,interface=[14,2,SEED],
                runtime_class='DirectTwoRuntime',policy_class='DirectTwoPolicy',preview_rule_mm=distance,
                source_checkpoint=str(INITIAL),source_checkpoint_sha256=sha(INITIAL),
                inherited_ppo_updates=88,new_ppo_updates=0,training_kind='Offline two-action conditional initialization',
                observation_features=DirectTwoRuntime(references['dense']).feature_names(),
                normalized_actor=actor.state_dict(),fit_optimizer=optimizer.state_dict(),center=center,scale=scale,
                export_equivalence_error=export_error),checkpoint)
            result=check(policy,distance,checkpoint,'dense',folder/f'dense_{step:05d}.npz')
            records.append(dict(distance_mm=distance,step=step,checkpoint=str(checkpoint),result=result))
            write(STAGE/'evaluations.json',records);write(folder/'fit_log.json',log)
            print('TWO_ACTION_CHECK',distance,step,result['success'],result['complete_time_s'],result['geometry']['gates_passed'],flush=True)
            if step<12000:
                for group in optimizer.param_groups:group['lr']*=.4
    successful=[r for r in records if r['result']['success']]
    best=min(successful,key=lambda r:r['result']['complete_time_s']) if successful else None
    original=None
    if best and time.perf_counter()<deadline-240.:
        checkpoint=Path(best['checkpoint']);policy=DirectTwoPolicy(SEED)
        policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'])
        original=check(policy,best['distance_mm'],checkpoint,'original',STAGE/'selected_original.npz')
    write(STAGE/'result.json',dict(selected=best,selected_original=original,evaluations=records,
        evaluated_distances=sorted({r['distance_mm'] for r in records}),new_ppo_updates=0,
        wall_s=time.perf_counter()-started,independent_full_method_training=False,goal_complete=False,method_frozen=False,
        next='Actual PPO and checkpoint opportunity;paired-resolution checks. This pilot is one conditional initialization,not three independent training seeds.'))


def main():
    if '--worker' in sys.argv:
        worker();return
    verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['direct_two_action_control.py','compact_farvector_runtime.py',
        'compact_control.py','reconstruct_farvector_observations.py']]+[BASE/'probe_joint_target_coordinates.py']
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Matched14input two-output initialization and declared preview selection',
        wall_budget_s=1800,worker_deadline_s=1640,child_timeout_s=1740,
        source_sha256=hashes,distances_mm=DISTANCES,checkpoints=[1500,5000,12000],seed=SEED,
        initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        evidence_gap='Existing strong two-action reference has a larger inherited representation;it cannot stand in for a controlled14input ablation.',
        procedure='Exactly14inputs,2outputs,two-dimensional Gaussian,same executor and limits. Copy common layers and first two outputs from verified main,then offline-fit feasible fixed-preview initial-heading targets. Give six declared distances three checkpoints each;use exposed cases only.',
        adoption='Shortest complete valid dense initialization selected for actual PPO;also check its complete original task. Never count failure-prefix time or call this completed ablation evidence.',
        stop='Six distances times three checkpoints or wall deadline;preserve all failures and historical budgets.',
        planned_training_seeds=[922302,922303,922304],seed_status='First conditional-initialization pilot only;other seeds have not run.',
        new_ppo_updates=0,goal_complete=False,method_frozen=False))
    started=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1740)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()

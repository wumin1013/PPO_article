"""Finite declared distance/fitting opportunity for the genuine two-output arm."""
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

STAGE=ROOT/'paired_two_action_initialization_seed922302'
TEACHER=ROOT/'historical_dense_predictive_action_diagnostic'
DISTANCES=[3.,4.,4.7782751045875855,6.,8.]


def dataset(distance,folder):
    import numpy as np
    from historical_two_action_residual import HistoricalTwoActionRuntime
    from probe_joint_target_coordinates import guide_angle
    from joint_runtime import wrap
    data=np.load(TEACHER/'trace.npz');reference,rows,actions=data['reference'],data['rows'],data['actions']
    runtime=HistoricalTwoActionRuntime(reference,preview_rule_mm=distance)
    env=runtime.env;features=[];targets=[];target_actions=[];clipped=0;max_direction_error=0.
    previous=np.array([0.,30.,distance])
    for i,teacher_action in enumerate(actions):
        if i:
            for row in rows[(i-1)*20:i*20]:env.tracker.update(row[:2])
            row=rows[i*20-1]
            env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
            env.heading=float(np.arctan2(row[9],row[8]));env.held=previous.copy()
        env.ticks=i*20;env.decisions=i
        x=runtime.observation()
        fixed=teacher_action.copy();fixed[2]=distance
        desired=wrap(guide_angle(env,teacher_action)+teacher_action[0])
        steer=wrap(desired-guide_angle(env,fixed))
        clipped+=int(abs(steer)>=np.pi/2)
        fixed[0]=np.clip(steer,-np.pi/2+1e-9,np.pi/2-1e-9)
        max_direction_error=max(max_direction_error,abs(wrap(guide_angle(env,fixed)+fixed[0]-desired)))
        features.append(x)
        targets.append(fixed[:2]/np.array([np.pi/2,50.])-np.array([0.,1.])-np.tanh(runtime.nominal_latent)[:2])
        target_actions.append(fixed);previous=fixed
    x=np.asarray(features,dtype=np.float32);y=np.asarray(targets,dtype=np.float32)
    np.savez_compressed(folder/'supervision.npz',features=x,residual_targets=y,physical_target_actions=np.asarray(target_actions))
    write(folder/'dataset_audit.json',dict(decisions=len(x),dimension=17,action_dim=2,global_preview_mm=distance,
        teacher_trace_sha256=sha(TEACHER/'trace.npz'),outside_steering_range_targets=clipped,
        maximum_initial_target_direction_error_rad=max_direction_error,
        scope='Same complete teacher trajectory as main initialization; legal two-action targets preserve the initial desired direction when within common steering range. This is offline supervision,not a simulated complete two-action run.'))
    return x,y,reference


def worker():
    import numpy as np
    import torch
    from historical_two_action_residual import HistoricalTwoActionRuntime,TwoActionResidualPolicy
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    verify_bound_inputs();torch.set_num_threads(1)
    start=time.perf_counter();records=[]
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    policy=TwoActionResidualPolicy();assert policy.actor[-1].out_features==2 and policy.log_std.shape==(2,)
    action,logp,value=policy.act(np.zeros(17,dtype=np.float32),True)
    assert action.shape==(2,) and np.isfinite(logp)
    write(STAGE/'interface_audit.json',dict(actor_output_features=2,Gaussian_dimension=2,raw_sample_shape=list(action.shape),
        inherited_backbone_shared=True,inherited_observation_dimension=72,residual_observation_dimension=17,
        independent_full_method_training=False,scope='Independent two-output residual initialization;common inherited backbone retained.'))
    for distance in DISTANCES:
        if time.perf_counter()>start+990.:break
        folder=STAGE/f'd_{distance:.6g}';folder.mkdir()
        xx,yy,reference=dataset(distance,folder)
        x=torch.tensor(xx);y=torch.tensor(yy);scale=torch.tensor([.05,.4])
        policy=TwoActionResidualPolicy(seed=922302)
        optimizer=torch.optim.Adam(policy.actor.parameters(),lr=.001)
        for epoch in range(1,1601):
            if time.perf_counter()>start+1050.:break
            optimizer.zero_grad();loss=(((policy.actor(x)-y)/scale)**2).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
            if epoch not in [200,600,1600]:continue
            checkpoint=folder/f'fit_{epoch:04d}.pt'
            torch.save(dict(policy=policy.state_dict(),epoch=epoch,seed=922302,dimension=17,action_dim=2,
                global_preview_mm=distance,training_kind='offline_supervised_two_action_residual_fit',ppo_updates=0,
                optimizer=optimizer.state_dict(),teacher_trace_sha256=sha(TEACHER/'trace.npz')),checkpoint)
            env=HistoricalTwoActionRuntime(reference,preview_rule_mm=distance)
            trace=folder/f'eval_{epoch:04d}.npz'
            result=evaluate(policy,env,trace,min(start+1050.,time.perf_counter()+120.))
            data=np.load(trace)
            assert data['latents'].shape[1]==2
            assert np.all(data['actions'][:,2]==distance)
            result['analytic']=analytic_command_audit(reference[0],data['rows'],geometry)
            if not result['analytic']['analytic_corridor_pass']:
                result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
            result.update(goal_complete=False,ppo_updates=0,global_preview_mm=distance,actor_output_dimension=2)
            write(trace.with_suffix('.json'),result)
            record=dict(distance_mm=distance,epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                scaled_training_mse=float(loss.detach()),result=result)
            records.append(record);write(STAGE/'evaluations.json',records)
            print('TWO_ACTION_INIT',distance,epoch,result['success'],result['geometry']['gates_passed'],result['complete_time_s'],flush=True)
    successes=[r for r in records if r['result']['success']]
    selected=min(successes,key=lambda r:r['result']['complete_time_s']) if successes else max(records,key=lambda r:(r['result']['geometry']['gates_passed'],-r['scaled_training_mse']))
    write(STAGE/'result.json',dict(kind='Finite two-output distance and initialization selection;zero PPO updates',
        selected=selected,evaluated_checkpoints=len(records),evaluated_distances=sorted({r['distance_mm'] for r in records}),
        complete_success_count=len(successes),ppo_updates=0,wall_s=time.perf_counter()-start,goal_complete=False,
        method_frozen=False,independent_full_method_training=False,
        next='Train this independently initialized two-output residual with the registered main-arm PPO decision budget; preserve full inherited-budget disclosure.'))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'historical_two_action_residual.py',ROOT/'historical_residual_ppo.py',ROOT/'compact_control.py',ROOT/'local_arc_support.py',BASE/'probe_joint_target_coordinates.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Genuine two-output conditional-ablation initialization',
        wall_budget_s=1200,child_timeout_s=1150,execution_deadline_s=1050,ppo_updates=0,
        global_preview_candidates_mm=DISTANCES,fitting_epochs=[200,600,1600],seed=922302,
        source_sha256=hashes,inherited_manifest=manifest,teacher_trace_sha256=sha(TEACHER/'trace.npz'),
        objective='Same full-teacher offline MSE initialization protocol;only two action residuals sampled/fit. Adam1e-3,scales[.05,.4],2x64tanh.',
        information_scope='Same17-input local/nominal information and shared72-input inherited backbone; removes preview control,not forward sensing. No claim of independent full historical model training.',
        evidence_gap='The prior14-input two-action model uses a different representation and history,so is not a controlled ablation of the new inherited method.',
        selection='Shortest complete feasible initialization first; if none complete,most witnessed gates then lowest normalized fit error is only a PPO starting point.',
        stop='Five distances and three fixed checkpoints each,or deadline. No extension to optimize fixed preview indefinitely.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1150)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-5000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()

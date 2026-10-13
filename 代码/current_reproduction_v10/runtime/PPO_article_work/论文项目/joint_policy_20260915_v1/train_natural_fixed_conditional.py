"""Bounded strong fixed-preview continuation over the same learned backbone."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time
import numpy as np
import torch
from study_io import BASE,sha,register,receipt,write_json
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_bend_residual_policy import BendResidualPolicy
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit
from natural_case_metrics import metrics


def main():
    torch.set_num_threads(1);case=BASE/'paper_evidence_review/natural_closed_case'
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    model=BASE/'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    upper=json.loads((case/'support_calibration/results.json').read_text(encoding='utf-8'))[0]['spec']['upper']
    parent=torch.load(model,map_location='cpu',weights_only=False)
    selected_ids=geometry['mechanism_columns']
    local=[]
    for corner in geometry['sharp_corners']:
        k=corner['reference_vertex_index'];v=reference[k];u=v-reference[k-1];u/=np.linalg.norm(u)
        w=reference[k+1]-v;w/=np.linalg.norm(w)
        # A complete100mm training path with the actual registered angle and a
        # common approach length, independent of candidate outcome.
        ref=np.array([v-50*u,v-25*u,v,v+25*w,v+50*w])
        local.append(dict(id=corner['id'],reference=ref))
    grid=[.8,1.5,3.,4.6,6.,8.,12.]
    stage=register('paper_evidence_review/natural_closed_case/fixed_conditional_training',
        [__file__,model,case/'support_calibration/receipt.json',case/'geometry.json',case/'reference.csv',
         BASE/'paper_bend_residual_policy.py',BASE/'paper_complete_evaluation.py'],
        dict(total_wall_s=1800,screen_wall_s=450,shortlist_wall_s=350,full_run_wall_s=950,
            global_d_grid_mm=grid,refinement_candidates=10,training_corner_ids=selected_ids,
            shortlist_all34_max=2,full_lap_max=1,conditional_seed=921101,ppo_updates=0),
        'Stronger fixed-preview control: same backbone, support transform, runtime and limits. '
        'Select one global D over seven values on three prespecified representative complete '
        '100mm training paths. Give10 shared steering/feed residual candidates; no per-angle '
        'parameters. Screen up to2 best unique candidates on all34 complete angle snippets, '
        'then evaluate one selected immutable policy on the full unchanged development case. '
        'This is closed-loop parameter training, not independent PPO replication or a fair '
        'end-to-end budget claim. Retain every failure and stop at the registered wall cap.')
    start=time.perf_counter();deadline=start+1750;rng=np.random.default_rng(921101)
    trials=[];costs=[];best=None
    train=[x for x in local if x['id'] in selected_ids]
    def rollout(spec,items,label,limit):
        records=[]
        for item in items:
            if time.perf_counter()>limit:break
            path=stage/(label+'_'+item['id']+'.npz')
            policy=BendResidualPolicy(load_credit_policy(parent),upper,spec['coefficients'],'fixed_preview')
            env=environment_factory(item['reference']);env.fixed_d=spec['fixed_d']
            r=evaluate(policy,env,path,limit,max_calls=1200)
            r.update(corner=item['id'],trace=str(path.relative_to(BASE)));records.append(r);costs.append(r)
        if len(records)!=len(items):return records,[999,999,999]
        failures=sum(not r['success'] for r in records)
        progress=sum(1-r['geometry']['gates_passed']/max(1,r['geometry']['gate_count']) for r in records if not r['success'])
        return records,[failures,progress,sum(r['complete_time_s'] or 0 for r in records)]
    for i in range(17):
        if time.perf_counter()>start+450:break
        if i<7:spec=dict(fixed_d=grid[i],coefficients=[0.,0.,0.])
        else:
            theta=np.asarray(best['spec']['coefficients'])+rng.normal(size=3)*np.array([.055,.6,0.])
            if i==7:theta=np.array([-.025,-.6,0.])
            theta=np.clip(theta,[-.2,-2.5,0.],[.2,1.,0.])
            spec=dict(fixed_d=best['spec']['fixed_d'],coefficients=theta.tolist())
        records,key=rollout(spec,train,'candidate'+str(i),min(start+450,deadline))
        r=dict(trial=i,spec=spec,key=key,records=records);trials.append(r)
        if best is None or key<best['key']:best=r
        write_json(stage/'training.json',dict(trials=trials,best=best['trial']))
        print('FIXED_TRAIN',i,spec,key,flush=True)
    finalists=[]
    for r in sorted(trials,key=lambda r:r['key']):
        if len(finalists)>=2 or time.perf_counter()>start+800:break
        if any(x['spec']==r['spec'] for x in finalists):continue
        records,key=rollout(r['spec'],local,'shortlist'+str(len(finalists)),min(start+800,deadline))
        q=dict(spec=r['spec'],key=key,records=records,from_trial=r['trial']);finalists.append(q)
        write_json(stage/'shortlist.json',finalists)
        print('FIXED_SHORTLIST',r['trial'],key,flush=True)
    chosen=min(finalists,key=lambda x:x['key']) if finalists else best
    spec=dict(chosen['spec'],upper=upper,backbone=str(model.relative_to(BASE)),backbone_sha256=sha(model),
              independent_ppo_training=False,conditional_parameter_seed=921101)
    write_json(stage/'selected_policy.json',spec)
    # Full-lap data are never used for another candidate in this bounded round.
    policy=BendResidualPolicy(load_credit_policy(parent),upper,spec['coefficients'],'fixed_preview')
    env=environment_factory(reference);env.fixed_d=spec['fixed_d']
    result=evaluate(policy,env,stage/'full_trace.npz',deadline)
    costs.append(result)
    if result['success']:
        z=np.load(stage/'full_trace.npz')
        result['analytic']=analytic_command_audit(reference[0],z['rows'],geometry)
        result['success']=result['analytic']['analytic_corridor_pass']
        if not result['success']:result['complete_time_s']=None
        write_json(stage/'metrics.json',metrics(reference,z['rows'],geometry))
    write_json(stage/'full_result.json',result)
    receipt(stage,dict(status='completed_fixed_conditional_training_and_execution',wall_s=time.perf_counter()-start,
        trials=len(trials),full_success=result['success'],full_time_s=result['complete_time_s'],
        ppo_updates=0,closed_loop_parameter_candidates=len(trials),independent_ppo_seeds=0,
        actor_calls=sum(r['actor_calls'] for r in costs),physics_ticks=sum(r['physics_ticks'] for r in costs),
        selected_policy_sha256=sha(stage/'selected_policy.json'),goal_complete=False))
    print('FIXED_CONDITIONAL_COMPLETE',result['success'],result['elapsed_prefix_or_complete_s'],flush=True)


if __name__=='__main__':main()

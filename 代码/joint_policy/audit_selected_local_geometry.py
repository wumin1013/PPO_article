"""Deterministic full training-pool audit before wider development testing."""
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from frame_feedback_policy import FrameFeedbackPolicy
from probe_constrained_parameter_step import summarize
from train_pilot import evaluate
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'local_geometry_continuous_reward_ppo_20260916';model=source/'policy_18645.pt'
    parent=BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt'
    stage=register('local_geometry_full_training_audit_20260916',[__file__,model,source/'receipt.json',source/'results.json',
        parent,BASE/'local_geometry_feedback_policy.py',BASE/'frame_feedback_policy.py',BASE/'learning_preparation_v4/paths.json'],
        dict(candidate_paths=64,parent_checks='candidate failure paths only',actor_calls=9000,physics_ticks=210000,wall_s=180),
        'Evaluate the retained12/12 development checkpoint on every existing training geometry, deterministic '
        'complete runs. Training-pool evaluation is not independent generalization. Audit full physical constraints '
        'and actual signed shape; report95/98/99mm/s duration, keeping failures separate. Only if candidate '
        'failures occur, evaluate preserved original broad-turn parent on those paths to distinguish inherited '
        'difficulty from regressions; no repeat of known original16 parent evaluations. Do not launch a long '
        'route if this pool audit reveals failures. No model update or sealed tests.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);policy.load_state_dict(saved['policy'])
    paths=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['training']]
    results=evaluate(policy,paths,stage,'training_pool',None,costs,started+180,environment_factory)
    report=summarize(stage,'training_pool',results,list(range(len(paths))))
    for i,row in enumerate(report['paths']):
        with np.load(stage/'training_pool'/f'{i}.npz') as z:speed=np.linalg.norm(z['rows'][:,2:4],axis=1)
        row.update(cruise_ge98_s=float((speed>=98).sum()*.001),cruise_ge99_s=float((speed>=99).sum()*.001))
    write_json(stage/'candidate_results.json',dict(candidate=report,costs=costs))
    failures=[r['path'] for r in report['paths'] if not r['success']]
    original_ids=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    reused=[];unknown=[]
    for index in failures:
        if index in original_ids:
            r=read(parent.parent/'training_3'/f'{original_ids.index(index)}.json')
            reused.append(dict(path=index,reused=True,success=r['success'],source=f'constrained_parameter_step_probe_20260916/training_3/{original_ids.index(index)}.json'))
        else:unknown.append(index)
    checks=[]
    if unknown:
        baseline=FrameFeedbackPolicy(frame='heading',history_independent=True)
        baseline.load_state_dict(torch.load(parent,map_location='cpu',weights_only=False)['policy'])
        checks=evaluate(baseline,[paths[i] for i in unknown],stage,'parent_failure_checks',None,costs,started+180,environment_factory)
        for row,index in zip(checks,unknown):row['path']=index
    assert costs['evaluation_actor_calls']<=9000 and costs['evaluation_physics_ticks']<=210000
    write_json(stage/'results.json',dict(candidate=report,failure_paths=failures,parent_checks=checks,parent_reused=reused,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_full_training_pool_audit',model_sha256=sha(model),candidate_successes=report['successes'],
                        failure_paths=failures,costs=costs,wall_s=time.perf_counter()-started))
    print('FULL_POOL',report['successes'],'/',len(paths),'failures',failures,'parent',[(r['path'],r['success']) for r in checks+reused],flush=True)


if __name__=='__main__':main()

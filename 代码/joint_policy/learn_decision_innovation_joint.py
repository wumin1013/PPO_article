"""Full-policy credit for a history-aware joint-action representation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from decision_innovation_policy import DecisionInnovationPolicy, metadata
from canonical_gate_state import checkpoint_environment_factory
from learn_nonlinear_joint_feedback import rollout, promotion
from learn_full_pool_joint_credit import aggregate
from learn_retained_joint_growth import smooth_cruise
from learn_absolute_shape_joint_credit import absolute_actual
from learn_pathwise_full_credit import pathwise_response, actual_selection, validation
from learn_full_pool_compensation import solve_response, progress
from probe_coordinated_suffix import reconstruct
from train_measured_joint_distribution import measured_score
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


WORK={}


def initialize(stage, parent, references, curved, retained, deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,parent=parent,references=references,curved=curved,
                retained=retained,deadline=deadline,
                factory=checkpoint_environment_factory('gate_band_canonical_kernel'))


def evaluate(job):
    tag,parameters=job
    policy=DecisionInnovationPolicy(WORK['parent'],parameters)
    def factory(ref):return policy.bind(WORK['factory'](ref))
    folder=WORK['stage']/tag;folder.mkdir()
    records=[];costs=dict(actor_calls=0,physics_ticks=0)
    for path,reference in WORK['references'].items():
        r,rows,actions,observations=rollout(policy,factory,reference,costs,WORK['deadline'])
        file=folder/f'path{path}.npz'
        np.savez_compressed(file,reference=reference,rows=rows,actions=actions,observations=observations)
        r.update(path=path,trace=file.relative_to(WORK['stage']).as_posix(),curved_path=WORK['curved'][path])
        if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
        records.append(r)
    summary=aggregate(records)
    checks=promotion(records,WORK['retained'])
    checks['curved_cruise_present']=bool(summary['successes']==64 and summary['curved_cruise99_s']>0.)
    result=dict(tag=tag,parameters=np.asarray(parameters).tolist(),records=records,summary=summary,
                costs=costs,promotion=checks,
                primary_preserved=all(checks[k] for k in ('all_complete','inward_first_turns','entry_nonworse')))
    result['absolute_objective']=absolute_actual(result,WORK['retained']) if summary['successes']==64 else None
    write_json(WORK['stage']/(tag+'.json'),result)
    return result


def preflight(stage,parent,baseline,factory):
    costs=dict(saved_prefix_geometry_ticks=0)
    rows=[]
    for path,frame in ((8,580),(12,660),(14,300),(15,660)):
        record=next(r for r in baseline['records'] if r['path']==path)
        with np.load(record['trace']) as z:
            ref,physical,actions=[z[k].copy() for k in ('reference','rows','actions')]
        env=reconstruct(ref,physical,actions,frame,costs,env_factory=factory)
        policy=DecisionInnovationPolicy(parent,np.zeros(9));policy.bind(env)
        latent=policy.act(env.observation())[0]
        assert np.array_equal(env.decode_action(latent),actions[frame//20])
        features,inputs=policy.last_features.copy(),policy.last_inputs.copy()
        _,_,_,info=env.step(latent)
        assert np.array_equal(info['rows'],physical[frame:frame+20])
        rows.append(dict(path=path,frame=frame,inputs=inputs.tolist(),features=features.tolist(),
                         zero_action_bitwise=True,zero_next20ms_bitwise=True))
    write_json(stage/'identity_and_feature_audit.json',dict(states=rows,costs=costs,
        inputs='Sine of new-versus-previous combined target at the same XY; sine of target-versus-filtered heading; lateral actual velocity in filtered-heading frame / max(speed,5).',
        parity='Steering odd; feed and preview even. All coefficients initially zero.',
        new_state_memory='Previous actually held action from the existing environment state.',
        action_time_structure='Unchanged zero-order hold; no interpolation or prescribed smoothing.'))


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    parent_file=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    assert sha(parent_file)=='5ab47111bfcec3a511903e20f1249c91bbfb65e5ab4cd1a3d2b4a493a979c26a'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    stage=register('canonical_gate_band_decision_innovation_joint_learning_20260919',
        [__file__,BASE/'decision_innovation_policy.py',BASE/'learn_pathwise_full_credit.py',
         BASE/'learn_full_pool_compensation.py',BASE/'learn_nonlinear_joint_feedback.py',
         BASE/'joint_runtime.py',BASE/'gate_band_state.py',BASE/'probe_joint_target_coordinates.py',
         parent_file,source/'baseline.json',
         BASE/'canonical_gate_band_measured_joint_distribution_learning_20260919/receipt.json',
         BASE/'canonical_gate_band_measured_joint_distribution_learning_20260919/results.json',
         BASE/'canonical_gate_band_full_pool_step_refinement_20260919/current_shape_timing.json'],
        dict(wall_s=900,workers=4,joint_parameters=9,axis_amplitude=1.,
             full64_axis_candidates=18,maximum_joint_candidates=2,identity_states=4,
             unchanged_decision_ms=20,unchanged_execution_ms=1),
        'The completed36-candidate measured distribution search yielded no acceptable model; '
        'best exploratory actual reversal16.058deg still fails per-corner checks. Current actors '
        'exclude prior-action inputs, while saved traces show large combined-target changes. '
        'Old one-action preview/steering compensation probes had mixed complete consequences, '
        'so no fixed compensation or temporal smoother is deployed. Change only the learnable '
        'state-function representation: nine zero-initialized joint coefficients of observed '
        'target innovation, heading error and physical lag. Cubic odd steering and squared even '
        'feed/preview features permit learned cross-action credit with reflection symmetry. '
        'Three output scales are0.1,0.2,0.2 latent units; coefficient signs are learned from '
        'actual complete64 rollouts. Retain parent actions exactly at zero and verify four saved '
        '20ms windows. No new speed/angle/path/time rule, guide change, interpolation, labels '
        'or relaxed constraints. Measure +/-nine axes, then a pathwise-complete signed-trajectory '
        'joint response with the unchanged per-corner,aggregate,time,cruise and geometry guards. '
        'At most half/full proposals; fail/non-descending directions end this representation '
        'round rather than repeat training. Selection remains unchanged, and formal86cd/best5ab '
        'are preserved. This is complete-policy parameter optimization, not PPO; no sealed test.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'学习具有目标更新与运动滞后观测的三动作残差，完整64路径九方向对照，决策执行周期及采用条件不变。')
    try:
        write_json(stage/'baseline.json',data)
        factory=checkpoint_environment_factory('gate_band_canonical_kernel')
        preflight(stage,parent,baseline,factory)
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,refs,curved,retained,started+885))
        jobs=[(f'axis{i}_{s:+d}',s*np.eye(9)[i]) for i in range(9) for s in (-1,1)]
        for c in pool.imap_unordered(evaluate,jobs):
            c['measured_score']=measured_score(c,baseline,retained);outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('INNOVATION_AXIS',c['tag'],c['summary'],flush=True)
        indexed={c['tag']:c for c in outcomes}
        paired=[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(9)]
        response,audit=pathwise_response(stage,baseline,retained,paired)
        write_json(stage/'response_eligibility.json',audit)
        torch.save(dict(response=response,baseline=baseline,retained=retained),stage/'response_model.pt')
        # Max per-head correction <= twice its one-feature scale, on the whole feature cube.
        trust=np.zeros((24,9));row=0
        import itertools
        for head in range(3):
            for sign in itertools.product((-1.,1.),repeat=3):
                trust[row,np.arange(3)*3+head]=sign;row+=1
        step,report=solve_response(baseline,retained,response,trust,True,
            min(started+820,time.perf_counter()+80),aggregate_guards=True)
        write_json(stage/'optimizer.json',report)
        print('INNOVATION_SOLVER',report['message'],report['predicted_objective'],step.tolist(),flush=True)
        jobs=[(f'innovation_joint_{s}',s*step) for s in (.5,1.)] if np.max(abs(step))>1e-6 else []
        proposals=[]
        for c in pool.imap_unordered(evaluate,jobs):
            c['measured_score']=measured_score(c,baseline,retained)
            outcomes.append(c);proposals.append(c)
            print('INNOVATION_JOINT',c['tag'],c['summary'],flush=True)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
        pool.close();pool.join();pool=None
        complete=[c for c in proposals if c['summary']['successes']==64]
        write_json(stage/'prediction_check.json',validation(response,stage,baseline,retained,complete))
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(metadata(parent,selected['parameters']),stage/'research_candidate.pt')
        for c in proposals:torch.save(metadata(parent,c['parameters']),stage/(c['tag']+'.pt'))
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_decision_innovation_joint_learning',full64_candidates=len(outcomes),
            best=selected['tag'] if selected else None,wall_s=time.perf_counter()-started,
            research_model_sha256=sha(stage/'research_candidate.pt') if selected else None,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_decision_innovation_joint_learning',error=str(exc),
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据目标更新与运动滞后联合学习的完整闭环结果推进形状，未冻结方案或进行封存测试。')


if __name__=='__main__':mp.freeze_support();main()

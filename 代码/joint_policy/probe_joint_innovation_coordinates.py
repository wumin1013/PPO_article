"""Matched full-return factorial probe; no action rule or policy is deployed."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import itertools
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from deep_joint_credit_policy import load_credit_policy
from joint_innovation_coordinates import innovation_coordinate_action
from probe_coordinated_suffix import reconstruct
from analyze_nonlinear_joint_response import branch
from localize_joint_shape_residual import characterize
from learn_complete_joint_teachers import teacher_score
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_bounded_causal_block_joint_teacher_learning_20260920'
    parent_file=BASE/'canonical_gate_band_lateral_aware_expert_weight_learning_20260920/width_1.2/candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    assert sha(expert_file)==read(source/'receipt.json')['expert_model_sha256']
    data=read(source/'baseline.json');paths=[206,304]
    stage=register('canonical_gate_band_innovation_coordinate_response_20260920',
        [__file__,BASE/'joint_innovation_coordinates.py',BASE/'analyze_nonlinear_joint_response.py',
         BASE/'learn_complete_joint_teachers.py',source/'saved_causal_response_audit.json',
         source/'receipt.json',source/'baseline.json',parent_file,expert_file],
        dict(wall_s=120,new_complete_branches=16,paths=paths,decision_ms=20,execution_ms=1,
             innovation_weights=[.1,.25,.25],reused_zero_paths=2),
        'The newest teachers still leave large preview-induced combined-target jumps, '
        'while steering expert differences can be tiny. At each parent trajectory largest '
        'decision jump, measure one20ms action in observable innovation coordinates: '
        'steering removes0.1 of the current combined target change; feed and preview '
        'use0.25 of the same retained-expert differences. Measure seven nonzero head '
        'subsets and a matched all-expert-direction control, then the unchanged parent '
        'on all real future states through ordered completion and rest. Reuse zero '
        'trajectories and check reconstructed next20ms identity. All interventions are '
        'offline physical diagnostics, not learned policies or deployed smoothing. '
        'Full response, including adverse shape/cruise outcomes, determines whether '
        'these coordinates merit joint parameter learning. No physics/shape relaxation.')
    started=time.perf_counter();outcomes=[];queries=[]
    costs=dict(saved_prefix_geometry_ticks=0,actor_calls=0,physics_ticks=0)
    policy=load_credit_policy(torch.load(parent_file,map_location='cpu',weights_only=False))
    expert=CanonicalGatePolicy(256)
    expert.load_state_dict(torch.load(expert_file,map_location='cpu',weights_only=False)['policy'])
    base_factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    def factory(ref):return policy.bind(base_factory(ref))
    try:
        for p in paths:
            baseline=next(r for r in data['baseline']['records'] if r['path']==p)
            retained=next(r['baseline_shape'] for r in data['retained'] if r['path']==p)
            with np.load(baseline['trace']) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            event=next(j for j in characterize(baseline['trace'])['target_jumps'] if j['time_ms']%20==0)
            frame=event['time_ms']
            env=reconstruct(ref,rows,actions,frame,costs,env_factory=base_factory)
            policy.bind(env);obs=env.observation();original=policy.act(obs,True)[0]
            direction=expert.act(obs,True)[0]-original
            initial=env.state()
            assert np.array_equal(env.decode_action(original),actions[frame//20])
            assert np.array_equal(innovation_coordinate_action(env,original,original+direction,np.zeros(3)),original)
            _,_,_,identity=env.step(original)
            assert np.array_equal(identity['rows'],rows[frame:frame+20])
            env.restore(initial)
            steering_range=[float(np.degrees(env.decode_action(original+w*direction)[0]-env.decode_action(original)[0]))
                            for w in [-.25,1.25]]
            queries.append(dict(path=p,frame=frame,original_target_jump_deg=event['magnitude_deg'],
                expert_box_steering_delta_deg=steering_range,identity_next20ms_bitwise=True))
            outcomes.append(dict(path=p,arm='zero',reused=True,record=baseline,
                                 score=teacher_score(baseline,baseline,retained)))
            arms=[('innovation_'+''.join(map(str,m)),np.array(m)*[.1,.25,.25])
                  for m in itertools.product((0,1),repeat=3) if any(m)]
            arms.append(('expert_111',np.array([.1,.25,.25])))
            for name,weights in arms:
                env.restore(initial)
                proposal=((original+weights*direction).astype(original.dtype) if name=='expert_111' else
                          innovation_coordinate_action(env,original,original+direction,weights))
                r,full,held,suffix_obs=branch(policy,factory,ref,rows,actions,initial,
                    proposal.astype(float)-original,costs,started+115)
                file=stage/f'{name}_path{p}.npz'
                np.savez_compressed(file,reference=ref,rows=full,actions=held,suffix_observations=suffix_obs,
                                    frame=frame,latent_offset=proposal-original)
                c=dict(path=p,arm=name,record=r,score=teacher_score(r,baseline,retained),
                       trace=file.name,reused=False)
                outcomes.append(c)
                write_json(stage/'results.json',dict(queries=queries,arms=outcomes,offline_only=True))
                print('INNOVATION_COORDINATE_RESPONSE',p,name,r['success'],c['score'],
                      r['shape']['full_physical_reverse_deg'],r['shape']['out_edge_outward_mm'],flush=True)
        receipt(stage,dict(status='completed_innovation_coordinate_response',new_branches=16,
            wall_s=time.perf_counter()-started,source_model_sha256=sha(parent_file),
            costs=costs,physical_diagnostic_only=True,deployed=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_innovation_coordinate_response',error=str(exc),
            completed=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':main()

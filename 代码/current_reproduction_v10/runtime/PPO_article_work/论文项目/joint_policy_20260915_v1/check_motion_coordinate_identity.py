"""Validate exact reuse when extending the state head with motion features."""
import numpy as np
import argparse
import torch
from learned_action_pulse_policy import LearnedActionPulsePolicy
from canonical_gate_state import checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry-state',action='store_true')
    parser.add_argument('--anticipation-state',action='store_true');args=parser.parse_args()
    if args.anticipation_state:args.geometry_state=True
    torch.set_num_threads(1)
    state_stage=BASE/'canonical_gate_band_shared_innovation_coordinate_learning_20260920'
    stage=(BASE/'canonical_gate_band_geometry_motion_credit_learning_20260920' if args.anticipation_state
           else BASE/'canonical_gate_band_motion_state_innovation_learning_20260920' if args.geometry_state else state_stage)
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
    data=read(stage/'baseline.json')['baseline']['records']
    state_audit=read(state_stage/'innovation_feedback_state_audit.json')
    history=read(stage/'innovation_coordinates/results.json')['candidates']
    if args.geometry_state:history=[r for r in history if np.any(np.asarray(r['coefficients'])[18 if args.anticipation_state else 9:])]
    chosen=min(history,key=lambda r:r['score']);u=np.asarray(chosen['coefficients'])
    spec=dict(turn_rate_scale_rad_s=state_audit['turn_rate_99pct_rad_s'],
              gate_forward_time_scale_s=state_audit['gate_forward_time_median_s'])
    small=LearnedActionPulsePolicy(parent,u,'innovation_coordinates',expert,spec)
    extended=LearnedActionPulsePolicy(parent,np.r_[u,np.zeros(9)],'innovation_coordinates',expert,spec)
    records=[]
    for q in state_audit['queries']:
        r=next(r for r in data if r['path']==q['path'])
        with np.load(r['trace']) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,q['frame'],dict(saved_prefix_geometry_ticks=0),
                        env_factory=checkpoint_environment_factory('gate_band_canonical_kernel'))
        small.bind(env);extended.bind(env);x=env.observation()
        a,b=small.act(x,True)[0],extended.act(x,True)[0]
        assert np.array_equal(a,b),(q['path'],a,b)
        records.append(dict(path=q['path'],frame=q['frame'],actions_bitwise=True))
    write_json(stage/('anticipation_feature_identity_audit.json' if args.anticipation_state else
        'geometry_feature_identity_audit.json' if args.geometry_state else 'motion_feature_identity_audit.json'),dict(records=records,new_rollouts=0,
        nonzero_cached_candidate=chosen['tag'],parent_sha256=sha(parent_file),
        zero_added_coefficients_preserve_old_function=True,
        code_sha256=sha(BASE/'learned_action_pulse_policy.py')))
    print('MOTION_EXTENSION_IDENTITY',len(records),'actual saved states',chosen['tag'],'bitwise')


if __name__=='__main__':main()

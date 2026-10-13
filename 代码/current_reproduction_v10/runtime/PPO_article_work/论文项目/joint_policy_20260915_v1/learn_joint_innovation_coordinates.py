"""Learn shared three-action innovation weights from actual complete returns."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import torch
from learn_shared_temporal_joint_policy import initialize,optimize
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--motion-state',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920'
    probe=BASE/'canonical_gate_band_innovation_coordinate_response_20260920'
    parent_file=source/'research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    result=read(source/'results.json')
    selected=next(c for c in result['candidates'] if c['tag']==result['best'])
    assert selected['research_acceptable'] and not (source/'selection_hold.json').exists()
    assert read(source/'raw_result_audit.json')['model_sha256']==sha(parent_file)
    assert sha(expert_file)=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    records=copy.deepcopy(selected['records'])
    for r in records:r['trace']=str(source/r['trace'])
    retained=read(source/'baseline.json')['retained']
    data=dict(baseline=dict(records=records,summary=selected['summary']),retained=retained)
    paths=[14,104,204,206,304]
    feature_spec=dict(definition='Reflection-even squared tanh of observed target jump, '
        'target-heading gap and physical heading lag, three shared weights per action. '
        'Steering acts in combined-target-angle coordinates; feed and preview act along '
        'current expert differences. No path, angle table, absolute time or teacher blocks.')
    prior=BASE/'canonical_gate_band_shared_innovation_coordinate_learning_20260920'
    if args.motion_state:
        assert not any(a['improved'] for a in read(prior/'results.json')['arms']), 'Finish positive full-pool validation first'
        audit=read(prior/'innovation_feedback_state_audit.json')
        assert read(prior/'motion_feature_identity_audit.json')['parent_sha256']==sha(parent_file)
        feature_spec.update(turn_rate_scale_rad_s=audit['turn_rate_99pct_rad_s'],
            dynamic_features='tanh(observed signed target innovations/0.5) times tanh(actual signed planar turning rate/data99pct); reflection even')
        assert feature_spec['turn_rate_scale_rad_s']>0.
    stage=register('canonical_gate_band_motion_state_innovation_learning_20260920' if args.motion_state
                   else 'canonical_gate_band_shared_innovation_coordinate_learning_20260920',
        [__file__,BASE/'learn_shared_temporal_joint_policy.py',BASE/'learned_action_pulse_policy.py',
         BASE/'joint_innovation_coordinates.py',BASE/'learn_complete_joint_teachers.py',
         BASE/'deep_joint_credit_policy.py',BASE/'complete_corner_metrics.py',parent_file,expert_file,
         source/'results.json',source/'receipt.json',source/'raw_result_audit.json',
         source/'saved_lateral_motion_audit.json',probe/'results.json',probe/'receipt.json']+
         ([prior/'results.json',prior/'receipt.json',prior/'innovation_coordinates/results.json',
           prior/'innovation_feedback_state_audit.json',prior/'motion_feature_identity_audit.json'] if args.motion_state else []),
        dict(wall_s=720,shared_parameters=18 if args.motion_state else 9,maximum_new_candidate_vectors=79,
             optimization_paths=paths,training_pool_paths=88,coefficient_bounds=[-3.,3.],
             zero_paths_repeated=0,feature_spec=feature_spec),
        ('Matched state-information extension after the nine-coefficient function has '
         'no feasible improvement: add three reflection-even products of observed signed '
         'innovations with actual signed turning rate, normalized by the99th percentile '
         'of saved full88 training decision states. This is a physical-state input, '
         'not a turn-angle table or time schedule. Keep the EXACT same parent, five '
         'optimization paths, objective, constraints, solver and budget. All completed '
         'nine-coefficient candidates are reused exactly with nine appended zeros; '
         'no existing physical rollout is repeated. The state audit shows omitted '
         'motion information, not proof that it caused failure; this comparison tests '
         'whether it actually resolves the measured protected-path credit conflict. '
         if args.motion_state else '')+
        'Saved-state audits show the expert-weight steering range is only about0.3deg '
        'at36/50deg combined-target jumps. Sixteen new full-return factorial branches '
        'show target attenuation is NOT universally beneficial:206 worsens,304 reduces '
        'exit excursion but increases reversal, and feed/preview interactions differ. '
        'Therefore do not deploy a damping rule. Learn nine shared state-dependent '
        'coefficients for all three actions jointly from actual complete returns in '
        'innovation coordinates. The audited bounded-teacher state policy is the new '
        'parent. Unlike fixed temporal teacher fits, one common function acts at every '
        'real decision and is optimized simultaneously on14,104,204,206,304. Preserve '
        'current lateral displacement,energy,exit and original14 shape; keep declared '
        'current/retained curvature and speed envelopes on unresolved cases, all '
        'physical/completion/time/cruise guards, and both parameter/trajectory feasibility '
        'in incumbent selection. A positive subset update requires the full88 pool '
        'with exact reuse of its five measured outcomes. If no feasible improvement, '
        'inspect measured credit conflicts and change representation rather than rerun '
        'unchanged search. Complete-return parameter optimization, not PPO, final '
        'adoption or generalization.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter()
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        training=[r for r in records if r['path'] in paths]
        old=[next(o['baseline_shape'] for o in retained if o['path']==r['path']) for r in training]
        initialize(stage,parent,training,old,started+705,args.motion_state,prior if args.motion_state else None,
                   expert,True,feature_spec,[104,204,206,304])
        outcome=optimize('innovation_coordinates')
        write_json(stage/'results.json',dict(arms=[outcome],training_only=True,adopted=False))
        receipt(stage,dict(status='completed_motion_state_innovation_learning' if args.motion_state else 'completed_shared_innovation_coordinate_learning',
            new_candidates=outcome['new_candidates'],
            reused_candidates=outcome['reused_candidates'],
            improved_arms=['innovation_coordinates'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_motion_state_innovation_learning' if args.motion_state else 'failed_shared_innovation_coordinate_learning',error=str(exc),
            wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':main()

"""Matched complete-trajectory action-head attribution of a learned teacher."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import itertools
import argparse
import time
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from deep_joint_credit_policy import load_credit_policy
from canonical_gate_state import checkpoint_environment_factory,CanonicalGatePolicy
from probe_coordinated_suffix import reconstruct
from analyze_nonlinear_joint_response import measure
from learn_complete_joint_teachers import teacher_score, SCALE
from distill_complete_joint_teachers import select_teachers
from study_io import BASE, register, receipt, write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expert-direction',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_neighborhood_joint_teacher_learning_20260919'
    if args.expert_direction:source=BASE/'canonical_gate_band_expert_direction_joint_teacher_learning_20260920'
    data = read(source/'baseline.json')
    selected = ([dict(path=t['path'],candidate=t['best'],start_tick=t['start_tick'],stop_tick=t['stop_tick'])
                 for t in read(source/'results.json')['teachers'] if t['score_improved']]
                if args.expert_direction else select_teachers(source, data['baseline']))
    # Predeclare the strongest reversal reduction as a mechanism case, not a test score.
    records = {r['path']: r for r in data['baseline']['records']}
    item = max(selected, key=lambda c: records[c['path']]['shape']['full_physical_reverse_deg']
               - c['candidate']['record']['shape']['full_physical_reverse_deg'])
    b = records[item['path']]
    old = next(r['baseline_shape'] for r in data['retained'] if r['path'] == item['path'])
    parent_file = BASE/'canonical_gate_band_remaining_corner_state_distillation_20260919/research_candidate.pt'
    expert=None;extra=[]
    if args.expert_direction:
        parent_file=BASE/'canonical_gate_band_visible_credit_shared_kernel_learning_20260920/research_candidate.pt'
        expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
        assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
        assert sha(expert_file)==read(source/'receipt.json')['expert_model_sha256']
        expert=CanonicalGatePolicy(256)
        expert.load_state_dict(torch.load(expert_file,map_location='cpu',weights_only=False)['policy'])
        extra=[expert_file]
    stage = register('canonical_gate_band_expert_direction_teacher_head_credit_20260920' if args.expert_direction
                     else 'canonical_gate_band_neighborhood_teacher_head_credit_20260919',
        [__file__, BASE/'learn_complete_joint_teachers.py', source/'results.json',
         source/'receipt.json', source/'baseline.json', source/item['candidate']['trace'], parent_file]+extra,
        dict(wall_s=90, paths=[item['path']], new_complete_branches=6,
             cached_arms=['000', '111'], output_heads=['steering', 'feed', 'preview']),
        ('Offline matched head-mask attribution of the newly optimized expert-direction teacher206. '
         'Recompute both learned policies on every actual state, with the saved learned temporal weights. '
         if args.expert_direction else
         'Offline matched head-mask attribution of the teacher with the largest measured reversal '
         'reduction among the eight newly added training paths. ')+
        'Reuse zero and full teacher outcomes; '
        'execute only the six missing action-head subsets, all from identical saved physical state '
        'with the same feedback parent and full ordered completion/braking/rest. This tests finite '
        'joint-action credit for this learned temporal perturbation, not a universal cause or PPO '
        'estimator bias. No masks or temporal teacher schedules become deployment rules.')
    started = time.perf_counter()
    policy = load_credit_policy(torch.load(parent_file, map_location='cpu', weights_only=False))
    factory = checkpoint_environment_factory('gate_band_canonical_kernel')
    costs = dict(saved_prefix_geometry_ticks=0, actor_calls=0, physics_ticks=0)
    with np.load(b['trace']) as z:
        ref, rows, actions = [z[k].copy() for k in ('reference', 'rows', 'actions')]
    start, stop = item['start_tick'], item['stop_tick']
    env = reconstruct(ref, rows, actions, start, costs, env_factory=factory)
    initial = env.state()
    policy.bind(env)
    latent=policy.act(env.observation(),True)[0]
    assert np.array_equal(env.decode_action(latent),actions[start//20])
    _,_,_,identity=env.step(latent)
    assert np.array_equal(identity['rows'],rows[start:start+20])
    count = (stop-start)//20+1
    basis = CubicSpline(np.linspace(0, 1, 6), np.vstack([np.zeros((1, 4)), np.eye(4),
                        np.zeros((1, 4))]), bc_type='clamped')(np.linspace(0, 1, count))
    if args.expert_direction:
        with np.load(source/item['candidate']['trace']) as z:basis=z['basis'].copy()
    parameters = np.asarray(item['candidate']['parameters']).reshape(4, 3)
    outcomes = [dict(mask=[0, 0, 0], record=b, score=teacher_score(b, b, old), reused=True),
                dict(mask=[1, 1, 1], record=item['candidate']['record'],
                     score=item['candidate']['score'], reused=True)]
    try:
        for mask in itertools.product((0, 1), repeat=3):
            if sum(mask) in (0, 3):
                continue
            env = factory(ref)
            env.restore(initial)
            policy.bind(env)
            chunks, held = [], []
            while not env.done:
                if time.perf_counter() > started+85:
                    raise TimeoutError('Head-credit wall budget')
                x=env.observation()
                latent = policy.act(x, True)[0]
                index = (env.ticks-start)//20
                direction=SCALE if expert is None else expert.act(x,True)[0]-latent
                offset = basis[index]@parameters*direction*np.asarray(mask) if index < count else np.zeros(3)
                _, _, _, info = env.step((latent+offset).astype(latent.dtype))
                chunks.append(info['rows'])
                held.append(info['held_action'])
            full = np.vstack([rows[:start], *chunks])
            all_actions = np.vstack([actions[:start//20], held])
            r = measure(ref, full, env)
            tag = ''.join(map(str, mask))
            file = stage/f'mask_{tag}.npz'
            np.savez_compressed(file, reference=ref, rows=full, actions=all_actions)
            outcomes.append(dict(mask=list(mask), record=r, score=teacher_score(r, b, old),
                                 trace=file.name, reused=False))
            write_json(stage/'results.json', dict(path=item['path'], arms=outcomes, offline_only=True))
        receipt(stage, dict(status='completed_neighborhood_teacher_head_credit', path=item['path'],
            new_complete_branches=6, wall_s=time.perf_counter()-started,
            offline_only=True, deployed=False, goal_complete=False))
        print('HEAD_CREDIT', item['path'], [(r['mask'], r['record']['success'], r['score'],
              r['record']['shape']['full_physical_reverse_deg']) for r in outcomes], flush=True)
    except Exception as exc:
        receipt(stage, dict(status='failed_neighborhood_teacher_head_credit', error=str(exc),
            completed=len(outcomes)-2, wall_s=time.perf_counter()-started, goal_complete=False))
        raise


if __name__ == '__main__':
    main()

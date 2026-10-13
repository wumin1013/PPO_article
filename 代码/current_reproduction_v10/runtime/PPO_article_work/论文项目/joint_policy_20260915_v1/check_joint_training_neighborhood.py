"""Measured training augmentation check; never a sealed/generalization test."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import multiprocessing as mp
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from deep_joint_credit_policy import load_credit_policy
from learn_nonlinear_joint_feedback import rollout
from learn_full_pool_joint_credit import aggregate
from learn_retained_joint_growth import smooth_cruise
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


WORK = {}


def initialize(stage, models, cases, deadline):
    torch.set_num_threads(1)
    policies = {}
    for name, model in models.items():
        md = torch.load(model, map_location='cpu', weights_only=False)
        if md['architecture'] == 'gate_band_canonical_kernel':
            policy = CanonicalGatePolicy(md['centers'])
            policy.load_state_dict(md['policy'])
        else:
            policy = load_credit_policy(md)
        policies[name] = policy
    WORK.update(stage=stage, policies=policies, cases=cases, deadline=deadline,
                factory=checkpoint_environment_factory('gate_band_canonical_kernel'))


def evaluate(job):
    arm, case = job
    policy = WORK['policies'][arm]
    def factory(ref):
        env = WORK['factory'](ref)
        if hasattr(policy, 'bind'):
            policy.bind(env)
        return env
    costs = dict(actor_calls=0, physics_ticks=0)
    ref = np.asarray(WORK['cases'][case]['reference'])
    r, rows, actions, obs = rollout(policy, factory, ref, costs, WORK['deadline'])
    file = WORK['stage']/f'{arm}_path{case}.npz'
    np.savez_compressed(file, reference=ref, rows=rows, actions=actions, observations=obs)
    r.update(path=case, arm=arm, trace=file.name, curved_path=True, costs=costs)
    if r['success']:
        r['smooth_cruise99_s'] = smooth_cruise(rows)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-stage', required=True)
    parser.add_argument('--interpolation', action='store_true')
    args = parser.parse_args()
    source = BASE/args.parent_stage
    results = read(source/'results.json')
    parent = next(c for c in results['candidates'] if c['tag'] == results['best'])
    models = dict(
        retained=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt',
        pre_teacher=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt',
        joint_state=source/'research_candidate.pt')
    if args.interpolation:
        models.pop('pre_teacher')
        models['before_augmentation']=BASE/'canonical_gate_band_remaining_corner_state_distillation_20260919/research_candidate.pt'
    length_scale=.985 if args.interpolation else .97
    angle_change=.5 if args.interpolation else 1.
    cases = {}
    for path in [1, 12, 14, 15]:
        record = next(r for r in parent['records'] if r['path'] == path)
        with np.load(source/record['trace']) as z:
            reference = z['reference'].copy()
        for kind in ('incoming_length', 'angle'):
            ref = reference.copy()
            if kind == 'incoming_length':
                ref[0] = ref[1] + length_scale*(ref[0]-ref[1])
            else:
                theta = np.sign(record['shape']['turn_deg'])*angle_change*np.pi/180
                rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])
                ref[2] = ref[1] + rotation@(ref[2]-ref[1])
            cases[(200 if args.interpolation else 100)+len(cases)] = dict(source_path=path, kind=kind, reference=ref.tolist())
    stage = register('canonical_gate_band_joint_training_interpolation_20260919' if args.interpolation else
                     'canonical_gate_band_joint_training_neighborhood_20260919',
        [__file__, BASE/'joint_teacher_state_policy.py', BASE/'learn_nonlinear_joint_feedback.py',
         BASE/'analyze_nonlinear_joint_response.py', source/'results.json', source/'receipt.json',
         source/'raw_result_audit.json', *models.values()],
        dict(wall_s=240, workers=3, new_training_paths=len(cases), arms=list(models),
             incoming_length_scale=length_scale, signed_angle_change_deg=angle_change),
        ('The first expanded training fit improved the original and added paths; now measure '
         'the eight MIDPOINT training augmentations between them to expose interpolation '
         'gaps. They remain eligible training data, not held-out generalization tests. '
         'Use1.5%incoming-length and0.5degree angle changes, compare retained86cd, '
         'before-augmentation b5b and latest state policy. ' if args.interpolation else '')+
        'Measure support of the learned observable-state joint teacher corrections on eight '
        f'explicitly added TRAINING cases. Independently shorten only the incoming edge by{100*(1-length_scale):g}% '
        f'or increase turn magnitude by{angle_change:g}degree on four training corners. Compare retained86cd, '
        'the declared earlier state policy and the latest state policy with identical actual physical execution. '
        'Original completed paths are not rerun. These new cases are eligible for subsequent '
        'learning, never final independent, frozen or sealed generalization evidence. '
        'Keep full raw outcomes including failures, no failed-prefix time as completion. '
        'If state corrections are brittle, collect measured joint teachers on expanded training '
        'states before any further claim or freeze; no hand-coded action schedule is deployed.')
    write_json(stage/'training_cases.json', cases)
    started = time.perf_counter()
    outcomes = []
    pool = None
    try:
        pool = mp.get_context('spawn').Pool(3, initialize, (stage, models, cases, started+225))
        jobs = [(arm, case) for arm in models for case in cases]
        for r in pool.imap_unordered(evaluate, jobs):
            outcomes.append(r)
            write_json(stage/'results.json', dict(records=outcomes, training_only=True))
            print('TRAINING_NEIGHBORHOOD', r['arm'], r['path'], r['success'],
                  r['shape']['full_physical_reverse_deg'], flush=True)
        pool.close()
        pool.join()
        pool = None
        summaries = {arm: aggregate([r for r in outcomes if r['arm'] == arm]) for arm in models}
        write_json(stage/'results.json', dict(records=outcomes, summaries=summaries, training_only=True))
        receipt(stage, dict(status='completed_training_neighborhood_check', summaries=summaries,
            source_model_sha256={k: sha(v) for k, v in models.items()}, wall_s=time.perf_counter()-started,
            training_only=True, final_adopted=False, goal_complete=False))
        print('TRAINING_NEIGHBORHOOD_COMPLETE', summaries, flush=True)
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage, dict(status='failed_training_neighborhood_check', error=str(exc),
            completed=len(outcomes), wall_s=time.perf_counter()-started, training_only=True,
            final_adopted=False, goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()

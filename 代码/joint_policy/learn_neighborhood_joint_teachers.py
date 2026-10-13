"""Measured joint-action teaching on explicitly expanded training states."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import multiprocessing as mp
import time
import torch
from learn_complete_joint_teachers import initialize, optimize_path, teacher_score
from learn_full_pool_joint_credit import aggregate
from learn_absolute_shape_joint_credit import absolute_actual
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_remaining_corner_state_distillation_20260919'
    check = BASE/'canonical_gate_band_joint_training_neighborhood_20260919'
    result = read(source/'results.json')
    baseline = copy.deepcopy(next(c for c in result['candidates'] if c['tag'] == result['best']))
    for r in baseline['records']:
        r['trace'] = str(source/r['trace'])
    retained = read(source/'baseline.json')['retained']
    added = read(check/'results.json')['records']
    for r in sorted((r for r in added if r['arm'] == 'joint_state'), key=lambda r: r['path']):
        r = copy.deepcopy(r)
        r['trace'] = str(check/r['trace'])
        baseline['records'].append(r)
        old = next(b for b in added if b['path'] == r['path'] and b['arm'] == 'retained')
        retained.append(dict(path=r['path'], baseline_shape=old['shape'],
            complete_time_s=old['complete_time_s'], complete_time_change_s=0.))
    baseline['summary'] = aggregate(baseline['records'])
    baseline['absolute_objective'] = absolute_actual(baseline, retained)
    parent_file = source/'research_candidate.pt'
    assert sha(parent_file) == 'b5b7d39361474a315b501805cf291c3d1ef4060298b205daac3d32bc8874624b'
    parent = torch.load(parent_file, map_location='cpu', weights_only=False)
    stage = register('canonical_gate_band_neighborhood_joint_teacher_learning_20260919',
        [__file__, BASE/'learn_complete_joint_teachers.py', BASE/'joint_teacher_state_policy.py',
         parent_file, source/'results.json', source/'receipt.json', source/'raw_result_audit.json',
         check/'results.json', check/'receipt.json', check/'training_cases.json'],
        dict(wall_s=600, workers=4, paths=list(range(100, 108)),
             maximum_new_actual_evaluations_per_path=240, initial_radius=.025,
             coefficient_bounds=[-.6, .6], constraint_tolerance=1e-8),
        'Training-neighborhood check found deterioration under1degree or3%incoming-length '
        'changes despite8/8 complete success. Broaden TRAINING coverage with actual constrained '
        'joint-action teachers on all eight added paths. Reuse completed current-parent zero '
        'trajectories; verify next20ms identity before optimization. Preserve per-path entry, '
        'exit,reversal,energy,peak,apex speed,time,cruise and unchanged hard constraints. '
        'Full72 baseline includes all original64 plus eight added training paths. Teachers '
        'remain offline; only a state function fitted jointly and tested on all72 can be '
        'saved as a research checkpoint. This is neither PPO nor final generalization evidence.')
    write_json(stage/'baseline.json', dict(baseline=baseline, retained=retained))
    started = time.perf_counter()
    pool = None
    outcomes = []
    try:
        jobs = []
        for b in baseline['records'][64:]:
            o = next(r for r in retained if r['path'] == b['path'])['baseline_shape']
            seed = dict(tag='reused_seed', path=b['path'], parameters=[0.]*12,
                        score=teacher_score(b, b, o), trace=b['trace'], record=b)
            jobs.append((b['path'], b, o, seed))
        pool = mp.get_context('spawn').Pool(4, initialize, (stage, parent, started+585, 'cobyla'))
        for r in pool.imap_unordered(optimize_path, jobs):
            outcomes.append(r)
            write_json(stage/'results.json', dict(teachers=outcomes, offline_teacher_only=True,
                                                 deployable_policy=False, training_only=True))
        pool.close()
        pool.join()
        pool = None
        receipt(stage, dict(status='completed_neighborhood_joint_teacher_learning', paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),
            trials=sum(c['trials'] for c in outcomes), source_model_sha256=sha(parent_file),
            wall_s=time.perf_counter()-started, offline_teacher_only=True,
            deployable_policy=False, adopted=False, goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage, dict(status='failed_neighborhood_joint_teacher_learning', error=str(exc),
            completed_paths=len(outcomes), wall_s=time.perf_counter()-started,
            offline_teacher_only=True, deployable_policy=False, adopted=False, goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()

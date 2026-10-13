"""Direct bounded line-search teacher learning after measured broad-search failure."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path
from learn_full_pool_compensation import progress
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_complete_joint_teacher_learning_20260919'
    data=read(BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json')
    parent_file=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    paths=[12,14,15]
    stage=register('canonical_gate_band_bounded_joint_teacher_learning_20260919',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'decision_innovation_policy.py',parent_file,
         source/'results.json',source/'receipt.json']+[source/f'path{p}'/'results.json' for p in paths]+
        [BASE/'canonical_gate_band_joint_teacher_state_distillation_20260919/receipt.json',
         BASE/'canonical_gate_band_joint_teacher_state_distillation_20260919/raw_result_audit.json',
         BASE/'canonical_gate_band_joint_teacher_state_distillation_20260919/research_candidate.pt'],
        dict(wall_s=600,workers=3,paths=paths,maximum_actual_evaluations_per_path=180,
             initial_direction_scale=.03,coefficient_bounds=[-.3,.3],maximum_iterations=5),
        'CEM yielded no better teacher on12/14/15. In particular, path14 had73/73 '
        'complete trials but best nonzero score617 versus88.46 baseline, showing '
        'the sampled joint perturbations were too large or poorly coordinated; '
        'do not repeat that sampler. Compare bounded Powell actual-return line '
        'search in the SAME12 joint teacher coordinates and SAME objective. '
        'Use0.03 initial directions,bounds+/-0.3 and at most180 evaluations per '
        'path, reusing zero outcomes exactly. No local trajectory-response model '
        'predicts candidate quality; each line-search value is a true complete '
        'physical rollout or fixed failure penalty. This remains OFFLINE teacher '
        'parameter optimization, not a deployed schedule or PPO. Preserve the '
        'new a731 state-policy checkpoint and its64/64 result. New useful teachers '
        'must pass nondegrading shape/time/cruise selection and joint state-policy '
        'distillation plus full closure, without discarding the established+140 gain.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'保留a731研究策略，用有界直接线搜索修正未改善的三动作教师，所有评分来自真实完整执行。')
    try:
        jobs=[(p,next(r for r in data['baseline']['records'] if r['path']==p),
            next(r for r in data['retained'] if r['path']==p)['baseline_shape']) for p in paths]
        pool=mp.get_context('spawn').Pool(3,initialize,(stage,parent,started+585,'powell'))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,adopted=False,goal_complete=False))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_bounded_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,deployable_policy=False,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_bounded_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'继续联合状态策略学习以修复困难角，保留a731与原正式模型，所有最终验收仍待完成。')


if __name__=='__main__':mp.freeze_support();main()

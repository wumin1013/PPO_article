"""Actual-outcome constrained teacher learning; no scalar-regret tradeoff."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path
from learn_full_pool_compensation import progress
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_bounded_joint_teacher_learning_20260919'
    source_results=read(source/'results.json')
    data=read(BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json')
    parent_file=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    paths=[12,14]
    stage=register('canonical_gate_band_constrained_joint_teacher_learning_20260919',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'decision_innovation_policy.py',parent_file,
         source/'results.json',source/'receipt.json']+[source/f'path{p}'/'results.json' for p in paths],
        dict(wall_s=480,workers=2,paths=paths,maximum_new_actual_evaluations_per_path=240,
             initial_radius=.025,coefficient_bounds=[-.6,.6],constraint_tolerance=1e-8),
        'Bounded direct search improved real complete trajectories but scalar penalties '
        'still accepted a0.000340mm exit increase on-140 and0.01928/mm peak-curvature '
        'increase on-155. Do not distill those tradeoffs or loosen acceptance. Change '
        'the teacher optimizer to COBYLA with explicit ACTUAL-output inequalities '
        'for entry,exit,reverse,energy,peak,apex speed,complete time,cruise,inward '
        'first turn and corridor margin. Start from the already measured improving '
        'teachers and cache their outcomes; no zero/seed rerun. Each distinct function '
        'and constraint query shares a single real complete rollout. Expand only '
        'offline coefficient bounds from0.3 to0.6 because several useful feed/preview '
        'coordinates reached bounds; actual physical/action/coverage/stop limits '
        'are unchanged. A cached feasible zero remains available. At most240 new '
        'measurements per path. Teachers remain nondeployable; useful outcomes '
        'require transfer to the preserved state-policy parent,joint fitting and '
        'full64 closure. This is constrained offline teacher parameter learning, '
        'not PPO or deployment of an angle/time template.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'以实际完整回报和显式形状不等式联合优化-140与-155教师，消除软惩罚仍接受的退化。')
    try:
        jobs=[]
        for path in paths:
            item=next(r for r in source_results['teachers'] if r['path']==path)
            seed=copy.deepcopy(item['best']);seed['trace']=str(source/seed['trace'])
            jobs.append((path,next(r for r in data['baseline']['records'] if r['path']==path),
                next(r for r in data['retained'] if r['path']==path)['baseline_shape'],seed))
        pool=mp.get_context('spawn').Pool(2,initialize,(stage,parent,started+465,'cobyla'))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,adopted=False,goal_complete=False))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_constrained_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,deployable_policy=False,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_constrained_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'将实际非退化的困难角教师转为联合状态策略，并继续完整形状与巡航验收。')


if __name__=='__main__':mp.freeze_support();main()

"""Joint complete-return teachers also protect within-band wrong-way XY motion."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_complete_joint_teachers import initialize,optimize_path,teacher_score
from complete_corner_metrics import complete_corner_metrics
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_expert_direction_exact_state_shared_kernel_learning_20260920'
    credit=BASE/'canonical_gate_band_expert_direction_teacher_head_credit_20260920'
    parent_file=source/'research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)=='8052f99760d6e4c2a584a76a8e8b33a6eb8d432b3012c9c5f2b726a118cbc165'
    assert sha(expert_file)=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    result=read(source/'results.json')
    chosen=next(c for c in result['candidates'] if c['tag']==result['best'])
    records=copy.deepcopy(chosen['records'])
    for r in records:
        r['trace']=str(source/r['trace'])
        if 'shape' in r:
            with np.load(r['trace']) as z:
                r['shape']=complete_corner_metrics(z['reference'],z['rows'])
    data=dict(baseline=dict(records=records,summary=aggregate(records)),
              retained=read(source/'baseline.json')['retained'])
    paths=[104,204,206]
    stage=register('canonical_gate_band_lateral_aware_joint_teacher_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'complete_corner_metrics.py',
         BASE/'learn_full_pool_joint_credit.py',BASE/'localize_joint_shape_residual.py',
         parent_file,expert_file,source/'receipt.json',source/'raw_result_audit.json',
         source/'saved_lateral_motion_audit.json',credit/'results.json',credit/'receipt.json'],
        dict(wall_s=600,workers=3,paths=paths,maximum_new_evaluations_per_path=240,
             parameters_per_path=12,temporal_scope='event_resolved',
             lateral_loss_scale_mm=.005,lateral_duration_scale_s=.05,
             protect_current_lateral_total=True,protect_current_energy=True,protect_current_exit=True),
        'The current learned state model passes80 complete paths, but a rawXY audit reveals '
        'within-band outward movement missed by the edge-excursion check. The latest206 '
        'update lowers total reversal and peak curvature yet slightly increases incoming '
        'wrong-way lateral displacement and significant duration91to93ms. Add raw1ms '
        'incoming-normal negative displacement before vertex approach to complete-return '
        'credit and a nonincrease guard; also penalize its significant duration using the '
        'existing0.05deg/5mm/s diagnostic convention. This repairs an objective omission, '
        'not physical constraints. Recompute only saved baseline metrics, no baseline '
        'physical repetition. Optimize three-action learned-expert directions jointly '
        'on104,204,206, then require shared observable-state fitting and all80 actual '
        'closures. Keep all prior physical,coverage,braking,rest,time,cruise,energy,exit '
        'and declared unresolved-case shape envelopes. Offline temporal teachers remain '
        'nondeployable. No PPO, no independent or sealed test, no final acceptance.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        jobs=[]
        for p in paths:
            b=next(r for r in records if r['path']==p)
            o=next(r for r in data['retained'] if r['path']==p)['baseline_shape']
            seed=dict(tag='reused_parent',path=p,parameters=[0.]*12,record=b,
                      score=teacher_score(b,b,o),trace=b['trace'])
            jobs.append((p,b,o,seed))
        pool=mp.get_context('spawn').Pool(3,initialize,
            (stage,parent,started+585,'cobyla','event_resolved',True,True,expert,True))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,protect_current_lateral_total=True))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_lateral_aware_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(t['score_improved'] for t in outcomes),trials=sum(t['trials'] for t in outcomes),
            source_model_sha256=sha(parent_file),expert_model_sha256=sha(expert_file),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,deployable_policy=False,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_lateral_aware_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':mp.freeze_support();main()

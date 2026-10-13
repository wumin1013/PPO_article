"""Fix teacher action-credit support using observed early reversal intervals."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path,teacher_score
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_neighborhood_teacher_state_distillation_20260919'
    result=read(source/'results.json')
    baseline=copy.deepcopy(next(c for c in result['candidates'] if c['tag']==result['best']))
    for r in baseline['records']:r['trace']=str(source/r['trace'])
    retained=read(source/'baseline.json')['retained']
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='549aa16711b5557aefb6006f806b12e34bfae263be687302d47a3a3c4ae479c3'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    paths=[3,4,14,104]
    stage=register('canonical_gate_band_full_window_joint_teacher_learning_20260919',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'joint_teacher_state_policy.py',
         parent_file,source/'results.json',source/'receipt.json',source/'raw_result_audit.json',
         source/'remaining_shape_localization.json'],
        dict(wall_s=600,workers=4,paths=paths,maximum_new_actual_evaluations_per_path=240,
             initial_radius=.025,coefficient_bounds=[-.6,.6],constraint_tolerance=1e-8,
             free_parameters_per_path=12,temporal_scope='full_window'),
        'Saved1ms raw trajectories prove most remaining+35/-60 reversal occurs BEFORE '
        'the prior apex-minus120ms teacher start, giving that parameterization an '
        'unmodifiable error floor. Move the OFFLINE three-action teacher support to '
        'tick0 and include the first action, using four free spline coefficients '
        'per head and a zero final endpoint: still twelve parameters, same optimizer '
        'and evaluation budget. Also test the two-155 training cases with residual '
        'curvature/exit errors. This changes credit/action support, not the20ms '
        'decision or1ms execution. Original teacher schedules are not deployed. '
        'Reuse completed zero trajectories, check first20ms identity, maintain '
        'all actual per-path guards; only a jointly fitted state policy passing '
        'the full72 training pool may be saved. No final generalization claim.')
    started=time.perf_counter();pool=None;outcomes=[]
    write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
    try:
        jobs=[]
        for path in paths:
            b=next(r for r in baseline['records'] if r['path']==path)
            o=next(r for r in retained if r['path']==path)['baseline_shape']
            seed=dict(tag='reused_seed',path=path,parameters=[0.]*12,
                score=teacher_score(b,b,o),trace=b['trace'],record=b)
            jobs.append((path,b,o,seed))
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,started+585,'cobyla','full_window'))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,temporal_scope='full_window'))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_full_window_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_full_window_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()

"""Learn localized joint-action credit at measured target discontinuities."""
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
    source=BASE/'canonical_gate_band_full_window_teacher_state_distillation_20260919'
    result=read(source/'results.json')
    baseline=copy.deepcopy(next(c for c in result['candidates'] if c['tag']==result['best']))
    for r in baseline['records']:r['trace']=str(source/r['trace'])
    retained=read(source/'baseline.json')['retained']
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='2b277d3c2cf87018d8fa16a8417daba5e6eb10298f8eb8cee3d3b9cbfef2daf7'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_target_jump_joint_teacher_learning_20260919',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'localize_joint_shape_residual.py',
         BASE/'joint_teacher_state_policy.py',parent_file,source/'results.json',source/'receipt.json',
         source/'raw_result_audit.json'],
        dict(wall_s=360,workers=2,paths=[14,104],maximum_new_actual_evaluations_per_path=240,
             initial_radius=.025,coefficient_bounds=[-.6,.6],constraint_tolerance=1e-8,
             free_parameters_per_path=12,temporal_scope='target_jump_window'),
        'Full-window learning reduced early reversal but-155 exit and curvature gaps remain. '
        'The saved actual target has a large near-apex20ms jump and peak-curvature protection '
        'is binding. Test action-time resolution without changing any shape guards: same '
        'twelve joint parameters, now independently covering four20ms decisions at '
        'largest-target-jump-minus40ms through plus20ms, then zero. Window selection '
        'is OFFLINE from the actual parent trace, never a runtime event rule. Every '
        'candidate resumes the full learned parent on real subsequent states through '
        'completion. Only an observable-state joint fit passing all72 training paths '
        'can be saved; no PPO, template deployment, independent or sealed-test claim.')
    write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        jobs=[]
        for path in [14,104]:
            b=next(r for r in baseline['records'] if r['path']==path)
            o=next(r for r in retained if r['path']==path)['baseline_shape']
            seed=dict(tag='reused_seed',path=path,parameters=[0.]*12,
                      score=teacher_score(b,b,o),trace=b['trace'],record=b)
            jobs.append((path,b,o,seed))
        pool=mp.get_context('spawn').Pool(2,initialize,(stage,parent,started+345,'cobyla','target_jump_window'))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,temporal_scope='target_jump_window'))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_target_jump_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_target_jump_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()

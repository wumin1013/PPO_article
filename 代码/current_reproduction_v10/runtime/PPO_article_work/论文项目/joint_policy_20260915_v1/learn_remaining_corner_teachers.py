"""Apply validated actual-return teacher learning to unresolved training corners."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path,teacher_score
from learn_full_pool_compensation import progress
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_constrained_teacher_state_distillation_20260919'
    result=read(source/'results.json')
    baseline=copy.deepcopy(next(c for c in result['candidates'] if c['tag']==result['best']))
    for r in baseline['records']:r['trace']=str(source/r['trace'])
    retained=read(source/'baseline.json')['retained']
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='029c6ec5e740d6bfabc1fce208d65a6440f4b408147dc9956c0eaa7576917abe'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    paths=[0,1,3,4]
    stage=register('canonical_gate_band_remaining_corner_teacher_learning_20260919',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'joint_teacher_state_policy.py',
         BASE/'deep_joint_credit_policy.py',BASE/'decision_innovation_policy.py',parent_file,
         source/'results.json',source/'receipt.json',source/'raw_result_audit.json'],
        dict(wall_s=600,workers=4,paths=paths,maximum_new_actual_evaluations_per_path=240,
             initial_radius=.025,coefficient_bounds=[-.6,.6],constraint_tolerance=1e-8),
        'The learned observable-state policy now passed full64 with88.798s,6.877s '
        'curved99 cruise,13.17586deg reversal and0.01801283mm exit. The other12 '
        'isolated training corners still need shape improvement. Apply the '
        'validated actual-output constrained joint-teacher mechanism to the '
        'four largest remaining non-large-angle reversal cases0,1,3,4, without '
        'rerunning any completed teacher experiment. Use the actual029c parent '
        'including all learned140/155 improvements, not an older actor. Verify '
        'a zero next20ms window at each start before learning, reuse full zero '
        'outcomes, and optimize three actions under the same explicit shape, '
        'speed,complete-time,cruise and physical constraints. Each teacher uses '
        'the existing12 OFFLINE temporal parameters; no such schedule is '
        'deployed. Any selected teacher must become a state policy and pass '
        'whole64 closure while preserving the new parent. These are training '
        'routes only; no independent,full long-path,fair or sealed-test claim.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'以029c研究策略为起点处理四个尚有明显反转的中小角，继续使用实测约束三动作教师。')
    try:
        write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
        jobs=[]
        for path in paths:
            b=next(r for r in baseline['records'] if r['path']==path)
            o=next(r for r in retained if r['path']==path)['baseline_shape']
            seed=dict(tag='reused_seed',path=path,parameters=[0.]*12,
                score=teacher_score(b,b,o),trace=b['trace'],record=b)
            jobs.append((path,b,o,seed))
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,started+585,'cobyla'))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,adopted=False,goal_complete=False))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_remaining_corner_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_remaining_corner_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'将中小角有效教师联合学入状态策略并复核完整路径，仍未达到全部最终验收。')


if __name__=='__main__':mp.freeze_support();main()

"""Optimize observable teacher credit while protecting current shape energy."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    previous=BASE/'canonical_gate_band_adaptive_credit_joint_teacher_learning_20260920'
    failed=BASE/'canonical_gate_band_adaptive_credit_shared_kernel_learning_20260920'
    source=BASE/'canonical_gate_band_shared_temporal_joint_learning_20260919'
    data=read(source/'baseline.json')
    parent_file=BASE/'canonical_gate_band_target_jump_teacher_state_distillation_20260919/research_candidate.pt'
    assert sha(parent_file)=='20e60338eb4e6980eefc013766cf04a8d943d0e6f3d8bc97e658dfddf340cb00'
    old={t['path']:t for t in read(previous/'results.json')['teachers']}
    stage=register('canonical_gate_band_visible_credit_joint_teacher_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'joint_anticipation_features.py',parent_file,
         previous/'results.json',previous/'receipt.json',source/'baseline.json',
         failed/'receipt.json',failed/'reflection_identifiability_audit.json',failed/'state_alias_audit.json'],
        dict(wall_s=600,workers=3,paths=[104,204,206],maximum_new_evaluations_per_path=241,
             free_parameters_per_path=12,temporal_scope='visible_approach',protect_current_energy=True,
             action_interval_ms=20,execution_interval_ms=1),
        'The previous complete teacher requested a signed steering residual at path206 '
        'decision18 where all shared observable inputs were exactly reflection-identical. '
        'Both fitted widths failed the same nonzero label; no rollout was needed. Correct '
        'the OFFLINE teacher support to begin at first actual observable bend, rather than '
        'one decision before visibility. Keep the current/retained reference envelope on '
        'three unresolved paths, but additionally protect EACH current shape energy, since '
        'previous teachers together necessarily failed the existing full-pool energy screen. '
        'Reuse actual seeds for unchanged supports104/204; measure the shifted206 seed anew. '
        'Optimize three actions jointly with actual complete feedback returns, not extra '
        'feed labels. Original physics,coverage,braking,rest and time/cruise screens remain. '
        'Afterwards require shared state fitting and all80 actual closures. No teacher '
        'schedule is deployable; no PPO or final shape acceptance is claimed.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        jobs=[]
        for path in (104,204,206):
            b=next(r for r in data['baseline']['records'] if r['path']==path)
            o=next(r for r in data['retained'] if r['path']==path)['baseline_shape']
            seed=dict(old[path]['best'])
            seed['trace']=str(previous/seed['trace'])
            if path==206:seed['evaluate_parameters']=True
            jobs.append((path,b,o,seed))
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        pool=mp.get_context('spawn').Pool(3,initialize,
            (stage,parent,started+585,'cobyla','visible_approach',True,True))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,temporal_scope='visible_approach',protect_current_energy=True))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_visible_credit_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(t['score_improved'] for t in outcomes),
            trials=sum(t['trials'] for t in outcomes),source_model_sha256=sha(parent_file),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,deployable_policy=False,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_visible_credit_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':mp.freeze_support();main()

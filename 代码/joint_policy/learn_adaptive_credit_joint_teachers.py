"""Complete feedback teachers for unresolved interpolation-training cases."""
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
    source=BASE/'canonical_gate_band_shared_temporal_joint_learning_20260919'
    prior=BASE/'canonical_gate_band_reference_envelope_joint_learning_20260920'
    data=read(source/'baseline.json')
    baseline,retained=copy.deepcopy(data['baseline']),data['retained']
    parent_file=BASE/'canonical_gate_band_target_jump_teacher_state_distillation_20260919/research_candidate.pt'
    assert sha(parent_file)=='20e60338eb4e6980eefc013766cf04a8d943d0e6f3d8bc97e658dfddf340cb00'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_adaptive_credit_joint_teacher_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'joint_anticipation_features.py',
         BASE/'joint_teacher_state_policy.py',parent_file,source/'baseline.json',
         prior/'results.json',prior/'receipt.json',
         BASE/'canonical_gate_band_joint_expert_residual_learning_20260920/anticipation_feature_audit.json'],
        dict(wall_s=600,workers=3,paths=[104,204,206],maximum_new_actual_evaluations_per_path=240,
             free_parameters_per_path=12,temporal_scope='observed_approach',
             reference_envelope_paths=[104,204,206],initial_radius=.025,bounds=[-.6,.6]),
        'Shared18-parameter expert corrections could improve three added difficult cases '
        'only while harming the protected original-155 path. Obtain more explicit '
        'complete-trajectory three-action credit separately on unresolved TRAINING '
        'cases104,204,206; then a shared observable-state fit must protect the whole80 '
        'pool. Keep the already declared current/retained reference envelope on these '
        'three unaccepted cases, all original physical/coverage/braking/rest limits, '
        'and current time/cruise requirements. Offline teacher support begins one '
        'decision before the saved local sensor first sees nonzero future-bend '
        'geometry, ending240ms after the parent apex; four joint coefficients '
        'per action remain twelve parameters. The window and its diagnostic visibility '
        'threshold are never deployed. Do not repeat zero full paths; verify next20ms '
        'identity. Teachers alone are not learned state-policy acceptance or PPO.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        jobs=[]
        for path in [104,204,206]:
            b=next(r for r in baseline['records'] if r['path']==path)
            o=next(r for r in retained if r['path']==path)['baseline_shape']
            seed=dict(tag='reused_seed',path=path,parameters=[0.]*12,
                      score=teacher_score(b,b,o),trace=b['trace'],record=b)
            jobs.append((path,b,o,seed))
        pool=mp.get_context('spawn').Pool(3,initialize,(stage,parent,started+585,'cobyla','observed_approach',True))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,temporal_scope='observed_approach',reference_envelope_paths=[104,204,206]))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_adaptive_credit_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_adaptive_credit_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()

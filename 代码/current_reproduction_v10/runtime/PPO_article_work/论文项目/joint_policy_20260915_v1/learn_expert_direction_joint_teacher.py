"""Joint feedback expert directions replace ill-scaled latent coordinate moves."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path,teacher_score
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    parent_stage=BASE/'canonical_gate_band_visible_credit_shared_kernel_learning_20260920'
    prior=BASE/'canonical_gate_band_event_resolved_joint_teacher_learning_20260920'
    parent_file=parent_stage/'research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)=='defa0bbad0e32985fc8ff001eed3306e40d431b14500ee004bd43fc8c31db510'
    assert sha(expert_file)=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    data=read(prior/'baseline.json')
    stage=register('canonical_gate_band_expert_direction_joint_teacher_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'localize_joint_shape_residual.py',
         BASE/'canonical_gate_state.py',parent_file,expert_file,prior/'receipt.json',
         prior/'path206/results.json',prior/'baseline.json',parent_stage/'raw_result_audit.json'],
        dict(wall_s=600,paths=[206],maximum_new_actual_evaluations=240,free_parameters=12,
             temporal_scope='event_resolved',latent_direction='retained_expert_minus_current_at_actual_state',
             learned_weight_bounds=[-.25,1.25],initial_radius=.15,protect_current_energy=True,
             protect_current_exit=True),
        'Small raw-latent event coefficients improved the objective but were rejected by '
        'a0.00058mm/s apex-speed deficit; this is not a physical capacity proof. On the '
        'same audited parent and actual complete path206, replace fixed latent scales '
        'with the three-action difference between two already learned policies evaluated '
        'on EACH ACTUAL feedback state. Optimize twelve event-basis mixture weights '
        'jointly. This gives a coordinated direction toward the retained model that had '
        'lower curvature and higher apex speed, while allowing the current policy to '
        'retain its learned approach cruise. Prior global18-parameter expert weights '
        'harmed another corner; this offline teacher is only an oracle for subsequent '
        'shared state fitting and complete80-path checking. No path or time mixture '
        'is deployed. Protect current exit and energy in addition to the existing '
        'envelope,time,cruise,physics,coverage,braking/rest. Zero baseline is reused; '
        'new20ms identity check only. Reject if no actual feasible improvement. Not PPO.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter()
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        initialize(stage,parent,started+585,'cobyla','event_resolved',True,True,expert,True)
        b=next(r for r in data['baseline']['records'] if r['path']==206)
        o=next(r for r in data['retained'] if r['path']==206)['baseline_shape']
        seed=dict(tag='reused_parent',path=206,parameters=[0.]*12,record=b,
                  score=teacher_score(b,b,o),trace=b['trace'])
        r=optimize_path((206,b,o,seed))
        write_json(stage/'results.json',dict(teachers=[r],offline_teacher_only=True,
            deployable_policy=False,teacher_direction='learned_expert_difference',protect_current_energy=True))
        receipt(stage,dict(status='completed_expert_direction_joint_teacher_learning',
            paths=1,improved_teachers=int(r['score_improved']),trials=r['trials'],
            source_model_sha256=sha(parent_file),expert_model_sha256=sha(expert_file),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,
            deployable_policy=False,adopted=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_expert_direction_joint_teacher_learning',error=str(exc),
            wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':main()

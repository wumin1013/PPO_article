"""Resolve three-action credit around measured target and recovery events."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import torch
from learn_complete_joint_teachers import initialize,optimize_path,teacher_score
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_visible_credit_shared_kernel_learning_20260920'
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='defa0bbad0e32985fc8ff001eed3306e40d431b14500ee004bd43fc8c31db510'
    result=read(source/'results.json')
    selected=next(c for c in result['candidates'] if c['tag']==result['best'])
    assert selected['research_acceptable'] and selected['summary']['successes']==80
    baseline=dict(records=copy.deepcopy(selected['records']),summary=selected['summary'])
    for r in baseline['records']:r['trace']=str(source/r['trace'])
    retained=read(source/'baseline.json')['retained']
    data=dict(baseline=baseline,retained=retained)
    stage=register('canonical_gate_band_event_resolved_joint_teacher_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'localize_joint_shape_residual.py',
         BASE/'shared_teacher_kernel_policy.py',BASE/'deep_joint_credit_policy.py',parent_file,
         source/'receipt.json',source/'results.json',source/'raw_result_audit.json'],
        dict(wall_s=600,paths=[206],maximum_new_actual_evaluations=240,free_parameters=12,
             same_latent_scales=[.12,.25,.25],same_coefficient_bounds=[-.6,.6],
             temporal_scope='event_resolved',protect_current_energy=True),
        'The learned80-path policy passed original physical/completion checks and protected '
        'all other physical traces exactly, but path206 still has curvature about3.47/mm '
        'and apex speed21mm/s. Saved1ms motion localizes its large target jump before the '
        'apex and its main reverse turn after the apex. The previous four uniform temporal '
        'coefficients per head spread credit over the entire observed approach. Keep twelve '
        'three-action parameters, scales,bounds and actual-return optimization, but place '
        'the four OFFLINE free nodes immediately before/at the measured target jump, '
        'first decision after apex, and first decision after the dominant recovery reversal. '
        'Fixed zero endpoints remain first visible bend and apex+240ms. This is a new '
        'teacher parameterization, not extra unchanged training or a deployed event rule. '
        'Parent is the newly audited learned state model; reuse its complete zero baseline '
        'and verify next20ms identity. Protect current energy and declared shape envelope, '
        'time/cruise,1.25mm,V/A/J,coverage,braking/rest. If actual credit improves, distill '
        'into the same observable-state policy and test the complete80 pool before saving. '
        'Reject if no shape gain; retain the existing research model. Not PPO.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter()
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        initialize(stage,parent,started+585,'cobyla','event_resolved',True,True)
        b=next(r for r in baseline['records'] if r['path']==206)
        o=next(r for r in retained if r['path']==206)['baseline_shape']
        seed=dict(tag='reused_parent',path=206,parameters=[0.]*12,record=b,
                  score=teacher_score(b,b,o),trace=b['trace'])
        r=optimize_path((206,b,o,seed))
        write_json(stage/'results.json',dict(teachers=[r],offline_teacher_only=True,
            deployable_policy=False,temporal_scope='event_resolved',protect_current_energy=True))
        receipt(stage,dict(status='completed_event_resolved_joint_teacher_learning',
            paths=1,improved_teachers=int(r['score_improved']),trials=r['trials'],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_event_resolved_joint_teacher_learning',error=str(exc),
            wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':main()

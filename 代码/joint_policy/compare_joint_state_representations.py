"""New training geometries compare additive and feedback expert-weight functions."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import time
import numpy as np
from check_joint_training_neighborhood import initialize,evaluate
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    additive=BASE/'canonical_gate_band_lateral_aware_shared_kernel_learning_20260920'
    weights=BASE/'canonical_gate_band_lateral_aware_expert_weight_learning_20260920'
    original=BASE/'canonical_gate_band_expert_direction_exact_state_shared_kernel_learning_20260920'
    models=dict(retained=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt',
                parent=original/'research_candidate.pt')
    sources=[__file__,BASE/'check_joint_training_neighborhood.py',BASE/'shared_teacher_kernel_policy.py',
             BASE/'complete_corner_metrics.py',BASE/'learn_nonlinear_joint_feedback.py',
             original/'raw_result_audit.json']
    for name,source in [('additive',additive),('expert_weights',weights)]:
        results=read(source/'results.json')
        assert any(c['research_acceptable'] for c in results['candidates']),name
        sources.extend([source/'results.json',source/'receipt.json',source/'registration.json'])
        for c in results['candidates']:
            if c['research_acceptable']:models[name+'_'+c['tag']]=source/c['tag']/'candidate.pt'
    parent_results=read(original/'results.json')
    parent=next(c for c in parent_results['candidates'] if c['tag']==parent_results['best'])
    cases={}
    for path in [1,12,14,15]:
        record=next(r for r in parent['records'] if r['path']==path)
        with np.load(original/record['trace']) as z:reference=z['reference'].copy()
        for kind in ('incoming_length','angle'):
            ref=reference.copy()
            if kind=='incoming_length':ref[0]=ref[1]+.9925*(ref[0]-ref[1])
            else:
                theta=np.sign(record['shape']['turn_deg'])*.25*np.pi/180
                rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                ref[2]=ref[1]+rotation@(ref[2]-ref[1])
            cases[300+len(cases)]=dict(source_path=path,kind=kind,reference=ref.tolist())
    stage=register('canonical_gate_band_joint_state_representation_neighborhood_20260920',
        sources+list(models.values()),dict(wall_s=240,workers=3,new_training_paths=8,arms=list(models),
            incoming_length_scale=.9925,signed_angle_change_deg=.25,original80_repeated=0),
        'Exact teacher-state interpolation and successful full80 reproduction do not '
        'establish robust feedback away from those states. Compare the two preregistered '
        'state-function widths for additive residuals and learned feedback expert weights '
        'on eight NEW explicitly TRAINING geometries:0.75% incoming-edge shortening or '
        '0.25degree signed-angle increase from original paths1,12,14,15. Reuse the same '
        'teacher data, parent and learning screens for the representation contrast. Also '
        'measure retained86 and the pre-update8052 parent on these new geometries only. '
        'Preserve every failure and actual complete time; no failed prefix is efficiency. '
        'These are eligible learning/selection cases, not independent/frozen/sealed tests. '
        'All80 completed paths are reused, not rerun. Require actual inward onset, rawXY '
        'reverse lateral displacement/duration, two-edge excursions,heading,curvature, '
        'bend speed,cruise,coverage,braking/rest and unchanged1.25mm,V/A/J,20/1ms. '
        'If both representations remain brittle, revise the state learning mechanism; '
        'do not claim generalization from exact training collocation.')
    write_json(stage/'training_cases.json',cases)
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        pool=mp.get_context('spawn').Pool(3,initialize,(stage,models,cases,started+225))
        jobs=[(name,p) for name in models for p in cases]
        for r in pool.imap_unordered(evaluate,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(records=outcomes,training_only=True))
            print('REPRESENTATION_NEIGHBORHOOD',r['arm'],r['path'],r['success'],
                  r['shape']['full_physical_reverse_deg'],r['shape']['out_edge_outward_mm'],flush=True)
        pool.close();pool.join();pool=None
        summaries={name:aggregate([r for r in outcomes if r['arm']==name]) for name in models}
        write_json(stage/'results.json',dict(records=outcomes,summaries=summaries,training_only=True))
        receipt(stage,dict(status='completed_joint_state_representation_neighborhood',summaries=summaries,
            model_sha256={k:sha(v) for k,v in models.items()},wall_s=time.perf_counter()-started,
            training_only=True,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_joint_state_representation_neighborhood',error=str(exc),
            completed=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':mp.freeze_support();main()

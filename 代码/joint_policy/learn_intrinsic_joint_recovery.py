"""Correct an actual physical-curvature regression using full-loop joint credit."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
from functools import lru_cache
import numpy as np
import torch
from intrinsic_curve_metrics import intrinsic_metrics
from audit_context_restoration_scaling import solve_correction
from duration_joint_metrics import guards,penalty
from cruise_duration_credit import event_learning_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


@lru_cache(maxsize=2048)
def intrinsic(trace):
    with np.load(trace) as z:
        result=intrinsic_metrics(z['rows']);u,v=np.diff(z['reference'],axis=0)
    angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)
    return np.array([result['peak_curvature_mm_inv']/.2,
                     1.25*result['bending_integral_mm_inv']/angle**2/.02])


def intrinsic_guards(records,baseline):
    return np.concatenate([intrinsic(str(b['trace']))-intrinsic(str(r['trace']))
        if r['success'] else np.full(2,-100.) for r,b in zip(records,baseline)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--positive-margin',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    base=BASE/'canonical_gate_band_centered_metric_joint_recovery_20260920'
    source=BASE/'canonical_gate_band_intrinsic_curvature_joint_recovery_20260920' if args.positive_margin else base
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    assert read(base/'selection_hold.json')['intrinsic_peak_regression_path']==204
    model_file=source/('state_joint/candidate.pt' if args.positive_margin else 'research_candidate.pt')
    md=torch.load(model_file,map_location='cpu',weights_only=False)
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    index={c['tag']:c for c in history};zero,early=history[:2]
    current=index['intrinsic_joint_5' if args.positive_margin else 'local_joint_46'];origin=index['joint_transfer_0']
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    active_paths=read(source/'metric_zero_support_audit.json')['affected_paths']
    subspace=read(base/'subspace.json');basis=np.asarray(subspace['basis']);steps=np.asarray(subspace['steps'])
    wall_s=90 if args.positive_margin else 180
    stage=register('canonical_gate_band_positive_margin_joint_recovery_20260920' if args.positive_margin else
                   'canonical_gate_band_intrinsic_curvature_joint_recovery_20260920',
        [__file__,BASE/'intrinsic_curve_metrics.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'duration_joint_metrics.py',BASE/'cruise_duration_credit.py',
         BASE/'heading_duration_credit.py',BASE/'audit_context_restoration_scaling.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         source/'receipt.json',model_file,source/'state_joint/results.json',
         base/'selection_hold.json',base/'intrinsic_curve_audit.json',base/'energy_resolution_audit.json',
         base/'measured_local_credit.json',base/'local_recovery.json',base/'subspace.json',parent_file]+
         ([source/'positive_margin_capacity.json',source/'recovery_audit.json',
           BASE/'canonical_gate_band_bounded_output_precision_probe_20260920/matched_results.json'] if args.positive_margin else []),
        dict(wall_s=wall_s,workers=4,maximum_new_vectors=2 if args.positive_margin else 8,maximum_corrections=2 if args.positive_margin else 6,
             correction_radius_choices=[.25,1.,4.],coefficient_bounds=[-1.,1.]),
        ('The last six guarded corrections predict exact boundary feasibility and '
         'leave6.03e-7 actual violation. Three matched output-float64 controls worsen '
         'the original guards, so keep original float32 actions. Offline LP audit '
         'finds positive normalized model margins up to1e-6 feasible. Compare TWO '
         'new corrections from the identical intrinsic_joint_5 anchor and identical '
         'measured Jacobian: margin0 versus margin1e-6. These stricter predicted '
         'margins do not relax any real guard. Maximum2 vectors/90s, with no '
         'adaptive continuation inside this matched comparison. '
         if args.positive_margin else '')+
        'The46 candidate passed all original physical/shape/time/cruise guards, but '
        'raw V/A curvature increased on204 by0.000226764/mm. Hold it. Add two '
        'raw speed>5 physical-heading-validity metrics to the previous per-route '
        'protections: maximum abs(cross(V,A))/speed^3 and its bending integral. '
        'Keep every original XY/spatial/count/low-speed/stop check. Recompute these '
        'metrics on existing symmetric response traces and measured recovery secants; '
        'no old experiment is repeated. Learn at most6 joint corrections plus2 '
        'backtracks in the same audited state/history function, preserving the learned '
        'early prefix and zero support on81 parent routes. Validate complete routes '
        'for every new vector. Stop this mechanism if the augmented response is '
        'infeasible; do not weaken guards. Complete-loop parameter learning, not PPO.')
    started=time.perf_counter();deadline=started+wall_s-15;folder=stage/'state_joint';folder.mkdir()
    md['reused_candidates']=len(history)
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'metric_zero_support_audit.json',read(source/'metric_zero_support_audit.json'))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def original_guards(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    for c in history:
        extra=intrinsic_guards(c['records'],selected)
        c['intrinsic_inequalities']=extra.tolist()
        c['augmented_measured_inequalities']=np.r_[c['measured_inequalities'],extra].tolist()
        c['augmented_learning_inequalities']=np.r_[event_learning_guards(c['records'],selected,c['measured_inequalities']),extra].tolist()
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['augmented_measured_inequalities'],0.)))
    model=read(base/'measured_local_credit.json');values=np.asarray(origin['augmented_learning_inequalities'])
    jac=np.stack([np.asarray(index[p['plus']]['augmented_learning_inequalities'])-
                  np.asarray(index[p['minus']]['augmented_learning_inequalities']) for p in model['pairs']],axis=1)
    gradient=np.asarray(model['gradient'])
    def coords(c):return np.asarray(c.get('local_coordinates',np.zeros(basis.shape[1])))
    for block in read(base/'local_recovery.json')['blocks']:
        if 'tag' not in block:continue
        a,b=index[block['anchor']],index[block['tag']];d=coords(b)-coords(a);denom=d@d
        jac+=np.outer(np.asarray(b['augmented_learning_inequalities'])-a['augmented_learning_inequalities']-jac@d,d)/denom
        gradient+=(b['score']-a['score']-gradient@d)*d/denom
    if args.positive_margin:
        latest=read(source/'recovery_audit.json')
        jac=np.asarray(latest['final_jacobian']);gradient=np.asarray(latest['loss_gradient'])
    write_json(stage/'augmented_credit.json',dict(jacobian=jac.tolist(),gradient=gradient.tolist(),
        source_anchor=current['tag'],additional_guard_names=['physical_peak_curvature','physical_bending_integral']))
    cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    prefix={r['path']:r['first_later_kernel_ms'] for r in read(BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920/conditioned_support_audit.json')['records']}
    prefix_rows={}
    for r in early['records']:
        with np.load(r['trace']) as z:prefix_rows[r['path']]=z['rows'][:prefix[r['path']]].copy()
    def expand(v):
        out=np.zeros((3,7,5,3));out[:,:,[0,1,3,4]]=np.asarray(v).reshape(3,7,4,3)
        return out.reshape(-1)
    blocks=[];pool=None;backtracks=0
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                assert len(history)-md['reused_candidates']<8
                assert parameters_in_bounds(u,-1.,1.)
                prefix_name='margin_joint' if args.positive_margin else 'intrinsic_joint'
                tag=f'{prefix_name}_{len(history)-md["reused_candidates"]}'
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in active_paths])
                by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
                for r in measured:
                    with np.load(r['trace']) as z:assert np.array_equal(z['rows'][:prefix[r['path']]],prefix_rows[r['path']])
                original=original_guards(records);extra=intrinsic_guards(records,selected)
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=original.tolist(),intrinsic_inequalities=extra.tolist(),
                    augmented_measured_inequalities=np.r_[original,extra].tolist(),
                    augmented_learning_inequalities=np.r_[event_learning_guards(records,selected,original),extra].tolist(),
                    actual_rerun_paths=active_paths,exact_zero_support_paths=[p for p in paths if p not in active_paths],
                    early_prefix_bitwise=True)
                history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('INTRINSIC_JOINT',tag,c['score'],min(c['augmented_measured_inequalities']),violation(c),flush=True)
            return cache[key]
        for iteration in range(2 if args.positive_margin else 6):
            values=np.asarray(current['augmented_learning_inequalities']);models=[]
            margin=(0.,1e-6)[iteration] if args.positive_margin else 0.
            for radius in (.25,1.,4.):
                solved=solve_correction(jac,gradient,values,early['score']-current['score'],radius=radius,margin=margin)
                models.append(dict(radius=radius,**solved))
                if solved['success']:break
            row=dict(iteration=iteration,anchor=current['tag'],predicted_margin=margin,models=models);blocks.append(row)
            if not solved['success']:break
            delta=np.asarray(solved['scaled_parameter_step']);u=np.asarray(current['coefficients'])
            step=expand(steps*(basis@delta));candidate=evaluate(u+step)
            row.update(tag=candidate['tag'],violation=violation(candidate))
            if args.positive_margin:continue
            if feasible(candidate):current=candidate;break
            denom=delta@delta
            jac+=np.outer(np.asarray(candidate['augmented_learning_inequalities'])-values-jac@delta,delta)/denom
            gradient+=(candidate['score']-current['score']-gradient@delta)*delta/denom
            if violation(candidate)>=violation(current) and backtracks<2 and len(history)-md['reused_candidates']<8:
                half=evaluate(u+.5*step);backtracks+=1
                if violation(half)<violation(candidate):candidate=half
                if feasible(candidate):current=candidate;break
            if violation(candidate)<violation(current):current=candidate
        if args.positive_margin and best()['score']<zero['score']:current=best()
        write_json(stage/'recovery_audit.json',dict(blocks=blocks,nearest=current['tag'],
            final_jacobian=jac.tolist(),loss_gradient=gradient.tolist(),new_vectors=len(history)-md['reused_candidates']))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_intrinsic_joint_recovery',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()

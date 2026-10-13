"""Learn complete-loop local joint responses at an actual near-feasible policy."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from audit_context_restoration_scaling import solve_correction
from audit_remaining_joint_credit import names_for
from duration_joint_metrics import guards, penalty
from heading_duration_credit import learning_guards
from cruise_duration_credit import event_learning_guards
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE, register, receipt, write_json
from search_complete_return import read
from joint_teacher_state_policy import features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bend-metric',action='store_true')
    parser.add_argument('--centered-metric',action='store_true')
    args=parser.parse_args()
    if args.centered_metric:args.bend_metric=True
    credit_guards=event_learning_guards if args.centered_metric else learning_guards
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    response=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    anchor=next(c for c in history if c['tag']=='geometry_secant_17')
    zero,early=history[:2]
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    old=read(source/'final_secant_credit.json');prior=np.asarray(old['jacobian'])
    values=np.asarray(anchor['learning_inequalities'])
    names=names_for(selected)
    zero_paths=[]
    if args.bend_metric:
        capacity=read(source/('centered_bend_metric_capacity.json' if args.centered_metric else 'bend_metric_capacity.json'))
        md.update(observed_bend_metric_centers=np.asarray(capacity['centers']),
                  observed_bend_metric_weights=np.asarray(capacity['choices'][0]['weights']))
        with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
        f=features(md,training['observations'],training['innovations'])
        active_paths=sorted(np.unique(training['path_ids'][np.any(f!=0.,axis=(1,2))]).tolist())
        assert active_paths==[14,104,105,204,205,304,305]
        assert len(np.unique(training['path_ids']))==len(baseline)
        zero_paths=[p for p in paths if p not in active_paths]
        # Only identity and the unchanged early-only function remain reusable.
        assert not np.any(np.asarray(early['coefficients']).reshape(3,7,5,3)[:,:,[0,1,3,4]])
        history=[zero,early]
        if args.centered_metric:
            transfer=BASE/'canonical_gate_band_centered_bend_function_transfer_20260920'
            anchor=copy.deepcopy(next(c for c in read(transfer/'state_joint/results.json')['candidates'] if c['tag']=='joint_transfer_0'))
            history.append(anchor)
            for c in history:
                c['learning_inequalities']=credit_guards(c['records'],selected,c['measured_inequalities']).tolist()
            values=np.asarray(anchor['learning_inequalities'])
    # The prior only selects a small joint exploration subspace. It is NOT used
    # as the new measured response inside that subspace.
    active=(values<.003)&(np.linalg.norm(prior,axis=1)>1e-12)
    if args.bend_metric:active&=np.asarray([int(name.split(':')[0]) not in zero_paths for name in names])
    normalized=prior[active]/np.linalg.norm(prior[active],axis=1)[:,None]
    _,singular,vh=np.linalg.svd(normalized,full_matrices=False)
    dimensions=min(20,int((singular>singular[0]*1e-8).sum()))
    basis=vh[:dimensions].T
    steps=np.repeat(np.asarray(read(response/'joint_main_turn_credit_models.json')['steps'])[:,None],7,axis=1).reshape(-1)
    def expand(z):
        v=np.asarray(z).reshape(3,7,4,3);out=np.zeros((3,7,5,3))
        out[:,:,[0,1,3,4],:]=v
        return out.reshape(-1)
    u0=np.asarray(anchor['coefficients'])
    wall_s=300 if args.centered_metric else 480
    stage=register('canonical_gate_band_centered_metric_joint_recovery_20260920' if args.centered_metric else
                   'canonical_gate_band_bend_metric_joint_learning_20260920' if args.bend_metric else
                   'canonical_gate_band_recentered_joint_subspace_learning_20260920',
        [__file__,BASE/'audit_remaining_joint_credit.py',BASE/'observable_memory_joint_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'heading_duration_credit.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',BASE/'audit_context_restoration_scaling.py',
         parent_file,source/'receipt.json',source/'state_joint/candidate.pt',source/'state_joint/results.json',
         source/'remaining_credit_audit.json',source/'final_secant_credit.json',BASE/'joint_teacher_state_policy.py']+
         ([BASE/'audit_bend_metric_capacity.py',source/'bend_metric_capacity.json',
           source/'geometry_support_audit.json',
           BASE/'canonical_gate_band_recentered_joint_subspace_learning_20260920/measured_credit_quality.json'] if args.bend_metric else [])+
         ([BASE/'cruise_duration_credit.py',transfer/'receipt.json',transfer/'fit_audit.json',
           transfer/'state_joint/results.json',source/'centered_bend_metric_capacity.json'] if args.centered_metric else []),
        dict(wall_s=wall_s,workers=4,subspace_dimensions=dimensions,symmetric_amplitude=.5,
             doubled_amplitude_controls=2,maximum_new_vectors=2*dimensions+4+6+int(args.bend_metric and not args.centered_metric),coefficient_bounds=[-1.,1.]),
        ('Start from the corrected metric with bounded joint-function transfer: all '
         'three float32 actions are exactly preserved at350 stored decisions of the '
         'seven target routes, and the other81 protected routes have zero new support. '
         'The initial point therefore reuses genuine existing trajectories, not a '
         'new rollout. Prior mixed-context distance centers were incorrect and their '
         'responses are NOT reused. Collect fresh responses for the corrected policy. '
         'Use exact count-equivalent heading AND cruise-event margins during learning, '
         'and every original count/shape/physical guard for selection. Budget300s. '
         if args.centered_metric else '')+
        ('Fresh20-direction responses remained infeasible in the sampled subspace. '
         'Exact state replay shows the same radial kernels also touch six distinct '
         'visible-geometry states, producing conflicting small guard changes. Add a '
         'nonnegative squared distance in the EXISTING continuous reference-bend sensor, '
         'with the smallest audited training-derived weight that excludes those states. '
         'This only shrinks support, retains all seven difficult routes, and preserves '
         'the early kernel. Prove zero support on all88 protected parent trajectories; '
         'reuse the six exact-zero functions and collect fresh responses for the seven '
         'affected complete routes. No angle/speed table, route selector, or template. '
         'Do not reuse nonzero old-function outcomes. First measure the changed function '
         'with old learned coefficients, then learn three-action weights from actual '
         'local symmetric responses if it is not acceptable. '
         if args.bend_metric else '')+
        'The actual geometry_secant_17 still violates20 original per-route guards. '
        'Sequential Broyden updates fit the last direction but disagree with other measured '
        'directions. Recenter physical response collection at that actual policy. Select20 '
        'orthogonal joint parameter directions from normalized near-active guard rows of '
        'the old response; completely REPLACE their local Jacobian by fresh symmetric '
        'full-route measurements at amplitude0.5. Two amplitude1 controls check local '
        'nonlinearity/rounding. All three outputs and causal memory coordinates vary together. '
        'Freeze the already learned early kernel and verify the true early motion prefix. '
        'Learn bounded corrections from measured local responses with exact continuous '
        'heading-count slack; all original guards remain in final selection. Maximum480s. '
        'If response is inconsistent or no feasible correction exists, change representation '
        'rather than extending the same optimizer. Closed-loop parameter learning, not PPO.')
    started=time.perf_counter();deadline=started+wall_s-25;folder=stage/'state_joint';folder.mkdir()
    md['reused_candidates']=len(history)
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    if args.bend_metric:
        write_json(stage/'metric_zero_support_audit.json',dict(parent_paths=len(baseline),
            affected_paths=active_paths,zero_paths=zero_paths,
            all_unaffected_paths=[r['path'] for r in baseline if r['path'] not in active_paths],
            support_can_only_shrink=True,early_kernel_unchanged=True,
            interpretation='Exact zero added features at every protected parent decision '
            'imply identical actions and physical states by deterministic induction.'))
    write_json(stage/'subspace.json',dict(anchor=anchor['tag'],basis=basis.tolist(),steps=steps.tolist(),
        active_guard_names=[name for name,flag in zip(names,active) if flag],singular_values=singular.tolist()))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['measured_inequalities'],0.)))
    prefix={r['path']:r['first_later_kernel_ms'] for r in read(BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920/conditioned_support_audit.json')['records']}
    prefix_rows={}
    for r in early['records']:
        with np.load(r['trace']) as z:prefix_rows[r['path']]=z['rows'][:prefix[r['path']]].copy()
    cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    blocks=[];pool=None
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(v,kind):
            u=u0+expand(steps*(basis@v));key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                tag=f'local_joint_{len(history)-md["reused_candidates"]}'
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in paths if p not in zero_paths])
                by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
                for r in records:
                    with np.load(r['trace']) as z:actual=z['rows'][:prefix[r['path']]].copy()
                    assert np.array_equal(actual,prefix_rows[r['path']]),('early_prefix_changed',tag,r['path'])
                original=constraints(records)
                c=dict(tag=tag,kind=kind,coefficients=u.tolist(),local_coordinates=np.asarray(v).tolist(),
                    records=records,score=score(records),measured_inequalities=original.tolist(),
                    learning_inequalities=credit_guards(records,selected,original).tolist(),
                    actual_rerun_paths=[p for p in paths if p not in zero_paths],
                    exact_zero_support_paths=zero_paths,early_prefix_bitwise=True)
                history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('RECENTERED_JOINT',tag,kind,c['score'],min(original),violation(c),flush=True)
            return cache[key]
        if args.bend_metric:
            anchor=evaluate(np.zeros(dimensions),'changed_observation_metric')
            values=np.asarray(anchor['learning_inequalities'])
            if feasible(anchor) and anchor['score']<zero['score']-.001:
                pool.close();pool.join();pool=None
                finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
                return
        pairs=[];jac=[];gradient=[];curvature=[]
        for j in range(dimensions):
            direction=np.eye(dimensions)[j]*.5
            plus=evaluate(direction,'central_plus');minus=evaluate(-direction,'central_minus')
            p,m=np.asarray(plus['learning_inequalities']),np.asarray(minus['learning_inequalities'])
            jac.append(p-m);gradient.append(plus['score']-minus['score'])
            curvature.append(p+m-2*values)
            pairs.append(dict(direction=j,plus=plus['tag'],minus=minus['tag']))
            write_json(stage/'response_pairs.json',pairs)
        jac=np.asarray(jac).T;gradient=np.asarray(gradient);curvature=np.asarray(curvature).T
        controls=[]
        for j in range(min(2,dimensions)):
            direction=np.eye(dimensions)[j]
            plus=evaluate(direction,'amplitude_control_plus');minus=evaluate(-direction,'amplitude_control_minus')
            derivative=(np.asarray(plus['learning_inequalities'])-np.asarray(minus['learning_inequalities']))/2
            controls.append(dict(direction=j,derivative_error=(derivative-jac[:,j]).tolist(),
                plus=plus['tag'],minus=minus['tag']))
        write_json(stage/'measured_local_credit.json',dict(jacobian=jac.tolist(),gradient=gradient.tolist(),
            even_response=curvature.tolist(),pairs=pairs,amplitude_controls=controls))
        current=anchor;v=np.zeros(dimensions)
        for iteration in range(6):
            if time.perf_counter()>deadline-12:break
            local=np.asarray(current['learning_inequalities'])
            proposals=[]
            for radius in (.25,1.,4.):
                solved=solve_correction(jac,gradient,local,early['score']-current['score'],radius=radius)
                proposals.append(dict(radius=radius,**solved))
                if solved['success']:break
            row=dict(iteration=iteration,anchor=current['tag'],models=proposals)
            blocks.append(row)
            if not solved['success']:break
            delta=np.asarray(solved['scaled_parameter_step'])
            candidate=evaluate(v+delta,'measured_local_correction')
            row.update(tag=candidate['tag'],actual_violation=violation(candidate),
                prediction_error=(np.asarray(candidate['learning_inequalities'])-local-jac@delta).tolist())
            if feasible(candidate):break
            denom=float(delta@delta)
            if denom<1e-20:break
            jac+=np.outer(np.asarray(candidate['learning_inequalities'])-local-jac@delta,delta)/denom
            gradient+=(candidate['score']-current['score']-gradient@delta)*delta/denom
            if violation(candidate)<violation(current):current=candidate;v=v+delta
        write_json(stage/'local_recovery.json',dict(blocks=blocks,nearest=current['tag'],
            original_guard_failures=[dict(name=names[i],value=float(x)) for i,x in enumerate(current['measured_inequalities']) if x < -1e-8]))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_recentered_joint_subspace',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()

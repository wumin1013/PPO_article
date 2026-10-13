"""Sequential full-path proposals with explicit exact brake-branch re-evaluation."""
import argparse
import time
import os
import numpy as np
import torch
from scipy.linalg import solve_triangular
from canonical_gate_state import checkpoint_environment_factory
from transition_waveform_model import chain_responses, signal_jacobians
from terminal_branch_affine import branch_affine
from terminal_completion_witness import TerminalCompletionWitness
from event_adaptive_terminal_prediction import predict_terminal
from waveform_constraint_model import waveform_facets
from capsule_corridor_model import CapsuleCorridorModel
from cruise_update_geometry import CruiseUpdateGeometry
from convex_cut_solver import solve_convex_cuts
from audit_ordinary_joint_errors import metrics
from audit_function_constraint_interference import command_segment_margin
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha
from probe_closed_loop_tangent_credit import STATE_SCALE,ACTION_SCALE
from joint_runtime import wrap

SHAPE=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
       'negative_signed_heading_deg','heading_overshoot_deg')
ENERGY=('full_xy_normalized_energy','full_physical_normalized_energy')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_complete_branch_joint_proposal_20260918')
    parser.add_argument('--start-from')
    parser.add_argument('--interior-reserve',type=float,default=0.)
    parser.add_argument('--objective',choices=('minimum_cruise_deficit','minimum_shape','minimum_shape_energy'),default='minimum_cruise_deficit')
    parser.add_argument('--completion-witness',action='store_true')
    parser.add_argument('--state-validity-cut',action='store_true')
    parser.add_argument('--audit-direction',help='Re-evaluate a saved direction without optimization or physical feedback')
    parser.add_argument('--audit-scale',type=float,default=1.)
    parser.add_argument('--restore-feasibility',action='store_true',help='One nearest-point constraint restoration, not objective continuation')
    parser.add_argument('--preserve-pure-feed',action='store_true')
    parser.add_argument('--extra-credit')
    parser.add_argument('--extra-reference')
    args=parser.parse_args()
    assert 0<=args.interior_reserve<=.001
    assert 0<args.audit_scale<=1. and (args.audit_direction or args.audit_scale==1.)
    assert not args.restore_feasibility or args.audit_direction
    assert bool(args.extra_credit)==bool(args.extra_reference)
    source=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    reference=BASE/'canonical_gate_band_shape_priority_current_reference_20260918'
    geom_source=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    calibration=BASE/'canonical_gate_band_ordinary_joint_error_sources_20260918'
    prior_proposal=BASE/'canonical_gate_band_ordinary_joint_proposal_20260918'
    data,ref_data=read(source/'results.json'),read(reference/'results.json')
    extension_sources=[]
    if args.extra_credit:
        extra=BASE/args.extra_credit;extra_reference=BASE/args.extra_reference
        extra_data=read(extra/'results.json');extra_ref=read(extra_reference/'results.json')
        assert extra_data['parent_sha256']==extra_ref['parent_sha256']==data['parent_sha256']
        original_paths={r['path'] for r in data['records']}
        added_paths={r['path'] for r in extra_data['records']}
        assert added_paths.isdisjoint(original_paths) and all(p>=16 for p in added_paths)
        for record in extra_data['records']:
            local_file=(extra/record['file']).resolve()
            data['records'].append(dict(record,file=os.path.relpath(local_file,source)))
            extension_sources.append(local_file)
        ref_data['baseline']['metrics']+=extra_ref['baseline']['metrics']
        assert {r['path'] for r in ref_data['baseline']['metrics']}==original_paths|added_paths
        extension_sources += [extra/'results.json',extra/'receipt.json',extra_reference/'results.json',extra_reference/'receipt.json']
    errors=read(calibration/'results.json');previous=read(prior_proposal/'results.json')
    parent=BASE/data['parent_model']
    assert sha(parent)==ref_data['parent_sha256']==errors['parent_sha256']==previous['parent_sha256']
    geometry=CruiseUpdateGeometry(geom_source,'physical',sha(parent))
    pure_feed_features=None
    if args.preserve_pure_feed:
        training=read(BASE/'learning_preparation_v4/paths.json')['training']
        pure_paths={i for i,p in enumerate(training) if len(p)==2}
        with np.load(geom_source/'geometry.npz') as z:
            ids=z['cruise_state_indices'];mask=np.isin(z['state_path'][ids],list(pure_paths))
            pure_feed_features=z['cruise_features'][mask].copy()
        assert len(pure_feed_features)==599
        extension_sources += [BASE/'learning_preparation_v4/paths.json']
    factory=checkpoint_environment_factory(torch.load(parent,map_location='cpu',weights_only=False)['architecture'])
    warm_sources=[];initial=np.zeros(768)
    audited_direction=None
    if args.audit_direction:
        audit_source=BASE/args.audit_direction
        assert read(audit_source/'results.json')['parent_sha256']==sha(parent)
        with np.load(audit_source/'direction.npz') as z:audited_direction=args.audit_scale*z['direction'].copy().ravel()
        warm_sources += [audit_source/'direction.npz',audit_source/'results.json',audit_source/'receipt.json']
    if args.start_from:
        warm=BASE/args.start_from;warm_data=read(warm/'results.json')
        assert warm_data['parent_sha256']==sha(parent)
        assert not warm_data['history'][-1]['changed_terminal_modes']
        with np.load(warm/'direction.npz') as z:initial=z['direction'].copy().ravel()
        warm_sources += [warm/'direction.npz',warm/'results.json',warm/'receipt.json',
            BASE/'canonical_gate_band_complete_branch_interior_20260918/results.json',
            BASE/'canonical_gate_band_complete_branch_interior_20260918/receipt.json']
    if args.completion_witness:
        audit=BASE/'canonical_gate_band_completion_witness_audit_20260918'
        assert read(audit/'results.json')['all_passed']
        assert read(audit/'receipt.json')['all29_passed']
        warm_sources += [audit/'results.json',audit/'receipt.json',
            BASE/'canonical_gate_band_complete_shape_completion_causes_20260918/results.json']
    state_cut=None
    if args.state_validity_cut:
        cause=BASE/'canonical_gate_band_failed_completion_transition_cause_20260918'
        scales=BASE/'canonical_gate_band_transition_state_scale_audit_20260918'
        feedback=BASE/'canonical_gate_band_complete_completion_feedback_20260918'
        evidence=read(cause/'results.json');assert evidence['parent_sha256']==sha(parent)
        record=evidence['summary']['largest_error_contributors'][0]
        path=evidence['path'];decision=record['decision'];assert (path,decision)==(31,25)
        valid=next(r for r in read(scales/'results.json')['records'] if r['scale']==.25)
        assert valid['propagated_xy_error_norm_mm']<.001
        failed_candidate=read(feedback/'results.json')['candidates'][0]
        j=next(j for j,r in enumerate(failed_candidate['results']) if r['original_path']==path)
        trace=BASE/failed_candidate['traces'][j];item=next(r for r in data['records'] if r['path']==path)
        with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
        with np.load(trace) as z:actual=z['rows'].copy()
        tick=decision*20-1;nominal=local['rows'][tick]
        delta=np.r_[(actual[tick,:6]-nominal[:6])/STATE_SCALE[:6],
            wrap(np.arctan2(actual[tick,9],actual[tick,8])-np.arctan2(nominal[9],nominal[8]))]
        radius_state=.25*float(np.linalg.norm(delta));response_state=np.zeros((7,768))
        for k in range(decision):
            direct=np.zeros((3,768))
            for h in range(3):direct[h,h*256:(h+1)*256]=local['features'][k,h]/ACTION_SCALE[h]
            sj=local['state_jacobian'][k]
            response_state=sj[:,:7]@response_state+sj[:,7:]@direct
        state_cut=dict(path=path,decision=decision,radius=radius_state,response=response_state)
        warm_sources += [cause/'results.json',cause/'receipt.json',scales/'results.json',scales/'receipt.json',
            scales/'mode_support_review.json',trace,source/item['file']]
    stage=register(args.stage,
        [__file__,BASE/'terminal_branch_affine.py',BASE/'terminal_completion_witness.py',BASE/'event_adaptive_terminal_prediction.py',
         BASE/'transition_waveform_model.py',BASE/'waveform_constraint_model.py',
         BASE/'terminal_duration_credit.py',BASE/'convex_cut_solver.py',parent,
         source/'results.json',source/'receipt.json',reference/'results.json',
         geom_source/'geometry.npz',geom_source/'results.json',geom_source/'receipt.json',
         calibration/'results.json',calibration/'receipt.json',prior_proposal/'results.json']+warm_sources+extension_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=720,
             audit_saved_direction_without_optimization=args.audit_direction,
             audited_direction_scale=args.audit_scale,
             nearest_feasible_restoration=args.restore_feasibility,
             preserved_pure_feed_states=len(pure_feed_features) if pure_feed_features is not None else 0,
             extra_credit=args.extra_credit,extra_reference=args.extra_reference,
             restoration_optimizer_wall_s=60 if args.restore_feasibility else 0,
             maximum_branch_relinearizations=3,maximum_exchange_rounds_per_branch=30,
             physical_metric_radius=.1,normalized_saved_action_bound=.5,
             complete_shape_reduction=.01,terminal_planner_queries=4500,
             start_from=args.start_from,positive_shape_ceiling_interior_fraction=args.interior_reserve,
             optimization_objective=args.objective,nominal_duration_feasibility_witness=args.completion_witness,
             exact_distance_enclosure_tolerance_mm=1e-10,
             state_validity_cut={k:v for k,v in state_cut.items() if k!='response'} if state_cut else None),
        (('The preceding half-step passed complete corner shape but lost171ms of '
          'pure-straight cruise and delayed path19 by20ms. Apply a DISCLOSED '
          'combined support correction: preserve nondecreasing feed on599 saved '
          'pure-straight interior states via exact affine latent inequalities, '
          'and add only the newly collected missing training-path credit and '
          'its same completion/corridor witnesses when specified. This does not '
          'identify either factor as the sole cause of any subsequent policy '
          'improvement. Nondecreasing saved requests do not certify actual '
          'cruise under changed states; all64 complete rollouts remain required. '
          'No runtime rule, angle speed table, physical relaxation or hidden '
          'reuse of rejected candidates. ' if args.preserve_pure_feed or args.extra_credit else '')+
         ('Perform ONE at-most60s nearest-point feasibility restoration of the saved '
          'shape direction in whitened parameter coordinates. The measured lone '
          'state-validity residual was3.497e-5; radial contraction fixed that '
          'residual but violated active fixed-capsule/curvature constraints. '
          'Keep ALL original constraints and1e-7 admission thresholds. Minimize '
          'squared displacement from that direction instead of resuming the '
          'shape-objective solve. Re-evaluate exact hybrid branches afterward; '
          'physical feedback remains forbidden unless both checks pass. No '
          'further restoration rounds or automatic adoption are authorized. '
          if args.restore_feasibility else '')+
         ('Audit only the already saved direction at the EXACT original first-round '
          'anchor and constraints. Do not run the conic optimizer. Record the '
          'largest named conditional constraint residuals separately from epigraph '
          'and exact-hybrid checks. Keep the old ineligible receipt unchanged. '
          'This diagnoses what blocked the prior shape candidate; it does not '
          'authorize physical feedback or relax a threshold. ' if args.audit_direction and not args.restore_feasibility else '')+
         ('Change only the optimization objective to the worst positive per-corner '
          'fraction of shape, plus the two existing curvature energies only when '
          'objective=minimum_shape_energy. Retain the same anchor, empirical coefficients, radius, '
          'support, complete brake branching and actual acceptance criteria as '
          'the interior speed-request arm. This tests objective competition; '
          'all three shared action heads remain trainable. No new metrics or '
          'runtime path rules. ' if args.objective!='minimum_cruise_deficit' else '')+
         'The prior fixed-budget solve reached stable modes with one2.872e-7 '
         'relative exact-shape violation; moving toward zero worsened another '
         'tail heading because its allowed region is branch dependent. Reuse '
         'that solved anchor and TIGHTEN each positive model shape ceiling by '
         f'{args.interior_reserve:g} of its original value, leaving every actual '
         'acceptance threshold, empirical error rate, radius and physical '
         'condition unchanged. This tests an interior correction, not a '
         'relaxed acceptance or indefinite repetition of the old solve. '
         if args.start_from else '')+
        ('Add one local-model validity cut at the observed dominant error-injection '
         'state. Bound the predicted seven-state displacement to one quarter '
         'of the measured failed-state displacement norm; this quarter ray had '
         'less than0.001mm propagated error in a fixed finite test. This is '
         'an empirical, direction-limited trust neighborhood, NOT a guarantee '
         'for other directions. It is a training optimization constraint only, '
         'not a deployed path/angle/tick rule. Keep other objectives, parameter '
         'bounds and physical acceptance unchanged, then test one complete '
         'feedback candidate. Do not merely inflate scalar error coefficients. '
         if state_cut else '')+
        ('Add a sufficient offline witness at every parent initiation state: '
         'the parent brake duration remains physically feasible and its endpoint '
         'keeps half the existing1mm planner admission margin. No shorter-duration '
         'infeasibility conditions are added, so shorter and earlier braking stay '
         'allowed. This addresses the measured position-lag missed event, not '
         'by modifying the runtime. The exact planner may select a shorter count '
         'with another endpoint; explicitly recheck all exact events and actual '
         'complete time. No asserted universal time bound. ' if args.completion_witness else '')+
        'Address both observed ordinary optimism and missing full brake-tail '
        'shape. Reuse current399a credit. Use empirical actual-minus-exact-hybrid '
        'scalar shape error divided by squared physical parameter norm, fit to '
        'all four already completed parameter scales. These are not prospective '
        'confidence bounds. The objective is explicitly recorded; retain '
        'radius0.1, saved-action support and every per-corner full shape/curvature '
        'ceiling plus half corridor margin. At each proposal anchor, query the '
        'UNCHANGED original terminal planner and build an affine tail ONLY for '
        'its explicit initiation/count. Verify exact anchor reconstruction, '
        'linearize full geometry there, and solve a conditional convex proposal. '
        'Then rerun ordered geometry and the exact planner at the new parameters '
        'and recompute ALL full-path shape/corridor conditions on that resulting '
        'curve. A changed or infeasible branch cannot be accepted using old-tail '
        'derivatives; at most three relinearizations are permitted. The final '
        'exact-predictor check is still only a prerequisite for complete physical '
        'feedback validation, never adoption or a physical certificate. No '
        'runtime clamps, acceleration resets, ideal templates or test access. '
        'This full-tail treatment and hybrid-error calibration form a disclosed '
        'combined correction; the separate ordinary-error-only arm isolates '
        'ordinary calibration relative to the previously saved experiment.')
    start=time.perf_counter();deadline=start+700;radius=.1
    costs=dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=0)
    metric=geometry.metric;chol=np.linalg.cholesky(metric)
    def unwhite(y):return solve_triangular(chol.T,y,lower=False,check_finite=False)
    def white_grad(g):return solve_triangular(chol,np.asarray(g).T,lower=True,check_finite=False).T
    def budget():
        if time.perf_counter()>=deadline:raise TimeoutError('Complete branch proposal budget')
        assert costs['terminal_planner_queries']<4500
    rates={}
    for r in errors['records']:
        norm2=(r['scale']*previous['metric_norm'])**2
        for key,value in r['actual_full'].items():
            rates[r['path'],key]=max(rates.get((r['path'],key),0.),
                                    (value-r['predicted_hybrid_full'][key])/norm2)
    write_json(stage/'error_model.json',dict(parent_sha256=sha(parent),guaranteed_bound=False,
        source='all64 saved corner responses: actual complete minus exact hybrid complete',
        coefficients=[dict(path=p,metric=k,rate=v) for (p,k),v in rates.items()]))
    base={};margin={}
    for r in ref_data['baseline']['metrics']:
        v=dict(zip(ref_data['metric_names'],r['values']))
        margin[r['path']]=.5*(1.25-v['segment_max_distance_mm'])
        if r['path']<16:base[r['path']]={k:v[k] for k in v if k not in ('segment_max_distance_mm','complete_time_s')}
    totals={k:sum(v[k] for v in base.values()) for k in SHAPE}
    baseline_loss=geometry.objective(np.zeros(768))[0]
    history=[];anchor=initial;accepted=False;last=None

    def exact_checks(x):
        records=[];corrected_totals=dict.fromkeys(SHAPE,0.);all_ok=True
        norm2=float(x@metric@x)
        for item in data['records']:
            budget()
            with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
            rows=local['rows'];stop=int(np.flatnonzero(rows[:,12])[0])
            response=chain_responses(local,x[None,:])[:,:,0]
            try:
                pred=predict_terminal(local['reference'],rows[:,:6]+response,stop,factory,costs,budget)
            except ValueError as error:
                records.append(dict(path=item['path'],supported=False,reason=str(error)));all_ok=False;continue
            r=dict(path=item['path'],supported=pred['supported'])
            if not pred['supported']:
                records.append(r);all_ok=False;continue
            # A1e-6 enclosure cannot decide a1e-8 admission slack at an active
            # capsule bound. Refine computation, never the acceptance ceiling.
            distance=command_segment_margin(local['reference'],pred['rows'][:,:2],tolerance=1e-10)['max_distance_upper_mm']
            r.update(initiation_tick=pred['initiation_tick'],braking_steps=pred['braking_steps'],
                full_time_s=pred['total_ticks']*.001,prefix_inside=pred['predicted_prefix_inside'],
                corridor_upper_mm=distance,corridor_ceiling_mm=1.25-margin[item['path']],
                corridor_pass=distance<=1.25-margin[item['path']]+1e-8,shape_violations=[])
            if not pred['predicted_prefix_inside'] or distance>1.25-margin[item['path']]+1e-8:all_ok=False
            if item['path'] in base:
                raw=metrics(local['reference'],pred['rows'])
                corrected={k:v+rates.get((item['path'],k),0.)*norm2 for k,v in raw.items()}
                r.update(raw_full_shape=raw,corrected_full_shape=corrected)
                for key,value in corrected.items():
                    violation=(value-base[item['path']][key])/max(base[item['path']][key],1e-3)
                    if violation>1e-7:
                        all_ok=False;r['shape_violations'].append(dict(metric=key,scaled_violation=violation))
                for key in SHAPE:corrected_totals[key]+=corrected[key]
            records.append(r)
        reductions={k:corrected_totals[k]/v for k,v in totals.items() if v>1e-8}
        all_ok &= all(v<=.99+1e-7 for v in reductions.values())
        support,_=geometry.support_constraint(x)
        all_ok &= support<=1e-7 and norm2<=radius**2+1e-10
        pure_minimum=float(np.min(pure_feed_features@x[256:512])) if pure_feed_features is not None else None
        if pure_minimum is not None:all_ok &= pure_minimum>=-1e-7
        return dict(eligible_for_actual_probe=bool(all_ok),records=records,
                    minimum_saved_pure_feed_latent_change=pure_minimum,
                    corrected_shape_fractions=reductions,saved_support_violation=support,
                    predicted_complete_time_s=sum(r.get('full_time_s',0.) for r in records)
                        if all(r['supported'] for r in records) else None)

    try:
        for outer in range(3):
            budget();waves={};capsules={};modes={};completion={};identity_max=0.
            for item in data['records']:
                budget()
                with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
                rows,ref=local['rows'],local['reference'];stop=int(np.flatnonzero(rows[:,12])[0])
                response=chain_responses(local)
                if args.completion_witness:
                    count=len(rows)-stop-23
                    completion[item['path']]=TerminalCompletionWitness(rows[stop-1,:6],response[stop-1],count,ref[-1])
                ordinary=rows[:,:6]+np.einsum('tcd,d->tc',response,anchor)
                pred=predict_terminal(ref,ordinary,stop,factory,costs,budget)
                if not pred['supported']:raise ValueError('Anchor requires unsupported later ordinary motion')
                zero,g=branch_affine(ref,rows,response,pred['initiation_tick'],pred['braking_steps'])
                center=zero+np.einsum('tcd,d->tc',g,anchor)
                error=float(abs(center-pred['rows']).max());identity_max=max(identity_max,error);assert error<1e-7
                # Exact selected branch is a linear map in splice state; only
                # geometry is relinearized about this conditional full curve.
                capsule=CapsuleCorridorModel(ref,center[:,:2],g[:,:2].copy())
                capsules[item['path']]=capsule
                modes[item['path']]=(pred['initiation_tick'],pred['braking_steps'])
                if item['path'] in base:
                    nominal,valid,gradient,density,_,dg=signal_jacobians(ref,center,g)
                    nominal-=np.einsum('tcd,d->tc',gradient,anchor)
                    density-=np.einsum('tcd,d->tc',dg,anchor)
                    u,v=np.diff(ref,axis=0);turn=float(np.degrees(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)))
                    constant=np.sum(density**2,axis=0);linear=np.einsum('tc,tcd->cd',2*density,dg)
                    quadratic=np.array([dg[:,c].T@dg[:,c] for c in range(2)])
                    waves[item['path']]=(nominal,valid,gradient,turn,constant,linear,quadratic)
            cache={};best={}
            def calculate(y):
                budget()
                if 'y' in cache and np.array_equal(y,cache['y']):return cache
                x=unwhite(y);mx=metric@x;norm2=float(x@mx)
                values=[];grads=[];labels=[];shape={}
                def add(value,g,scale,label):values.append(value/scale);grads.append(g/scale);labels.append(label)
                for path,(nominal,valid,g,turn,constant,linear,q) in waves.items():
                    v=waveform_facets(nominal,g,valid,turn,x);qx=q@x;e=constant+linear@x+qx@x
                    for c,key in enumerate(ENERGY):v[key]=(float(e[c]),linear[c]+2*qx[c],0.)
                    for key,(value,row,offset) in list(v.items()):
                        rate=rates.get((path,key),0.)
                        v[key]=(value+rate*norm2,row+2*rate*mx,offset)
                    shape[path]=v
                    for key,(value,row,_) in v.items():
                        add(value-(1.-args.interior_reserve)*base[path][key],row,
                            max(base[path][key],1e-3),f'{path}_{key}')
                for key,total in totals.items():
                    if total<=1e-8:continue
                    add(sum(v[key][0] for v in shape.values())-.99*total,
                        sum((v[key][1] for v in shape.values()),np.zeros(768)),total,'total_'+key)
                for path,capsule in capsules.items():
                    distance,row,_=capsule.evaluate(x-anchor)
                    add(distance-(1.25-margin[path]),row,max(margin[path],1e-3),f'{path}_full_corridor')
                value,row=geometry.support_constraint(x);add(value,row,1.,'saved_action_support')
                if pure_feed_features is not None:
                    change=pure_feed_features@x[256:512];j=int(np.argmin(change));row=np.zeros(768)
                    row[256:512]=-pure_feed_features[j]
                    add(float(-change[j]),row,1.,'saved_pure_feed_preservation')
                for path,witness in completion.items():
                    for value,row,scale,label in witness.evaluate(x):
                        add(value,row,scale,f'{path}_completion_{label}')
                if state_cut:
                    delta_state=state_cut['response']@x;length=np.linalg.norm(delta_state)
                    row=delta_state@state_cut['response']/max(length,1e-30)
                    add(float(length-state_cut['radius']),row,state_cut['radius'],'observed_state_validity')
                loss,loss_g=geometry.objective(x)
                if args.objective!='minimum_cruise_deficit':
                    objective_metrics=SHAPE+(ENERGY if args.objective=='minimum_shape_energy' else ())
                    entries=[(value/base[path][key],row/base[path][key])
                             for path,v in shape.items() for key,(value,row,_) in v.items()
                             if key in objective_metrics and base[path][key]>1e-8]
                    obj=np.array([v for v,_ in entries]);obj_g=white_grad([g for _,g in entries])
                else:
                    obj=np.array([loss/baseline_loss]);obj_g=white_grad(loss_g)[None,:]/baseline_loss
                cache.clear();cache.update(y=y.copy(),x=x,values=np.array(values),gradients=white_grad(grads),
                    labels=labels,loss=loss,objective_values=obj,objective_gradients=obj_g)
                return cache
            def oracle(y):c=calculate(y);return c['values'],c['gradients']
            def objective(y):c=calculate(y);return c['objective_values'],c['objective_gradients']
            def callback(y):
                c=calculate(y)
                if max(c['values'])<=1e-7 and np.linalg.norm(y)<=radius+1e-10:
                    value=(float(np.sum((y-chol.T@audited_direction)**2)) if args.restore_feasibility
                           else float(max(c['objective_values'])))
                    if not best or value<best['objective']:best.update(y=y.copy(),objective=value)
            if args.restore_feasibility:
                target=chol.T@audited_direction
                y,diagnostic=solve_convex_cuts(np.eye(768),-target,target,oracle,
                    radius,min(deadline,time.perf_counter()+60),callback,max_rounds=12)
                diagnostic['restoration_metric_displacement']=float(np.linalg.norm(y-target))
            elif audited_direction is not None:
                y=chol.T@audited_direction
                diagnostic=dict(method='saved_direction_constraint_audit',optimizer_iterations=0)
            else:
                y,diagnostic=solve_convex_cuts(np.eye(768)*1e-7,np.zeros(768),chol.T@anchor,oracle,
                    radius,min(deadline,time.perf_counter()+160),callback,max_rounds=30,objective_facets=objective)
            if best:y=best['y']
            c=calculate(y);x=c['x'].copy();conditional_ok=max(c['values'])<=1e-7
            check=exact_checks(x)
            changed=[r['path'] for r in check['records'] if r['supported'] and
                modes[r['path']]!=(r['initiation_tick'],r['braking_steps'])]
            record=dict(outer_round=outer+1,conditional_model_feasible=bool(conditional_ok),
                maximum_scaled_violation=float(max(c['values'])),metric_norm=float(np.linalg.norm(y)),
                optimization_objective=args.objective,selected_objective=float(max(c['objective_values'])),
                selected_request_deficit=c['loss'],anchor_reconstruction_error=identity_max,
                largest_named_constraint_residuals=[dict(label=c['labels'][j],scaled_value=float(c['values'][j]))
                    for j in np.argsort(c['values'])[-12:][::-1]],
                changed_terminal_modes=changed,exact_branch_checks=check,conic_diagnostics=diagnostic)
            history.append(record);last=x;np.savez_compressed(stage/f'round{outer+1}_direction.npz',direction=x.reshape(3,256))
            write_json(stage/'results.json',dict(parent_sha256=sha(parent),parent_model=data['parent_model'],
                history=history,baseline_request_deficit=baseline_loss,goal_complete=False))
            print('COMPLETE_BRANCH_ROUND',outer+1,conditional_ok,check['eligible_for_actual_probe'],
                'mode_changes',changed,'request_loss',c['loss'],'time',check['predicted_complete_time_s'],flush=True)
            if conditional_ok and check['eligible_for_actual_probe']:
                accepted=True;break
            if audited_direction is not None:break
            if not conditional_ok or not all(r['supported'] for r in check['records']):break
            anchor=x
        assert last is not None
        np.savez_compressed(stage/'direction.npz',direction=last.reshape(3,256))
        write_json(stage/'results.json',dict(parent_sha256=sha(parent),parent_model=data['parent_model'],
            history=history,baseline_request_deficit=baseline_loss,metric_norm=float(np.sqrt(last@metric@last)),
            optimization_objective=args.objective,
            eligible_for_actual_probe=accepted,model_feasibility_is_not_physical_feasibility=True,
            prospective_error_bound=False,goal_complete=False))
        receipt(stage,dict(status='completed_complete_branch_joint_proposal',eligible_for_actual_probe=accepted,
            costs=costs,wall_s=time.perf_counter()-start,learned_candidate_adopted=False,goal_complete=False))
    except Exception as error:
        receipt(stage,dict(status='failed_complete_branch_joint_proposal',error=str(error),costs=costs,
            wall_s=time.perf_counter()-start,learned_candidate_adopted=False,goal_complete=False))
        raise


if __name__=='__main__':main()

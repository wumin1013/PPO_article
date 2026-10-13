"""Current continuous-sensor credit, shared-parameter projection, no rollout."""
import argparse
import time
import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.optimize import linprog
from continuous_sensor_runtime import environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from constrained_parameter_solver import minimum_metric_step
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from audit_xy_curvature_resolution import CURVATURE_NAMES,curvature_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


NAMES=['in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
       *CURVATURE_NAMES,'negative_signed_heading_deg','heading_overshoot_deg',
       'segment_max_distance_mm','complete_time_s']
IMPROVEMENT_NAMES=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                   'negative_signed_heading_deg','heading_overshoot_deg')


def vector(record,curvature):
    shape=record['shape'] or dict.fromkeys(NAMES[:3],0.)
    return np.array([shape[k] for k in NAMES[:3]]+[curvature[k] for k in CURVATURE_NAMES]+
        ([max(0.,-shape['heading_min_deg']),shape['heading_overshoot_deg']] if record['shape'] is not None else [0.,0.])+
        [record['segment_distance']['max_distance_upper_mm'],record['full_time_s']])


def constraint_problem(gradient,baselines,ids,support,exclusions,upper_margin=None,omit_names=()):
    """Common local constraints; optional margins tighten shape predictions."""
    dimension=gradient.shape[-1]
    base={r['path']:vector(r,r['curvature']) for r in baselines}
    corner=[i for i,r in enumerate(baselines) if r['shape'] is not None]
    margin=np.zeros(gradient.shape[:2]) if upper_margin is None else np.asarray(upper_margin)
    a=[];b=[];labels=[]
    def add(row,bound,label,eta=0.):
        a.append(np.r_[row,eta]);b.append(float(bound));labels.append(label)
    for i in corner:
        for m in range(len(NAMES)-2):
            if NAMES[m] in omit_names:continue
            add(gradient[i,m],-margin[i,m],f'{ids[i]}_{NAMES[m]}')
            add(-gradient[i,m],base[ids[i]][m],f'{ids[i]}_{NAMES[m]}_nonnegative')
    for name in IMPROVEMENT_NAMES:
        if name in omit_names:continue
        m=NAMES.index(name)
        add(gradient[corner,m].sum(0),0.,f'total_{name}',sum(base[ids[i]][m] for i in corner))
    add(gradient[:,-1].sum(0),0.,'total_time')
    for i,r in enumerate(baselines):
        add(gradient[i,-2],.5*r['segment_distance']['margin_lower_mm'],f'margin_{ids[i]}')
    for i,row in enumerate(exclusions):add(row,0.,f'failed_{i}')
    for i,row in enumerate(support):
        add(row,1.,f'upper_{i}');add(-row,1.,f'lower_{i}')
    return np.asarray(a).reshape(-1,dimension+1),np.asarray(b),labels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='continuous_sensor_complete_credit_20260917')
    parser.add_argument('--parent',default='continuous_sensor_matched_function_transfer_20260917')
    parser.add_argument('--arm',default='updated')
    parser.add_argument('--stage',default='continuous_sensor_current_parameter_solve_20260917')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/args.source
    assert (source/'receipt.json').exists(),'Complete bounded credit collection before using it'
    parent=BASE/args.parent
    selected=next(r for r in read(parent/'results.json')['candidates']
        if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
    model=parent/selected['model'];raw=read(source/'results.json')
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    policy_class=KernelLocalFeedbackPolicy;current_factory=environment_factory
    if metadata['architecture'] in ('canonical_gate_kernel','gate_band_canonical_kernel'):
        from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
        policy_class=CanonicalGatePolicy;current_factory=checkpoint_environment_factory(metadata['architecture'])
    stage=register(args.stage,
        [__file__,source/'results.json',source/'queries.npz',source/'receipt.json',
         parent/'results.json',parent/'receipt.json',model,BASE/'continuous_sensor_runtime.py',
         BASE/'constrained_parameter_solver.py',BASE/'audit_xy_curvature_resolution.py',
         BASE/'probe_coordinated_suffix.py',BASE/'kernel_local_feedback_policy.py',
         BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py',BASE/'gate_band_state.py'],
        dict(new_physics_ticks=0,saved_geometry_ticks=65000,wall_s=240,
            learned_parameters=768,maximum_predicted_common_reduction=.2,
            normalized_saved_action_bound=1.,maximum_proposal_metric_norm=1.,
            corridor_margin_consumption_fraction=.5,nonnegative_predicted_shape_metrics=True,
            directed_heading_bounds=True,metric_names=NAMES,
            actual_credit_paths=[r['path'] for r in raw['baselines']],checkpoint_architecture=metadata['architecture']),
        'Use the explicit source/checkpoint/runtime and actual credit-path list above. '
        'Canonical-gate checkpoints include the corrected sensor, guide and gate-state '
        'representation; all their current physical credit is collected afresh. '
        'For every nonnegative shape metric also require predicted baseline+change>=0. '
        'A central finite slope at a zero excursion boundary must not create fictitious '
        'negative excursions that count as aggregate improvement. This is a correction '
        'to the local prediction domain, not a new reward or relaxed physical limit. '
        'Saved real velocity headings also show negative incoming-relative heading '
        'despite zero outward XY excursion. Reuse existing heading_min/overshoot '
        'data to constrain the two directed tangent bounds as well as full physical '
        'reverse turning. Include their nonnegative violation totals in the common '
        'improvement target. These are explicit user shape requirements, not extra '
        'reward weights or new physical probes. '
        'Reconstruct all29 contemporary screen trajectories for the action trust '
        'metric, rather than transferring old parent state support. Sum every '
        'single-decision consequence into shared-parameter derivatives, retain '
        'observed failed-axis exclusions, and compute raw1ms XY plus physical '
        'curvature metrics from saved complete suffixes. Registered isolated corners '
        'have shape gradients; registered mixed routes have physical-margin/time '
        'gradients. Remaining screen routes have saved-action bounds only, so all29 must '
        'subsequently receive small whole-policy checks. Require individual '
        'entry/exit/reverse, both directed headings and four curvature quantities to be nonworse in the '
        'local model, common total shape gain, no predicted time increase, half '
        'the measured corridor margin at most, and no exact-action freezing. '
        'No borrowed old credit/error margin or claim of safety. A proposal '
        'advances only if predicted gain>=10% and norm<=1, then positive/negative '
        'small full-policy tests must measure composition error before larger '
        'learning. No new physics, candidate promotion, or sealed access here.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0)
    p=policy_class(256);p.load_state_dict(metadata['policy'])
    xs=[];ws=[];support_ids=[]
    for r,trace in zip(selected['results'],selected['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,state_snapshots=states,env_factory=current_factory)
        states.append(env.state());env=current_factory(ref)
        for j,s in enumerate(states):
            env.restore(s);x=env.observation()
            assert np.array_equal(env.decode_action(p.act(x,True)[0]),actions[j])
            xs.append(x);ws.append(1/(len(selected['results'])*len(states)));support_ids.append(r['original_path'])
    xs=np.asarray(xs);ws=np.asarray(ws)
    features=feature_batches(p,xs)*p.residual_scale.detach().numpy()[None,:,None]/AMPLITUDE[None,:,None]
    metric=block_diag(*[features[:,h].T@(ws[:,None]*features[:,h])/3+.001*np.eye(256) for h in range(3)])
    support=np.zeros((len(xs),3,768))
    for h in range(3):support[:,h,h*256:(h+1)*256]=features[:,h]
    support=support.reshape(-1,768)
    with np.load(source/'queries.npz') as z:qx=z['observation'].copy()
    qfeatures=feature_batches(p,qx)*p.residual_scale.detach().numpy()[None,:,None]
    jac=np.zeros((len(qx),3,768))
    for h in range(3):jac[:,h,h*256:(h+1)*256]=qfeatures[:,h]
    ids=[r['path'] for r in raw['baselines']];nominal={};base={};baselines=[]
    for r in raw['baselines']:
        with np.load(BASE/r['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        c=curvature_metrics(ref,rows) if len(ref)==3 else dict.fromkeys(CURVATURE_NAMES,0.)
        r=dict(r,full_time_s=len(rows)*.001,curvature=c)
        nominal[r['path']]=(ref,rows);base[r['path']]=vector(r,c);baselines.append(r)
    lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    gradient=np.zeros((len(ids),len(NAMES),768));exclusions=[];modes={};single=[]
    for q in raw['queries']:
        qid=q['query'];path=q['path'];ref,rows=nominal[path]
        for head in range(3):
            pair=[lookup[qid,head,sign] for sign in (-1,1)];values=[]
            for row in pair:
                if not row['success']:
                    exclusions.append(row['direction']*jac[qid,head]/AMPLITUDE[head]);values.append(None);continue
                if len(ref)==3:
                    with np.load(BASE/row['trace']) as z:full=np.vstack([rows[:q['anchor']*20],z['suffix_rows']])
                    c=curvature_metrics(ref,full)
                else:c=dict.fromkeys(CURVATURE_NAMES,0.)
                values.append(vector(row,c))
            good=[i for i,v in enumerate(values) if v is not None]
            if len(good)==2:
                slope=(values[1]-values[0])/(pair[1]['latent_offset']-pair[0]['latent_offset']);mode='central'
            elif len(good)==1:
                j=good[0];slope=(values[j]-base[path])/pair[j]['latent_offset'];mode='one_sided'
            else:slope=np.zeros(len(NAMES));mode='neither_complete'
            modes[mode]=modes.get(mode,0)+1;single.append(slope)
            gradient[ids.index(path)]+=slope[:,None]*jac[qid,head]
    exclusions=np.asarray(exclusions).reshape(-1,768)
    np.savez_compressed(stage/'gradients.npz',gradient=gradient,path_id=ids,metric_names=np.asarray(NAMES),
        query_jacobian=jac,single_action_slopes=np.asarray(single),failed_axis_constraints=exclusions)
    np.savez_compressed(stage/'support.npz',observation=xs,weight=ws,path_id=support_ids,metric=metric)
    write_json(stage/'baselines.json',dict(baselines=baselines,modes=modes))
    a,b,labels=constraint_problem(gradient,baselines,ids,support,exclusions)
    scale=np.maximum(np.linalg.norm(a,axis=1),1e-12)
    lp=linprog(np.r_[np.zeros(768),-1.],A_ub=a/scale[:,None],b_ub=b/scale,
        bounds=[(None,None)]*768+[(0.,.2)],method='highs',options=dict(time_limit=90.,
            primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    result=dict(lp_success=bool(lp.success),message=lp.message,parent_sha256=sha(model),
        model=str(model.relative_to(BASE)),metric_names=NAMES,path_ids=ids,modes=modes,
        failed_axes=len(exclusions),eligible_for_small_probe=False,goal_complete=False)
    if lp.success:
        binding=[dict(constraint=labels[i],scaled_slack=float(lp.ineqlin.residual[i]),
            unscaled_dual=float(lp.ineqlin.marginals[i]/scale[i]))
            for i in range(len(labels)) if abs(lp.ineqlin.marginals[i])>1e-10]
        write_json(stage/'binding_constraints.json',dict(nonzero_dual_constraints=binding,
            interpretation='Duals describe this local finite-credit LP only, not physical impossibility.'))
        eta=float(lp.x[-1]);direction,projection=minimum_metric_step(metric,a[:,:768],b-a[:,-1]*eta,lp.x[:768],deadline=started+235.)
        pred=np.einsum('pmd,d->pm',gradient,direction)
        result.update(predicted_common_reduction=eta,projection=projection,
            maximum_normalized_action_change=float(abs(support@direction).max()),
            eligible_for_small_probe=bool(eta>=.1 and projection['metric_norm']<=1.))
        np.savez_compressed(stage/'direction.npz',direction=direction.reshape(3,256),
            predicted_changes=pred,path_id=ids,metric_names=np.asarray(NAMES))
    assert costs['saved_prefix_geometry_ticks']<=65000 and time.perf_counter()-started<240
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_continuous_sensor_current_parameter_solve',costs=costs,
        wall_s=time.perf_counter()-started,eligible_for_small_probe=result['eligible_for_small_probe'],goal_complete=False))
    print('CONTINUOUS_SENSOR_SOLVE',result,flush=True)


if __name__=='__main__':main()

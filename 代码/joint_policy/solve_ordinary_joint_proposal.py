"""Learn a shared proposal on ordinary motion; never differentiate a brake branch."""
import argparse
import time
import numpy as np
import torch
from scipy.linalg import solve_triangular
from transition_waveform_model import chain_responses, signal_jacobians
from waveform_constraint_model import waveform_facets
from capsule_corridor_model import CapsuleCorridorModel
from cruise_update_geometry import CruiseUpdateGeometry
from convex_cut_solver import solve_convex_cuts
from canonical_gate_state import checkpoint_environment_factory
from event_adaptive_terminal_prediction import predict_terminal
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha

SHAPE = ('in_edge_outward_mm', 'out_edge_outward_mm', 'full_physical_reverse_deg',
         'negative_signed_heading_deg', 'heading_overshoot_deg')
ENERGY = ('ordinary_xy_energy', 'ordinary_physical_energy')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_ordinary_joint_proposal_20260918')
    parser.add_argument('--ordinary-error-model')
    args=parser.parse_args()
    source = BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    reference = BASE/'canonical_gate_band_shape_priority_current_reference_20260918'
    geom_source = BASE/'canonical_gate_band_current_hold_geometry_20260918'
    validation = BASE/'canonical_gate_band_shape_priority_current_terminal_prediction_20260918'
    assert read(source/'receipt.json')['status'] == 'completed_current_complete_transition_credit'
    data, ref_data = read(source/'results.json'), read(reference/'results.json')
    parent = BASE/data['parent_model']
    assert sha(parent) == ref_data['parent_sha256'] == read(validation/'results.json')['parent_sha256']
    geometry = CruiseUpdateGeometry(geom_source, 'physical', sha(parent))
    rates={};error_sources=[]
    if args.ordinary_error_model:
        error_source=BASE/args.ordinary_error_model
        error_data=read(error_source)
        assert error_data['parent_sha256']==sha(parent) and not error_data['guaranteed_bound']
        key_map=dict(full_xy_normalized_energy='ordinary_xy_energy',
                     full_physical_normalized_energy='ordinary_physical_energy')
        rates={(r['path'],key_map.get(r['metric'],r['metric'])):r['rate'] for r in error_data['coefficients']}
        error_sources=[error_source,error_source.parent/'receipt.json']
    radius = .1
    stage = register(args.stage,
        [__file__, parent, source/'results.json', source/'receipt.json', reference/'results.json',
         reference/'receipt.json', geom_source/'geometry.npz', geom_source/'results.json',
         geom_source/'receipt.json', validation/'results.json', validation/'receipt.json',
         BASE/'transition_waveform_model.py', BASE/'waveform_constraint_model.py',
         BASE/'capsule_corridor_model.py', BASE/'convex_cut_solver.py',
         BASE/'event_adaptive_terminal_prediction.py']+error_sources,
        dict(new_actor_decisions=0, new_physics_ticks=0, wall_s=600, maximum_exchange_rounds=30,
             physical_metric_radius=radius, normalized_saved_action_bound=.5,
             ordinary_shape_reduction=.01, exact_terminal_checks_at_scales=[.1,.25,.5,1.],
             ordinary_error_model=args.ordinary_error_model),
        ('Single-factor follow-up: add the measured ordinary scalar optimism rate '
         'times squared physical update norm. Same current399a, radius, objective, '
         'action support and ordinary-only shape scope. Calibration is not a '
         'prospective guarantee. The prior four complete physical outcomes are '
         'reused, not repeated. ' if rates else '')+
        'Current399a ordinary shared-parameter sensitivities only. Minimize the '
        'existing saved-state99.5mm/s feed-request deficit jointly over768 steering, '
        'feed and preview coefficients. Retain every ordinary corner shape/curvature '
        'and energy ceiling, one-percent aggregate positive shape improvement, '
        'half corridor margin, and unchanged physical saved-action support. Exclude '
        'ALL terminal rows from both objective derivatives and constraints: no '
        'invalid frozen-duration tail derivative. The radius0.1 and terminal '
        'treatment BOTH differ from the last retained0.0069 shape solve; this is '
        'a joint mechanism proposal, not single-factor causal attribution. '
        'Evaluate exact original terminal planning on predicted ordinary motion '
        'at four fixed scales. All predictions are proposal diagnostics only; '
        'observed ordinary errors and duration uncertainty prevent safety '
        'certification. Require subsequent complete physical feedback tests; '
        'failed prefixes and saved-state request fit never count as gains. '
        'No runtime, limit, policy input, trajectory template or test changes.')
    started = time.perf_counter()
    deadline = started+580
    metric = geometry.metric
    chol = np.linalg.cholesky(metric)
    waves, capsules, base, margins = {}, {}, {}, {}
    for item in data['records']:
        with np.load(source/item['file']) as z:
            local = {k:z[k].copy() for k in z.files}
        ref, rows = local['reference'], local['rows']
        stop = int(np.flatnonzero(rows[:,12])[0])
        # Jacobian construction can contain terminal rows; none enter this solve.
        response = chain_responses(local)[:stop]
        ordinary = rows[:stop]
        capsule = CapsuleCorridorModel(ref, ordinary[:,:2], response[:,:2].copy())
        nominal_distance = capsule.evaluate(np.zeros(768))[0]
        capsules[item['path']] = capsule
        margins[item['path']] = .5*(1.25-nominal_distance)
        assert margins[item['path']] > 0
        if len(ref) == 3:
            nominal, valid, gradient, density, _, dg = signal_jacobians(ref, ordinary, response)
            u,v = np.diff(ref,axis=0)
            turn = float(np.degrees(np.arctan2(u[0]*v[1]-u[1]*v[0], u@v)))
            constant = np.sum(density**2, axis=0)
            linear = np.einsum('tc,tcd->cd', 2*density, dg)
            quadratic = np.array([dg[:,c].T@dg[:,c] for c in range(2)])
            waves[item['path']] = (nominal,valid,gradient,turn,constant,linear,quadratic)
            b = waveform_facets(nominal, gradient, valid, turn, np.zeros(768))
            base[item['path']] = {k:v[0] for k,v in b.items()}
            base[item['path']].update(zip(ENERGY,constant.tolist()))
        assert time.perf_counter() < deadline
    totals = {key:sum(v[key] for v in base.values()) for key in SHAPE}
    baseline_loss = geometry.objective(np.zeros(768))[0]
    cache = {}
    best = {}
    def unwhite(y):
        return solve_triangular(chol.T,y,lower=False,check_finite=False)
    def white_gradient(g):
        return solve_triangular(chol,np.asarray(g).T,lower=True,check_finite=False).T
    def calculate(y):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Ordinary proposal deadline')
        if 'y' in cache and np.array_equal(y,cache['y']):
            return cache
        x = unwhite(y)
        mx=metric@x;norm_squared=float(x@mx)
        all_values = {}
        values,gradients,labels = [],[],[]
        def add(value,gradient,scale,label):
            values.append(value/scale); gradients.append(gradient/scale); labels.append(label)
        for path,(nominal,valid,gradient,turn,constant,linear,quadratic) in waves.items():
            v = waveform_facets(nominal,gradient,valid,turn,x)
            qx = quadratic@x
            energy = constant+linear@x+qx@x
            for c,key in enumerate(ENERGY):
                v[key] = (float(energy[c]),linear[c]+2*qx[c],0.)
            for key,(value,g,offset) in list(v.items()):
                rate=rates.get((path,key),0.)
                if rate:v[key]=(value+rate*norm_squared,g+2*rate*mx,offset)
            all_values[path] = v
            for key,(value,g,_) in v.items():
                bound = base[path][key]
                add(value-bound,g,max(bound,1e-3),f'{path}_{key}')
        for key,total in totals.items():
            if total <= 1e-8:continue
            value = sum(v[key][0] for v in all_values.values())
            g = sum((v[key][1] for v in all_values.values()),np.zeros(768))
            add(value-.99*total,g,total,'total_'+key)
        for path,capsule in capsules.items():
            distance,g,_ = capsule.evaluate(x)
            add(distance-(1.25-margins[path]),g,max(margins[path],1e-3),f'{path}_ordinary_corridor')
        support,g = geometry.support_constraint(x)
        add(support,g,1.,'saved_action_support')
        loss,loss_gradient = geometry.objective(x)
        cache.clear()
        cache.update(y=y.copy(),x=x,values=np.array(values),gradients=white_gradient(gradients),
                     labels=labels,loss=loss,loss_gradient=white_gradient(loss_gradient),
                     shape_totals={k:sum(v[k][0] for v in all_values.values()) for k in SHAPE})
        return cache
    def oracle(y):
        c = calculate(y)
        return c['values'],c['gradients']
    def objective(y):
        c = calculate(y)
        return np.array([c['loss']/baseline_loss]),c['loss_gradient'][None,:]/baseline_loss
    def callback(y):
        c = calculate(y)
        if max(c['values']) <= 1e-7 and np.linalg.norm(y) <= radius+1e-10:
            if not best or c['loss'] < best['loss']:
                best.update(y=y.copy(),loss=c['loss'])
                np.savez_compressed(stage/'best_direction.npz',direction=c['x'].reshape(3,256))
    try:
        y,diagnostics = solve_convex_cuts(np.eye(768)*1e-7,np.zeros(768),np.zeros(768),
            oracle,radius,deadline,callback,max_rounds=30,objective_facets=objective)
        if best:y=best['y']
        c = calculate(y)
        eligible = bool(max(c['values']) <= 1e-7 and np.linalg.norm(y) <= radius+1e-10)
        direction = c['x'].copy()
        np.savez_compressed(stage/'direction.npz',direction=direction.reshape(3,256))
        result = dict(parent_sha256=sha(parent),parent_model=data['parent_model'],
            eligible_for_actual_probe=eligible,model_feasibility_is_not_physical_feasibility=True,
            metric_norm=float(np.linalg.norm(y)),maximum_scaled_violation=float(max(c['values'])),
            ordinary_error_model=args.ordinary_error_model,
            baseline_request_deficit=baseline_loss,proposed_request_deficit=c['loss'],
            baseline_ordinary_shape_totals=totals,proposed_ordinary_shape_totals=c['shape_totals'],
            most_violated=[dict(label=c['labels'][i],value=float(c['values'][i]))
                           for i in np.argsort(c['values'])[-8:]],diagnostics=diagnostics,goal_complete=False)
        write_json(stage/'results.json',result)
        if eligible:
            md=torch.load(parent,map_location='cpu',weights_only=False)
            factory=checkpoint_environment_factory(md['architecture'])
            costs=dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=0)
            def check():
                assert time.perf_counter()-started < 595 and costs['terminal_planner_queries'] < 3500
            predictions=[]
            for item in data['records']:
                with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
                rows=local['rows'];stop=int(np.flatnonzero(rows[:,12])[0])
                response=chain_responses(local,direction[None,:])[:,:,0]
                for scale in (.1,.25,.5,1.):
                    pred=predict_terminal(local['reference'],rows[:,:6]+scale*response,stop,factory,costs,check)
                    r=dict(path=item['path'],scale=scale,supported=pred['supported'])
                    if pred['supported']:
                        r.update(initiation_tick=pred['initiation_tick'],braking_steps=pred['braking_steps'],
                            complete_time_s=pred['total_ticks']*.001,prefix_inside=pred['predicted_prefix_inside'],
                            original_complete_time_s=len(rows)*.001)
                    predictions.append(r)
            result.update(terminal_predictions=predictions,prediction_costs=costs,
                prospective_prediction_validated=False,
                predicted_complete_time_by_scale={str(s):sum(r.get('complete_time_s',0.) for r in predictions
                    if r['scale']==s) if all(r['supported'] for r in predictions if r['scale']==s) else None
                    for s in (.1,.25,.5,1.)})
            write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_ordinary_joint_proposal',eligible_for_actual_probe=eligible,
            wall_s=time.perf_counter()-started,new_actor_decisions=0,new_physics_ticks=0,
            learned_candidate_adopted=False,goal_complete=False))
        print('ORDINARY_JOINT_PROPOSAL',eligible,float(np.linalg.norm(y)),baseline_loss,c['loss'],
              result.get('predicted_complete_time_by_scale'),flush=True)
    except Exception as error:
        receipt(stage,dict(status='failed_ordinary_joint_proposal',error=str(error),
            wall_s=time.perf_counter()-started,new_actor_decisions=0,new_physics_ticks=0,
            learned_candidate_adopted=False,goal_complete=False))
        raise


if __name__ == '__main__':main()

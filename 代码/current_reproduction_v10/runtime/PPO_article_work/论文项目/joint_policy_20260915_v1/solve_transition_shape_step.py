"""One bounded shared-parameter update from closed-loop transition credit."""
import argparse
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from waveform_constraint_model import waveform_facets, SIGNED_SHAPE_CHANNELS, corrected_signed_facet
from capsule_corridor_model import CapsuleCorridorModel
from cruise_update_geometry import CruiseUpdateGeometry,capsule_endpoint_facets
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha
from joint_regret_objective import normalized_regrets


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--restore',action='store_true')
    parser.add_argument('--measured-margin',action='store_true')
    parser.add_argument('--waveforms',default='canonical_gate_band_transition_waveform_model_20260917')
    parser.add_argument('--reference',default='canonical_gate_band_coherent_parameter_credit_20260917')
    parser.add_argument('--coverage',default='canonical_gate_band_transition_response_coverage_20260917')
    parser.add_argument('--events',default='canonical_gate_band_terminal_trigger_boundary_model_20260917')
    parser.add_argument('--stage')
    parser.add_argument('--restore-from')
    parser.add_argument('--energy-reduction',type=float,default=0.)
    parser.add_argument('--metric-radius',type=float,default=.1)
    parser.add_argument('--required-shape-reduction',type=float,default=.15)
    parser.add_argument('--energy-scope',choices=('per_corner','aggregate'),default='per_corner')
    parser.add_argument('--corridor-model',choices=('displacement','segment_capsules'),default='displacement')
    parser.add_argument('--objective',choices=('minimum_norm','minimum_energy','minimum_cruise_deficit','minimum_shape','minimum_joint_regret'),default='minimum_norm')
    parser.add_argument('--shape-ideal')
    parser.add_argument('--shape-scope',choices=('aggregate','per_corner'),default='aggregate')
    parser.add_argument('--energy-ideal')
    parser.add_argument('--optimizer',choices=('slsqp','convex_cuts'),default='slsqp')
    parser.add_argument('--distance-error-audit')
    parser.add_argument('--parameter-geometry')
    parser.add_argument('--geometry-arm',choices=('latent','physical'))
    parser.add_argument('--interpolate-last-from')
    parser.add_argument('--expose-capsule-endpoints',action='store_true')
    parser.add_argument('--preserve-cruise-deficit',action='store_true')
    parser.add_argument('--preserve-cruise-states',action='store_true')
    parser.add_argument('--cruise-reference-model')
    parser.add_argument('--start-from')
    parser.add_argument('--empirical-error-model')
    parser.add_argument('--capsule-audit',default='canonical_gate_band_refined_capsule_facet_audit_20260917')
    args=parser.parse_args()
    if args.optimizer=='convex_cuts':
        assert args.objective in ('minimum_norm','minimum_energy','minimum_shape','minimum_joint_regret','minimum_cruise_deficit') and not args.restore_from
    if args.objective in ('minimum_shape','minimum_joint_regret'):assert args.optimizer=='convex_cuts' and args.start_from
    assert bool(args.shape_ideal)==bool(args.energy_ideal)==(args.objective=='minimum_joint_regret')
    if args.shape_scope=='per_corner':assert args.objective=='minimum_shape'
    assert 0.<=args.energy_reduction<=.2
    assert 0.<args.metric_radius<=.1 and 0.<=args.required_shape_reduction<=.15
    if args.measured_margin:
        assert args.reference=='canonical_gate_band_coherent_parameter_credit_20260917','Bind fresh measured errors before recentering their allowance'
    args.restore=args.restore or args.measured_margin or bool(args.restore_from)
    if args.expose_capsule_endpoints:assert args.restore_from and args.parameter_geometry and args.corridor_model=='segment_capsules'
    source=BASE/args.waveforms
    probes=BASE/args.reference
    coverage=BASE/args.coverage
    event_source=BASE/args.events
    assert read(source/'receipt.json')['status']=='compiled_complete_transition_waveform_model'
    event_coverage=read(event_source/'receipt.json')['conditional_measured_coverage_passed']
    data=read(source/'results.json');physical=read(probes/'results.json')
    model=BASE/data['parent_model'];assert sha(model)==data['parent_sha256']==physical['parent_sha256']
    assert read(event_source/'results.json')['parent_sha256']==sha(model)
    distance_sources=[]
    if args.distance_error_audit:
        assert args.corridor_model=='segment_capsules'
        distance_source=BASE/args.distance_error_audit
        audit=read(distance_source/'results.json')
        assert read(distance_source/'receipt.json')['conditional_distance_coverage_passed']
        assert audit['conditional_distance_coverage_passed'] and audit['requires_same_capsule_half_margin_constraint']
        assert audit['parent_sha256']==sha(model)
        bindings=read(distance_source/'registration.json')['source_sha256']
        for p in (source/'results.json',probes/'results.json',event_source/'results.json',BASE/'capsule_corridor_model.py'):
            assert bindings[p.relative_to(BASE.parent.parent).as_posix()]==sha(p)
        distance_sources=[distance_source/'results.json',distance_source/'receipt.json',distance_source/'registration.json']
    else:
        assert event_coverage,'Failed vector audit requires a separately verified matching capsule-distance audit'
    geometry=None
    assert bool(args.parameter_geometry)==bool(args.geometry_arm)
    if args.parameter_geometry:
        geometry_source=BASE/args.parameter_geometry
        geometry=CruiseUpdateGeometry(geometry_source,args.geometry_arm,sha(model))
        distance_sources += [BASE/'cruise_update_geometry.py',geometry_source/'geometry.npz',geometry_source/'results.json',geometry_source/'receipt.json']
    if args.preserve_cruise_deficit:
        assert geometry and geometry.positive_cruise_latent_domain
    if args.preserve_cruise_states:assert geometry
    cruise_floor=np.zeros(len(geometry.preservation_features)) if geometry else None
    if args.cruise_reference_model:
        assert args.preserve_cruise_states
        import torch
        cruise_reference=BASE/args.cruise_reference_model
        current=torch.load(model,map_location='cpu',weights_only=False)
        retained=torch.load(cruise_reference,map_location='cpu',weights_only=False)
        assert current['architecture']==retained['architecture']
        for key,value in current['policy'].items():
            if key!='residual_coefficients':assert torch.equal(value,retained['policy'][key]),key
        feed_delta=(retained['policy']['residual_coefficients'][1].double()-current['policy']['residual_coefficients'][1].double()).numpy()
        cruise_floor=np.maximum(0.,geometry.preservation_features@feed_delta)
        distance_sources += [cruise_reference]
    cruise_ceiling=geometry.objective(np.zeros(768))[0] if args.preserve_cruise_deficit else None
    names=data['metric_names'];ids=[r['path'] for r in data['records']]
    radius=args.metric_radius;fraction=args.required_shape_reduction;action_bound=.5
    previous=BASE/('canonical_gate_band_transition_credit_feasibility_restore_20260917' if args.measured_margin
                   else 'canonical_gate_band_transition_credit_step_solve_20260917')
    if args.restore_from:previous=BASE/args.restore_from
    if args.restore:
        old=read(previous/'optimizer.json')
        assert old.get('parameter_geometry')==(geometry.identity if geometry else None),'Restoration must retain update geometry'
        assert old.get('preserve_cruise_deficit',False)==args.preserve_cruise_deficit
        assert old.get('preserve_cruise_states',False)==args.preserve_cruise_states
        assert old.get('cruise_reference_model')==args.cruise_reference_model
        assert old.get('corridor_model','displacement')==args.corridor_model,'Restoration must retain the same corridor model'
        assert old.get('energy_reduction_scope','per_corner')==args.energy_scope,'Restoration must retain the same energy scope'
        assert old.get('required_per_corner_existing_energy_reduction',0.)+old.get('required_aggregate_existing_energy_reduction',0.)==args.energy_reduction,'Restoration must retain the same energy target'
    warm_file=previous/('direction.npz' if args.measured_margin else 'last_iterate.npz')
    extra_sources=([warm_file,previous/'optimizer.json',previous/'receipt.json']
                   if args.restore else [])
    extra_sources += distance_sources
    duration_file=event_source/'duration_guards.npz'
    duration_guards=[]
    if duration_file.exists():
        from terminal_duration_credit import TerminalDurationGuard
        extra_sources += [duration_file,BASE/'terminal_duration_credit.py']
        with np.load(duration_file) as z:
            for key in z.files:
                if key.endswith('_count'):
                    label=key[:-6]
                    duration_guards.append((label,TerminalDurationGuard(
                        z[label+'_state'],z[label+'_response'],int(z[key]))))
    if args.optimizer=='convex_cuts':extra_sources += [BASE/'convex_cut_solver.py']
    if args.corridor_model=='segment_capsules':
        capsule_audit=BASE/args.capsule_audit
        assert read(capsule_audit/'receipt.json')['status']=='verified_affine_capsule_corridor_enclosures'
        assert read(capsule_audit/'results.json')['parent_sha256']==sha(model)
        extra_sources += [BASE/'capsule_corridor_model.py',capsule_audit/'results.json',capsule_audit/'receipt.json']
    if args.start_from:
        initial=BASE/args.start_from
        assert read(initial/'optimizer.json')['parent_sha256']==sha(model)
        assert read(initial/'optimizer.json').get('parameter_geometry')==(geometry.identity if geometry else None)
        extra_sources += [initial/'direction.npz',initial/'optimizer.json',initial/'receipt.json']
    ideals=None
    if args.objective=='minimum_joint_regret':
        ideals={}
        for name,folder,field in [('shape',args.shape_ideal,'selected_predicted_worst_shape_fraction'),
                                  ('energy',args.energy_ideal,'selected_predicted_mean_energy_reduction')]:
            ideal_source=BASE/folder;record=read(ideal_source/'optimizer.json')
            assert record['eligible_for_small_probe'] and record['parent_sha256']==sha(model)
            assert record['parameter_geometry']==geometry.identity and record['preserve_cruise_states']==args.preserve_cruise_states
            assert record['optimization_objective']=='minimum_'+name
            assert record['metric_radius']==radius and record['requested_common_reduction']==fraction
            assert record['corridor_model']==args.corridor_model and record['energy_reduction_scope']==args.energy_scope
            assert record['empirical_error_model']==args.empirical_error_model
            assert record['preserve_cruise_deficit']==args.preserve_cruise_deficit
            assert record.get('cruise_reference_model') in (None,args.cruise_reference_model)
            assert record['required_per_corner_existing_energy_reduction']+record['required_aggregate_existing_energy_reduction']==args.energy_reduction
            bindings=read(ideal_source/'registration.json')['source_sha256']
            for bound in (source/'results.json',event_source/'results.json',probes/'results.json',
                          BASE/args.empirical_error_model):
                assert bindings[bound.relative_to(BASE.parent.parent).as_posix()]==sha(bound),'Ideals require the exact same corrected model'
            value=record[field] if name=='shape' else 1.-record[field]
            assert 0.<value<1.
            ideals[name]=value
            extra_sources += [ideal_source/'optimizer.json',ideal_source/'receipt.json',ideal_source/'registration.json']
        extra_sources += [BASE/'joint_regret_objective.py']
    if args.objective=='minimum_energy':
        assert args.start_from and not args.restore and args.energy_reduction==0.,'Minimize attainable energy from a feasible unchanged-shape proposal'
    if args.objective=='minimum_cruise_deficit':
        assert geometry and args.start_from and not args.restore and args.energy_reduction==0.
        if args.optimizer=='convex_cuts':assert geometry.positive_cruise_latent_domain
    if args.interpolate_last_from:
        assert args.objective in ('minimum_cruise_deficit','minimum_shape')
        interpolation_source=BASE/args.interpolate_last_from
        previous_result=read(interpolation_source/'optimizer.json')
        assert previous_result['parent_sha256']==sha(model)
        assert previous_result['parameter_geometry']==geometry.identity
        assert previous_result['optimization_objective']==args.objective
        assert previous_result['metric_radius']==radius
        assert previous_result['requested_common_reduction']==fraction
        assert previous_result['empirical_error_model']==args.empirical_error_model
        assert previous_result['preserve_cruise_states']==args.preserve_cruise_states
        bindings=read(interpolation_source/'registration.json')['source_sha256']
        for bound in (source/'results.json',probes/'results.json',event_source/'results.json'):
            assert bindings[bound.relative_to(BASE.parent.parent).as_posix()]==sha(bound)
        extra_sources += [interpolation_source/'last_iterate.npz',interpolation_source/'optimizer.json',interpolation_source/'receipt.json']
    if args.empirical_error_model:
        assert not args.measured_margin and ((args.restore_from and args.objective=='minimum_norm') or args.optimizer=='convex_cuts')
        error_source=BASE/args.empirical_error_model
        assert read(error_source)['parent_sha256']==sha(model)
        assert read(error_source).get('parameter_geometry')==(geometry.identity if geometry else None),'Error rates must use this exact parameter metric'
        extra_sources += [error_source,error_source.parent/'receipt.json']
        extra_sources += [error_source.parent/r['file'] for r in read(error_source).get('waveform_bounds',[])]
    measured_sources=[BASE/'canonical_gate_band_transition_credit_composition_20260917',
                      BASE/'canonical_gate_band_transition_credit_learned_update_20260917']
    if args.measured_margin:extra_sources += [p/n for p in measured_sources for n in ('results.json','receipt.json')]
    wall_budget=90 if args.measured_margin or args.empirical_error_model else 60 if args.restore else 180
    iteration_budget=50 if args.measured_margin or args.empirical_error_model else 30 if args.restore else 100
    if args.optimizer=='convex_cuts':wall_budget=180;iteration_budget=100
    if args.interpolate_last_from:wall_budget=60;iteration_budget=0
    stage_name=('canonical_gate_band_transition_measured_margin_solve_20260917' if args.measured_margin else
                'canonical_gate_band_transition_credit_feasibility_restore_20260917' if args.restore else
                'canonical_gate_band_transition_credit_step_solve_20260917')
    if args.stage:stage_name=args.stage
    stage=register(stage_name,
        [__file__,model,source/'results.json',source/'receipt.json',probes/'results.json',
         probes/'basis.npz',probes/'receipt.json',coverage/'results.json',coverage/'receipt.json',
         event_source/'results.json',event_source/'receipt.json',event_source/'constraints.npz',
         BASE/'waveform_constraint_model.py']+extra_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=wall_budget,
             optimizer_iterations=iteration_budget,feasibility_restoration=args.restore,
             measured_nonlinear_margin=args.measured_margin,
             energy_reduction_scope=args.energy_scope,
             required_per_corner_existing_energy_reduction=args.energy_reduction if args.energy_scope=='per_corner' else 0.,
             required_aggregate_existing_energy_reduction=args.energy_reduction if args.energy_scope=='aggregate' else 0.,
             corridor_model=args.corridor_model,
             optimization_objective=args.objective,feasible_initial_proposal=args.start_from,
             normalized_objective_ideals=ideals,
             optimizer=args.optimizer,
             maximum_exchange_rounds=40 if args.optimizer=='convex_cuts' else None,
             empirical_error_model=args.empirical_error_model,
             parameter_geometry=geometry.identity if geometry else None,
             distance_error_audit=args.distance_error_audit,
             feasible_interpolation_source=args.interpolate_last_from,
             redundant_capsule_endpoints=args.expose_capsule_endpoints,
             preserve_cruise_deficit=args.preserve_cruise_deficit,
             preserve_cruise_states=args.preserve_cruise_states,
             cruise_reference_model=args.cruise_reference_model,
             normalization_retains_original_benchmarks=bool(ideals and args.cruise_reference_model),
             cruise_request_deficit_ceiling=cruise_ceiling,
             shared_parameters=768,requested_common_reduction=fraction,maximum_metric_norm=radius,
             maximum_normalized_saved_action_change=action_bound,corridor_margin_consumption=.5,
             relative_curvature_interior_reserve=1e-6,terminal_trigger_signed_margin_fraction=.5),
        'Solve a bounded shared-parameter update using the exact current model and source hashes. '
        'Require15% reduction of the five shape totals, every existing per-corner shape and '
        'curvature/energy bound, the explicitly registered metric radius (at most0.1), saved-action support and terminal-event '
        'branch constraints. The corridor budget selects either displacement or the audited '
        'fixed-segment capsule model, always preserving half the nominal1.25mm margin. '
        'A separately supplied signed-distance error audit does not relabel a failed XY-vector audit. '
        'The parameter_geometry arm changes the update metric and support together, while '
        'keeping network, action map, guide and execution fixed. minimum_cruise_deficit '
        'minimizes the weighted squared99.5mm/s request deficit on saved straight states; '
        'actual cruise, time and shape must still pass complete physical evaluation. '
        'A supplied cruise_reference_model raises each saved-state feed floor to the maximum '
        'of the center and that retained shared policy. All three heads remain trainable. '
        'With a stronger floor, original normalization ideals are benchmarks from the old '
        'larger feasible set, not claimed attainable optima for the restricted problem. '
        'Optional preserve_cruise_deficit keeps this saved-state request loss no higher '
        'than its nominal value; all supported feed latents must be positive so the '
        'loss is convex on the bounded domain. This protects a learning proxy only, '
        'and actual complete cruise/time nonregression still requires verification. '
        'Optional preserve_cruise_states prevents feed-latent reduction at EACH of the '
        'same saved straight states, using exact linear residual features and the '
        'monotone runtime decoder. This disallows average-loss compensation between '
        'states, changes no runtime action rule, and still requires real cruise checks. '
        'minimum_energy minimizes the two existing normalized energy totals. The budget '
        'minimum_shape instead minimizes the largest remaining fraction of the existing '
        'positive shape totals (entry/exit/reversal/negative heading/heading overshoot), '
        'using a convex epigraph. Nominal zero totals stay protected as zero constraints. '
        'It changes the optimization objective alone, without a new shape metric, '
        'trajectory template or online controller rule. '
        'minimum_joint_regret minimizes the maximum of the shape and energy '
        'remaining fractions relative to their separately learned best feasible '
        'values on this exact corrected local model: (objective-ideal)/(1-ideal). '
        'This is a common-network multiobjective update, not online mixing of '
        'policies, trajectories or hand-selected corner actions. '
        'records any explicit energy reduction target and scope. Non-norm objectives start '
        'from a verified feasible same-geometry proposal and retain their best feasible iterate. '
        'Restoration minimizes distance to the saved last iterate and adds1e-6 normalized '
        'interior reserve to positive-bound constraints without relaxing zero bounds. '
        'An empirical error model uses only its bound current-center metric rates times '
        'squared update norm; it is not a guarantee in a new direction. '
        'Exposed capsule endpoints are redundant constraints retaining the original exact '
        'maximum, intended to retain nearly tied active gradients. Optional feasible-segment '
        'interpolation checks30 bisection points between the verified initializer and saved '
        'last iterate, then uses0.999 of the feasible fraction; it performs no optimizer update. '
        'All solver limits and mode flags are explicit in the budget. A feasible proposal '
        'requires fresh both-sign small complete-policy validation before larger learned steps. '
        'No failed-prefix time comparison, runtime template, test-set access or final adoption.')
    start=time.perf_counter();deadline=start+wall_budget-5
    with np.load(probes/'basis.npz') as z:metric=z['metric'].copy();support=z['support'].copy()
    if geometry:metric=geometry.metric;support=geometry.support
    with np.load(event_source/'constraints.npz') as z:
        event_matrix=z['matrix'].copy();event_bound=z['bound'].copy();event_labels=z['labels'].tolist()
    chol=np.linalg.cholesky(metric)
    base={r['path']:dict(zip(names,r['values'])) for r in physical['baseline']['metrics']}
    error_rates={};error_observations=[];waveform_error_rates={}
    if args.empirical_error_model:
        measured=read(error_source)
        for row in measured['coefficients']:
            assert row['path'] in base and row['metric'] in names and row['rate']>=0.
            error_rates[row['path'],row['metric']]=row['rate']
        for row in measured.get('waveform_bounds',[]):
            assert (row['channel'],row['sign'])==SIGNED_SHAPE_CHANNELS[row['metric']]
            with np.load(error_source.parent/row['file']) as z:
                rates=z['rate'].copy()
                valid=z['valid'].copy() if 'valid' in z else None
            assert np.all(rates>=0.)
            waveform_error_rates.setdefault(row['path'],[]).append(dict(row,rate=rates,valid=valid))
        write_json(stage/'error_model.json',measured)
    if args.measured_margin:
        with np.load(warm_file) as z:
            measured_direction=z['direction'].ravel();prediction_ids=z['path_id'].tolist()
            predictions=dict(zip(z['prediction_scales'],z['predictions']))
        measured_norm_squared=float(measured_direction@metric@measured_direction)
        targets={(7,'full_xy_peak_curvature_mm_inv'),(7,'full_physical_peak_curvature_mm_inv'),
                 (14,'heading_overshoot_deg')}
        for folder in measured_sources:
            for candidate in read(folder/'results.json')['candidates']:
                step=candidate['step'];assert candidate['summary']['successes']==29
                actual={r['path']:r['values'] for r in candidate['metrics']}
                for path,name in targets:
                    m=names.index(name)
                    error=actual[path][m]-base[path][name]-predictions[step][prediction_ids.index(path),m]
                    rate=max(0.,float(error))/(step**2*measured_norm_squared)
                    error_rates[path,name]=max(error_rates.get((path,name),0.),rate)
                    error_observations.append(dict(path=path,metric=name,step=step,error=float(error),quadratic_rate=rate))
        write_json(stage/'error_model.json',dict(parent_sha256=sha(model),observations=error_observations,
            coefficients=[dict(path=p,metric=n,rate=r) for (p,n),r in sorted(error_rates.items())],
            guaranteed_bound=False))
    waves={};xy={};margin={};capsules={};endpoint_indices={}
    if args.expose_capsule_endpoints:
        with np.load(warm_file) as z:endpoint_seed=z['direction'].ravel().copy()
    for record in data['records']:
        path=record['path']
        with np.load(source/record['xy_file']) as z:
            xy[path]=z['gradient'].copy();nominal_xy=z['nominal'].copy()
        margin[path]=.5*(1.25-record['segment_max_distance_mm'])
        assert margin[path]>0
        if args.corridor_model=='segment_capsules':
            j=next(j for j,r in enumerate(physical['baseline']['results']) if r['original_path']==path)
            with np.load(BASE/physical['baseline']['traces'][j]) as z:reference=z['reference'].copy()
            capsules[path]=CapsuleCorridorModel(reference,nominal_xy,xy[path],record['segment_max_distance_mm'])
            assert capsules[path].evaluate(np.zeros(768))[0]<1.25-margin[path]
            if args.expose_capsule_endpoints:
                distance=capsule_endpoint_facets(capsules[path],endpoint_seed)
                # Retain the exact maximum as well. These explicit endpoints
                # add no restriction, but avoid dropping tied active gradients.
                endpoint_indices[path]=np.argsort(distance)[-16:]
        if 'waveform_file' in record:
            with np.load(source/record['waveform_file']) as z:waves[path]={k:z[k].copy() for k in z.files}
    def values(x):
        output={}
        for p,w in waves.items():
            d=waveform_facets(w['nominal'],w['gradient'],w['valid'],float(w['turn_deg']),x)
            qx=w['energy_quadratic']@x
            energy=w['energy_constant']+w['energy_linear']@x+qx@x
            for j,name in enumerate(('full_xy_normalized_energy','full_physical_normalized_energy')):
                d[name]=(float(energy[j]),w['energy_linear'][j]+2*qx[j],0.)
            if error_rates:
                mx=metric@x;norm_squared=x@mx
                for name in list(d):
                    rate=error_rates.get((p,name),0.)
                    if rate:
                        value,row,offset=d[name]
                        d[name]=(value+rate*norm_squared,row+2*rate*mx,offset)
            if p in waveform_error_rates:
                mx=metric@x
                for item in waveform_error_rates[p]:
                    assert len(item['rate'])==len(w['nominal'])
                    valid=item['valid'] if item['valid'] is not None else w['valid'][:,item['channel']]
                    d[item['metric']]=corrected_signed_facet(w['nominal'],w['gradient'],x,mx,x@mx,
                        item['rate'],valid,item['channel'],item['sign'],item.get('offset',0.))
            output[p]=d
        return output
    def constraints_at(x):
        current=values(x);res=[];jac=[];scales=[];labels=[]
        def add(value,row,scale,label):
            if args.restore:
                positive_bound=(label.startswith('total_') or label.endswith(('_corridor_displacement','_corridor_capsule')) or '_capsule_endpoint_' in label or
                    label in ('saved_action_support','metric_radius','cruise_request_deficit') or label.startswith('path') or
                    any(label==f'{p}_{n}' and base[p][n]>0 for p in waves for n in names[:-2]))
                if positive_bound:value+=1e-6*scale
            res.append(value);jac.append(row);scales.append(scale);labels.append(label)
        for p,items in current.items():
            for name,(value,row,_) in items.items():
                bound=base[p][name]
                if 'normalized_energy' in name and args.energy_scope=='per_corner':bound*=1.-args.energy_reduction
                smooth='energy' in name or 'curvature' in name
                reserve=1e-6*bound if smooth else 0.
                add(value-bound+reserve,row,max(bound,1e-3),f'{p}_{name}')
        for name in IMPROVEMENT_NAMES:
            bound=sum(base[p][name] for p in waves)
            add(sum(current[p][name][0] for p in waves)-(1-fraction)*bound,
                sum((current[p][name][1] for p in waves),np.zeros(768)),max(bound,1e-3),'total_'+name)
        if args.energy_scope=='aggregate':
            for name in ('full_xy_normalized_energy','full_physical_normalized_energy'):
                bound=sum(base[p][name] for p in waves)
                add(sum(current[p][name][0] for p in waves)-(1.-args.energy_reduction)*bound,
                    sum((current[p][name][1] for p in waves),np.zeros(768)),bound,'total_'+name)
        for p,g in xy.items():
            if p in capsules:
                distance,derivative,_=capsules[p].evaluate(x)
                add(distance-(1.25-margin[p]),derivative,max(margin[p],1e-3),f'{p}_corridor_capsule')
                if p in endpoint_indices:
                    distances,derivatives=capsule_endpoint_facets(capsules[p],x,endpoint_indices[p])
                    for k,d,g in zip(endpoint_indices[p],distances,derivatives):
                        add(d-(1.25-margin[p]),g,max(margin[p],1e-3),f'{p}_capsule_endpoint_{k}')
            else:
                displacement=np.einsum('tcd,d->tc',g,x);norm=np.linalg.norm(displacement,axis=1)
                k=int(norm.argmax());derivative=(displacement[k]@g[k])/max(norm[k],1e-30)
                add(norm[k]-margin[p],derivative,max(margin[p],1e-3),f'{p}_corridor_displacement')
        for row,bound,label in zip(event_matrix,event_bound,event_labels):
            add(row@x-bound,row,max(bound,1e-3),label)
        for prefix,guard in duration_guards:
            for value,row,scale,label in guard.evaluate(x):
                add(value,row,scale,prefix+'_'+label)
        if cruise_ceiling is not None:
            loss,gradient=geometry.objective(x)
            add(loss-cruise_ceiling,gradient,max(cruise_ceiling,1e-3),'cruise_request_deficit')
        if args.preserve_cruise_states:
            for k,feature in enumerate(geometry.preservation_features):
                row=np.zeros(768);row[256:512]=-feature
                add(float(cruise_floor[k]-feature@x[256:512]),row,1.,f'cruise_state_{k}')
        if geometry:
            support_value,support_row=geometry.support_constraint(x)
            add(support_value,support_row,1.,'saved_action_support')
        else:
            actions=support@x;k=int(abs(actions).argmax())
            add(abs(actions[k])-action_bound,np.sign(actions[k])*support[k],action_bound,'saved_action_support')
        norm2=x@metric@x
        add(norm2-radius**2,2*metric@x,radius**2,'metric_radius')
        return np.asarray(res),np.asarray(jac),np.asarray(scales),labels
    cache={};progress=[];best_feasible={}
    energy_names=('full_xy_normalized_energy','full_physical_normalized_energy')
    energy_totals={n:sum(base[p][n] for p in waves) for n in energy_names}
    shape_totals={n:sum(base[p][n] for p in waves) for n in IMPROVEMENT_NAMES}
    shape_totals={n:v for n,v in shape_totals.items() if v>1e-8}
    def shape_objectives(y):
        current=values(cached(y)['x']);objectives=[];gradients=[]
        if args.shape_scope=='per_corner':
            for p,v in current.items():
                for n in IMPROVEMENT_NAMES:
                    if base[p][n]<=1e-8:continue
                    objectives.append(v[n][0]/base[p][n])
                    gradients.append(v[n][1]/base[p][n])
        else:
            for n,total in shape_totals.items():
                objectives.append(sum(v[n][0] for v in current.values())/total)
                gradients.append(sum((v[n][1] for v in current.values()),np.zeros(768))/total)
        return np.asarray(objectives),solve_triangular(chol,np.asarray(gradients).T,lower=True,check_finite=False).T
    def energy_objective(y):
        c=cached(y);current=values(c['x']);value=0.;gradient=np.zeros(768)
        for n in energy_names:
            value+=.5*sum(v[n][0] for v in current.values())/energy_totals[n]
            gradient+=.5*sum((v[n][1] for v in current.values()),np.zeros(768))/energy_totals[n]
        return value,solve_triangular(chol,gradient,lower=True,check_finite=False)
    def search_objective(y):
        if args.objective=='minimum_energy':return energy_objective(y)
        if args.objective in ('minimum_shape','minimum_joint_regret'):
            objectives,gradients=joint_objectives(y) if ideals else shape_objectives(y);k=int(objectives.argmax())
            return float(objectives[k]),gradients[k]
        value,gradient=geometry.objective(cached(y)['x'])
        return value,solve_triangular(chol,gradient,lower=True,check_finite=False)
    def joint_objectives(y):
        values,gradients=shape_objectives(y);energy,energy_gradient=energy_objective(y)
        return (normalized_regrets(values,energy,ideals),
                np.vstack([gradients/(1.-ideals['shape']),energy_gradient/(1.-ideals['energy'])]))
    def cruise_objective_facets(y):
        value,gradient=search_objective(y)
        return np.array([value]),gradient[None,:]
    def cached(y):
        if time.perf_counter()>deadline:raise TimeoutError('Transition-model solve deadline')
        if 'y' not in cache or not np.array_equal(y,cache['y']):
            x=solve_triangular(chol.T,y,lower=False,check_finite=False)
            r,j,s,l=constraints_at(x)
            cache.update(y=y.copy(),x=x,r=r,j=j,s=s,labels=l)
        return cache
    def callback(y):
        c=cached(y)
        if args.objective!='minimum_norm' and max(c['r']/c['s'])<=1e-7 and max(c['r'])<=1e-6 and np.linalg.norm(y)<=radius+1e-12:
            value=search_objective(y)[0]
            if not best_feasible or value<best_feasible['objective']:
                best_feasible.update(objective=value,y=y.copy())
        progress.append(dict(iteration=len(progress)+1,norm=float(np.linalg.norm(y)),
                             maximum_scaled_violation=float(max(0.,(c['r']/c['s']).max()))))
        if len(progress)%10==0:print('TRANSITION_STEP_SOLVE',progress[-1],flush=True)
        np.savez_compressed(stage/'last_iterate.npz',direction=c['x'])
    constraints=dict(type='ineq',fun=lambda y:-cached(y)['r']/cached(y)['s'],
        jac=lambda y:-solve_triangular(chol,cached(y)['j'].T,lower=True,check_finite=False).T/cached(y)['s'][:,None])
    passed=False;message='';fit_success=False;y=np.zeros(768);conic_diagnostics=None
    if args.restore:
        assert read(previous/'optimizer.json')['parent_sha256']==sha(model)
        with np.load(warm_file) as z:y=chol.T@z['direction'].ravel()
    if args.start_from:
        with np.load(initial/'direction.npz') as z:y=chol.T@z['direction'].ravel()
        if args.objective!='minimum_norm':
            c=cached(y)
            assert max(c['r']/c['s'])<=1e-7 and max(c['r'])<=1e-6 and np.linalg.norm(y)<=radius+1e-12,'Objective search must start feasible'
            best_feasible.update(objective=search_objective(y)[0],y=y.copy())
    target=y.copy() if args.restore else np.zeros(768)
    try:
        if args.optimizer=='convex_cuts' and not args.interpolate_last_from:
            from convex_cut_solver import solve_convex_cuts
            transform=solve_triangular(chol.T,np.eye(768),lower=False,check_finite=False)
            if args.objective=='minimum_energy':
                hessian=sum((w['energy_quadratic'][j]/energy_totals[n]
                    for w in waves.values() for j,n in enumerate(energy_names)),np.zeros((768,768)))
                hessian+=sum(error_rates.get((p,n),0.)/energy_totals[n] for p in waves for n in energy_names)*metric
                linear=sum((.5*w['energy_linear'][j]/energy_totals[n]
                    for w in waves.values() for j,n in enumerate(energy_names)),np.zeros(768))
                hessian=transform.T@hessian@transform;linear=transform.T@linear
            elif args.objective in ('minimum_shape','minimum_joint_regret','minimum_cruise_deficit'):
                hessian=np.zeros((768,768));linear=np.zeros(768)
            else:
                hessian=np.eye(768);linear=-target
            def oracle(y):
                c=cached(y)
                return c['r']/c['s'],(c['j']@transform)/c['s'][:,None]
            y,conic_diagnostics=solve_convex_cuts(hessian,linear,y,oracle,radius,deadline,callback,
                objective_facets=(cruise_objective_facets if args.objective=='minimum_cruise_deficit' else
                                  joint_objectives if ideals else shape_objectives)
                    if args.objective in ('minimum_shape','minimum_joint_regret','minimum_cruise_deficit') else None)
            message='Convex supporting-plane exchange finished; full constraints independently checked'
            fit_success=bool(conic_diagnostics['history'] and
                conic_diagnostics['history'][-1]['maximum_scaled_violation']<=1e-8)
        elif args.interpolate_last_from:
            # Every modeled constraint is convex. Search only the segment from
            # the verified feasible initialization to the saved last iterate.
            with np.load(interpolation_source/'last_iterate.npz') as z:end=chol.T@z['direction'].ravel()
            begin=y.copy();lo=0.;hi=1.
            for _ in range(30):
                beta=.5*(lo+hi);trial=(1.-beta)*begin+beta*end;c=cached(trial)
                good=max(c['r']/c['s'])<=1e-7 and max(c['r'])<=1e-6 and np.linalg.norm(trial)<=radius+1e-12
                if good:lo=beta
                else:hi=beta
            beta=.999*lo;y=(1.-beta)*begin+beta*end
            callback(y);message=f'Convex feasible-segment interpolation beta={beta:.12g}'
        else:
            fun=(lambda y:search_objective(y)[0]) if args.objective!='minimum_norm' else (lambda y:.5*np.sum((y-target)**2))
            jac=(lambda y:search_objective(y)[1]) if args.objective!='minimum_norm' else (lambda y:y-target)
            fit=minimize(fun,y,jac=jac,constraints=constraints,
                method='SLSQP',callback=callback,options=dict(ftol=1e-12,maxiter=iteration_budget))
            y=fit.x;message=str(fit.message);fit_success=bool(fit.success)
    except TimeoutError as exc:
        message=str(exc)
        if 'y' in cache:y=cache['y']
    if best_feasible:y=best_feasible['y']
    x=solve_triangular(chol.T,y,lower=False,check_finite=False)
    res,_,scales,labels=constraints_at(x)
    passed=bool(max(res/scales)<=1e-7 and max(res)<=1e-6 and np.linalg.norm(y)<=radius+1e-12)
    result=dict(parent_sha256=sha(model),model=str(model.relative_to(BASE)),path_ids=ids,
        metric_names=names,eligible_for_small_probe=passed,requested_common_reduction=fraction,
        predicted_common_reduction=fraction if passed else None,metric_radius=radius,
        energy_reduction_scope=args.energy_scope,
        required_per_corner_existing_energy_reduction=args.energy_reduction if args.energy_scope=='per_corner' else 0.,
        required_aggregate_existing_energy_reduction=args.energy_reduction if args.energy_scope=='aggregate' else 0.,
        corridor_model=args.corridor_model,
        optimization_objective=args.objective,
        shape_objective_scope=args.shape_scope,
        optimizer=args.optimizer,conic_diagnostics=conic_diagnostics,
        empirical_error_model=args.empirical_error_model,
        parameter_geometry=geometry.identity if geometry else None,
        distance_error_audit=args.distance_error_audit,
        original_vector_coverage_passed=event_coverage,
        feasible_interpolation_source=args.interpolate_last_from,
        redundant_capsule_endpoints=args.expose_capsule_endpoints,
        preserve_cruise_deficit=args.preserve_cruise_deficit,
        preserve_cruise_states=args.preserve_cruise_states,
        cruise_reference_model=args.cruise_reference_model,
        cruise_reference_model_sha256=sha(cruise_reference) if args.cruise_reference_model else None,
        normalized_ideals_from_original_larger_feasible_set=bool(ideals and args.cruise_reference_model),
        cruise_request_deficit_ceiling=cruise_ceiling,
        selected_predicted_mean_energy_reduction=(1.-best_feasible['objective']) if best_feasible and args.objective=='minimum_energy' else None,
        selected_predicted_worst_shape_fraction=best_feasible.get('objective') if args.objective=='minimum_shape' else None,
        normalized_objective_ideals=ideals,
        selected_predicted_joint_regret=best_feasible.get('objective') if ideals else None,
        baseline_cruise_request_deficit=geometry.objective(np.zeros(768))[0] if geometry else None,
        selected_cruise_request_deficit=geometry.objective(x)[0] if geometry else None,
        projection=dict(metric_norm=float(np.linalg.norm(y))),solver_success=fit_success,
        message=message,iterations=progress,maximum_raw_violation=float(max(0.,res.max())),
        violations=[dict(constraint=n,excess=float(r)) for n,r in zip(labels,res) if r>1e-8],
        maximum_normalized_action_change=float(geometry.support_constraint(x)[0]+1.) if geometry else float(abs(support@x).max()),
        prediction_kind='complete_closed_loop_transition_waveform',
        learning_method='closed_loop_transition_model_constrained_parameter_learning',goal_complete=False)
    if passed:
        def predict(scale):
            current=values(scale*x);out=np.zeros((len(ids),len(names)))
            for i,p in enumerate(ids):
                if p in current:
                    for n,(v,_,_) in current[p].items():out[i,names.index(n)]=v-base[p][n]
                out[i,names.index('segment_max_distance_mm')]=(capsules[p].evaluate(scale*x)[0]-base[p]['segment_max_distance_mm']
                    if p in capsules else np.linalg.norm(np.einsum('tcd,d->tc',xy[p],scale*x),axis=1).max())
            return out
        steps=np.array([-.125,.125,.5,1.])
        np.savez_compressed(stage/'direction.npz',direction=x.reshape(3,256),path_id=ids,
            metric_names=np.array(names),predicted_changes=predict(1.),prediction_scales=steps,
            predictions=np.array([predict(step) for step in steps]))
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_transition_model_parameter_solve',eligible_for_small_probe=passed,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('TRANSITION_STEP_RESULT',passed,'norm',np.linalg.norm(y),'message',message,flush=True)


if __name__=='__main__':main()

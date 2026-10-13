"""Bounded learned-policy steps after a current canonical composition check."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha
from joint_regret_objective import normalized_regrets


def compare(candidate,baseline,names,required_reduction=.1,required_energy_reduction=0.,energy_scope='per_corner',require_cruise_gain=False,preserve_actual_cruise=False,check_screen_time=True):
    if candidate['summary']['successes']!=len(candidate['results']):
        return dict(eligible=False,reason='incomplete_path',regressions=[])
    original={r['path']:r for r in baseline['metrics']}
    regressions=[]
    for row in candidate['metrics']:
        for index in range(names.index('segment_max_distance_mm')):
            change=row['values'][index]-original[row['path']]['values'][index]
            if change>1e-6:regressions.append(dict(path=row['path'],metric=names[index],change=change))
    aggregates=[]
    for name in IMPROVEMENT_NAMES:
        index=names.index(name)
        old=sum(r['values'][index] for r in baseline['metrics'])
        new=sum(r['values'][index] for r in candidate['metrics'])
        aggregates.append(dict(metric=name,baseline=old,value=new,
            passes=bool(new<=(1.-required_reduction)*old if old>1e-8 else new<=1e-8)))
    time_ok=not check_screen_time or candidate['summary']['full_time_s']<=1.02*baseline['summary']['full_time_s']
    energies=[]
    per_corner_reduction=required_energy_reduction if energy_scope=='per_corner' else 0.
    for row in candidate['metrics']:
        for name in ('full_xy_normalized_energy','full_physical_normalized_energy'):
            if name not in names:continue
            index=names.index(name);old=original[row['path']]['values'][index]
            if old<=1e-8:continue
            value=row['values'][index]
            energies.append(dict(path=row['path'],metric=name,baseline=old,value=value,
                fractional_reduction=1.-value/old,
                passes=bool(value<=(1.-per_corner_reduction)*old+1e-6)))
    energy_totals=[]
    for name in ('full_xy_normalized_energy','full_physical_normalized_energy'):
        group=[r for r in energies if r['metric']==name]
        if not group:continue
        old=sum(r['baseline'] for r in group);value=sum(r['value'] for r in group)
        energy_totals.append(dict(metric=name,baseline=old,value=value,
            fractional_reduction=1.-value/old,
            passes=bool(value<=(1.-required_energy_reduction)*old+1e-6)))
    shape_ok=bool(not regressions and all(r['passes'] for r in aggregates) and time_ok)
    energy_ok=all(r['passes'] for r in energies) and all(r['passes'] for r in energy_totals)
    cruise_gain=candidate['summary']['complete_cruise_s']['99']-baseline['summary']['complete_cruise_s']['99']
    time_gain=baseline['summary']['full_time_s']-candidate['summary']['full_time_s']
    cruise_ok=bool(cruise_gain>=.05-1e-12 and time_gain>=.01-1e-12) if require_cruise_gain else (
        bool(cruise_gain>=-1e-12 and time_gain>=-1e-12) if preserve_actual_cruise else True)
    return dict(eligible=bool(shape_ok and energy_ok and cruise_ok),
        requires_actual_cruise_gain=require_cruise_gain,preserves_actual_cruise=preserve_actual_cruise,actual_cruise99_gain_s=cruise_gain,
        actual_time_gain_s=time_gain,actual_cruise_time_passes=cruise_ok,
        original_shape_time_eligible=shape_ok,regressions=regressions,aggregates=aggregates,
        time_passes=time_ok,time_checked=check_screen_time,required_energy_reduction=required_energy_reduction,
        energy_scope=energy_scope,energy_checks=energies,energy_totals=energy_totals)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='canonical_gate_state_initialization_20260917')
    parser.add_argument('--arm',default='fitted')
    parser.add_argument('--source',default='canonical_gate_measured_margin_solve_20260917')
    parser.add_argument('--probe',default='canonical_gate_margin_composition_20260917')
    parser.add_argument('--stage',default='canonical_gate_learned_shape_update_20260917')
    parser.add_argument('--required-energy-reduction',type=float,default=0.)
    parser.add_argument('--required-shape-reduction',type=float,default=.1)
    parser.add_argument('--energy-scope',choices=('per_corner','aggregate'),default='per_corner')
    parser.add_argument('--scale-selection')
    parser.add_argument('--require-cruise-gain',action='store_true')
    parser.add_argument('--preserve-full-cruise',action='store_true')
    parser.add_argument('--efficiency-reference',help='Required full64 retained comparator for joint-regret rounds')
    parser.add_argument('--cruise-scope',choices=('screen_and_full','full_training'),default='screen_and_full')
    args=parser.parse_args()
    if args.cruise_scope=='full_training':assert args.require_cruise_gain
    assert 0.<=args.required_energy_reduction<=.2
    assert 0.<args.required_shape_reduction<=.1
    torch.set_num_threads(1)
    parent=BASE/args.parent;source=BASE/args.source;probe=BASE/args.probe
    parent_data=read(parent/'results.json')
    selected=next(r for r in parent_data['candidates']
        if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
    model=parent/selected['model'];optimization=read(source/'optimizer.json')
    assert not optimization.get('diagnostic_only',False),'Diagnostic directions are not learned proposals'
    assert optimization['parent_sha256']==sha(model)
    shape_priority=optimization.get('optimization_objective')=='minimum_shape'
    joint_priority=optimization.get('optimization_objective')=='minimum_joint_regret'
    assert bool(args.efficiency_reference)==joint_priority
    if joint_priority:
        assert args.preserve_full_cruise and not args.require_cruise_gain
        assert args.energy_scope=='aggregate' and args.required_energy_reduction==.01
        efficiency_source=BASE/args.efficiency_reference
        efficiency_data=read(efficiency_source/'results.json')
        efficiency_full=efficiency_data['full_candidate']
        assert len(efficiency_full['results'])==64 and efficiency_full['summary']['successes']==64
        assert sorted(r['original_path'] for r in efficiency_full['results'])==list(range(64))
        efficiency_model=efficiency_source/efficiency_data['chosen']['model']
        assert sha(efficiency_model)=='406729c098e21566ded12bad87100ade1226e5874e4647a8a54ba0287a19f856'
        ideals=optimization['normalized_objective_ideals']
    if optimization.get('optimization_objective')=='minimum_cruise_deficit':
        assert args.require_cruise_gain,'Request fitting cannot replace actual cruise/time acceptance'
    assert read(probe/'comparison.json')['passed_composition']
    bindings=read(probe/'registration.json')['source_sha256']
    assert bindings[(source/'direction.npz').relative_to(BASE.parent.parent).as_posix()]==sha(source/'direction.npz'),'Composition must validate this exact direction'
    assert optimization['projection']['metric_norm']<=optimization.get('metric_radius',optimization.get('empirical_radius'))+1e-12
    ids=[r['original_path'] for r in selected['results']]
    steps=[.5,1.];extra_sources=[]
    if joint_priority:
        extra_sources += [BASE/'joint_regret_objective.py',efficiency_model,
                          efficiency_source/'results.json',efficiency_source/'receipt.json']
    if args.scale_selection:
        selection_file=BASE/args.scale_selection;scale_selection=read(selection_file)
        assert scale_selection['parent_sha256']==sha(model)
        assert scale_selection['direction_sha256']==sha(source/'direction.npz')
        assert scale_selection['required_actual_energy_reduction']==args.required_energy_reduction
        assert scale_selection['actual_energy_scope']==args.energy_scope
        steps=scale_selection['selected_steps'];assert len(steps)==1 and .5<steps[0]<1.
        extra_sources=[selection_file,selection_file.parent/'receipt.json']
    stage=register(args.stage,[__file__,model,parent/'results.json',parent/'receipt.json',
        source/'optimizer.json',source/'direction.npz',source/'receipt.json',
        probe/'comparison.json',probe/'receipt.json',BASE/'canonical_gate_state.py',
        BASE/'continuous_gate_guidance.py',BASE/'gate_band_state.py',BASE/'probe_direct_constraint_composition.py',
        BASE/'train_pilot.py',BASE/'learning_preparation_v4/paths.json']+extra_sources,
        dict(parameter_updates=1,steps=steps,screen_paths=ids,scale_selection=args.scale_selection,
            energy_acceptance_scope=args.energy_scope,
            minimum_actual_energy_reduction=args.required_energy_reduction,
            minimum_actual_shape_reduction=args.required_shape_reduction,
            selection_objective='matched_normalized_joint_regret' if joint_priority else 'worst_existing_shape_fraction' if shape_priority else 'original_energy_or_time_selection',
            efficiency_reference=args.efficiency_reference,
            screen_time_checked=not joint_priority,
            requires_actual_cruise_gain=args.require_cruise_gain,
            preserves_full_actual_cruise=args.preserve_full_cruise,
            actual_cruise_acceptance_scope=args.cruise_scope,
            minimum_actual_cruise99_gain_s=.05 if args.require_cruise_gain else None,
            minimum_actual_full_time_gain_s=.01 if args.require_cruise_gain else None,
            maximum_additional_parent_paths=35,maximum_additional_candidate_paths=35,
            evaluation_actor_calls=16000,evaluation_physics_ticks=400000,wall_s=300),
        ('Joint-regret selection uses actual shape and mean energy ratios normalized by the matched model ideals. '
         'The29-route screen checks shape and energy only; complete64 time and cruise determine efficiency. '
         'Full efficiency must also meet the saved4067 retained comparator. This new registered rule does not revise old failures. '
         if joint_priority else '')+
        ('This shape-priority round selects the eligible step with the smallest worst '
         'remaining fraction among existing positive shape totals. Same-scale comparisons '
         'to the energy arm isolate the optimization objective; this selection change '
         'is reported separately. ' if shape_priority else '')+
        ('This round requires nonincreasing screen time, then nonincreasing complete64 time '
         'and nondecreasing measured vector-speed>=99mm/s duration. Select by the mean of the two '
         'actual normalized energy ratios on the training screen unless shape-priority '
         'selection is explicitly registered. A saved-state request constraint '
         'cannot substitute for this full physical check. ' if args.preserve_full_cruise and not joint_priority else '')+
        (('The29-route screen contains no pure straights: require its complete shape acceptance '
          'and nonincreasing time, then require the SAME0.05s actual99mm/s cruise gain and0.01s '
          'time gain on the FULL64 including16 straight routes. This scope is declared before '
          'evaluating this new direction; it does not revise older failed checks. ' if args.cruise_scope=='full_training' else
         'This cruise round additionally requires actual full recorded vector-speed>=99mm/s '
         'duration to increase by at least0.05s and complete time to decrease by at least0.01s, '
         'on BOTH the screen and full64 routes. Request fit and incomplete prefixes cannot pass. '
         ) if args.require_cruise_gain else '')+
        'The current canonical shared-feedback direction passed both-sign29-route '
        'small composition checks with its registered local prediction model. Apply the SAME '
        'learned direction at the explicitly registered scales, normally0.5 and1.0. '
        'A scale_selection instead binds exactly one evidence-selected intermediate '
        'scalar, its source direction hash and unchanged actual acceptance criteria. '
        'No online action '
        'interventions. Require29/29, every corner entry/exit/reverse/directed heading '
        'and four curvature quantities nonworse, all positive shape totals reduced by '
        'at least the registered minimum_actual_shape_reduction (default10%), '
        'and, except the explicitly registered joint-regret screen, full time<=102% of parent. When the budget specifies a positive '
        'minimum_actual_energy_reduction, BOTH existing rawXY and physical '
        'energies must achieve that actual fractional reduction on every '
        'measured corner for per_corner scope, or on their totals for aggregate '
        'scope. Every individual energy must remain nonworse in either scope. '
        'Prediction alone cannot pass. '
        'Select at most one using training results. '
        'If eligible, evaluate the missing35 training routes for both parent and '
        'candidate to provide a fair full64 comparison; reuse a matching saved '
        'full64 parent when available and do not substitute another '
        'architecture\'s parent. Stop this direction after these two attempts if it '
        'fails and analyze scale-dependent prediction errors. Working candidate '
        'and S2 remain preserved. No development or sealed access, no final goal claim.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    current_factory=checkpoint_environment_factory(metadata['architecture'])
    p=CanonicalGatePolicy(256)
    with np.load(source/'direction.npz') as z:
        direction=z['direction'].copy();names=z['metric_names'].tolist()
    paths=[np.asarray(r) for r in read(BASE/'learning_preparation_v4/paths.json')['training']]
    def enriched(rs,ts,precomputed_metrics=None):
        summary=review(rs,ts)
        metrics=metric_rows(rs,ts,summary,names) if precomputed_metrics is None else precomputed_metrics
        return dict(summary=summary,metrics=metrics,results=rs,
                    traces=[str(t.relative_to(BASE)) for t in ts])
    baseline=enriched(selected['results'],[BASE/t for t in selected['traces']])
    def run(indices,tag):
        rs=evaluate(p,[paths[i] for i in indices],stage,tag,None,costs,start+300,current_factory)
        assert costs['evaluation_actor_calls']<=16000 and costs['evaluation_physics_ticks']<=400000
        rs=[dict(r,original_path=i) for i,r in zip(indices,rs)]
        return enriched(rs,[stage/tag/f'{j}.npz' for j in range(len(indices))])
    candidates=[]
    for step in steps:
        p.load_state_dict(metadata['policy'])
        with torch.no_grad():p.residual_coefficients.add_(torch.tensor(step*direction,dtype=torch.float32))
        name=f'candidate_{step:g}.pt'
        save_torch(stage/name,dict(policy=p.state_dict(),architecture=metadata['architecture'],
            observation_encoding=metadata['observation_encoding'],guidance_encoding=metadata['guidance_encoding'],
            state_support_encoding=metadata.get('state_support_encoding'),
            parent_sha256=sha(model),centers=256,coefficient=step,
            learning_method=optimization.get('learning_method','canonical_state_current_constrained_counterfactual_credit')))
        candidate=run(ids,f'screen_{step:g}')
        candidate.update(step=step,model=name)
        candidate['checks']=compare(candidate,baseline,names,
            required_reduction=args.required_shape_reduction,
            required_energy_reduction=args.required_energy_reduction,energy_scope=args.energy_scope,
            require_cruise_gain=args.require_cruise_gain and args.cruise_scope=='screen_and_full',
            check_screen_time=not joint_priority)
        if joint_priority:
            check=candidate['checks']
            if check.get('aggregates') and check.get('energy_totals'):
                fractions=[r['value']/r['baseline'] for r in check['aggregates'] if r['baseline']>1e-8]
                energy=np.mean([r['value']/r['baseline'] for r in check['energy_totals']])
                check['actual_joint_regret']=float(max(normalized_regrets(fractions,energy,ideals)))
        if not joint_priority and (args.preserve_full_cruise or (args.require_cruise_gain and args.cruise_scope=='full_training')):
            nonincrease=bool(candidate['summary']['successes']==len(ids) and
                candidate['summary']['full_time_s']<=baseline['summary']['full_time_s']+1e-12)
            candidate['checks']['screen_time_nonincrease']=nonincrease
            candidate['checks']['eligible']=bool(candidate['checks']['eligible'] and nonincrease)
        candidates.append(candidate)
        write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,
            metric_names=names,costs=costs,goal_complete=False))
        check=candidate['checks']
        print('CANONICAL_LEARNED_STEP',step,'eligible',check['eligible'],
            'regressions',check['regressions'],'aggregates',check.get('aggregates'),
            'energy_totals',check.get('energy_totals'),flush=True)
    eligible=[r for r in candidates if r['checks']['eligible']]
    chosen=None;full_parent=None;full_candidate=None
    if eligible:
        if joint_priority:
            chosen=min(eligible,key=lambda r:r['checks']['actual_joint_regret'])
        elif shape_priority:
            chosen=min(eligible,key=lambda r:max(a['value']/a['baseline']
                for a in r['checks']['aggregates'] if a['baseline']>1e-8))
        elif args.preserve_full_cruise:
            chosen=min(eligible,key=lambda r:sum(e['value']/e['baseline'] for e in r['checks']['energy_totals']))
        else:
            chosen=min(eligible,key=lambda r:r['summary']['full_time_s']) if args.require_cruise_gain else max(eligible,key=lambda r:r['summary']['independent_zero_return'])
        write_json(stage/'screen_selection.json',dict(model=chosen['model'],step=chosen['step'],training_only=True))
        remaining=[i for i in range(64) if i not in ids]
        saved_full=parent_data.get('full_candidate')
        reuse_full=bool(saved_full and parent_data.get('chosen',{}).get('model')==selected['model']
            and len(saved_full['results'])==64 and saved_full['summary']['successes']==64)
        if not reuse_full:
            p.load_state_dict(metadata['policy']);parent_remaining=run(remaining,'parent_remaining')
        p.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        candidate_remaining=run(remaining,'candidate_remaining')
        def combine(a,b):
            pairs=sorted(zip(a['results']+b['results'],a['traces']+b['traces']),key=lambda r:r[0]['original_path'])
            metrics=sorted(a['metrics']+b['metrics'],key=lambda r:r['path'])
            return enriched([r for r,t in pairs],[BASE/t for r,t in pairs],metrics)
        full_parent=saved_full if reuse_full else combine(baseline,parent_remaining)
        write_json(stage/'parent_reuse.json',dict(reused_full_training=reuse_full,
            parent_model_sha256=sha(model),new_parent_physical_paths=0 if reuse_full else len(remaining)))
        full_candidate=combine(chosen,candidate_remaining)
        if full_parent['summary']['successes']==64:
            full_candidate['checks']=compare(full_candidate,full_parent,names,
                required_reduction=args.required_shape_reduction,
                required_energy_reduction=args.required_energy_reduction,energy_scope=args.energy_scope,
                require_cruise_gain=args.require_cruise_gain,preserve_actual_cruise=args.preserve_full_cruise)
        else:full_candidate['checks']=dict(eligible=False,reason='parent_incomplete_no_aggregate_comparison')
        if joint_priority:
            current=full_candidate['summary'];retained=efficiency_full['summary']
            efficiency_check=dict(reference_model_sha256=sha(efficiency_model),
                reference_full_time_s=retained['full_time_s'],candidate_full_time_s=current['full_time_s'],
                reference_cruise99_s=retained['complete_cruise_s']['99'],candidate_cruise99_s=current['complete_cruise_s']['99'],
                passes=bool(current['successes']==64 and current['full_time_s']<=retained['full_time_s']+1e-12
                    and current['complete_cruise_s']['99']>=retained['complete_cruise_s']['99']-1e-12))
            full_candidate['checks']['retained_efficiency']=efficiency_check
            full_candidate['checks']['eligible']=bool(full_candidate['checks']['eligible'] and efficiency_check['passes'])
        write_json(stage/'training_selection.json',dict(model=chosen['model'],parent=full_parent,
            training=full_candidate['summary'],results=full_candidate['results'],
            traces=full_candidate['traces'],checks=full_candidate['checks']))
    write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,chosen=chosen,
        full_parent=full_parent,full_candidate=full_candidate,metric_names=names,costs=costs,
        working_model_pointer_changed=False,goal_complete=False))
    receipt(stage,dict(status='completed_canonical_learned_shape_update',costs=costs,
        screen_eligible=len(eligible),full_training_eligible=bool(full_candidate and full_candidate['checks']['eligible']),
        wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()

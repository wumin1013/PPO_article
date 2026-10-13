"""Complete a monotone intermediate step and assess cumulative block progress."""
import time
import shutil
import argparse
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_canonical_credit_step import compare
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='canonical_gate_band_half_step_measured_learned_update_20260917')
    parser.add_argument('--parent',default='canonical_gate_band_quadratic_error_learned_update_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_monotone_block_learning_20260917')
    parser.add_argument('--step',type=float,default=.5)
    parser.add_argument('--from-composition',action='store_true')
    parser.add_argument('--require-cruise-gain',action='store_true')
    parser.add_argument('--preserve-full-cruise',action='store_true')
    parser.add_argument('--required-energy-reduction',type=float,default=0.)
    parser.add_argument('--energy-scope',choices=('per_corner','aggregate'),default='per_corner')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/args.source
    parent=BASE/args.parent
    data=read(source/'results.json');previous=read(parent/'results.json')
    selected=next(r for r in data['candidates'] if r['step']==args.step)
    direct_parent=previous['full_candidate'];block_anchor=previous['full_parent'];names=data['metric_names']
    if args.from_composition:
        assert read(source/'comparison.json')['passed_composition'] and args.step>0
        selected=dict(selected,model=f'step{args.step:+g}.pt')
        selected['checks']=compare(selected,previous['chosen'],names,required_reduction=0.)
    else:assert not selected['checks']['eligible']
    assert not selected['checks']['regressions']
    assert selected['summary']['successes']==29 and selected['checks']['time_passes']
    assert all(r['value']<r['baseline'] if r['baseline']>1e-8 else r['value']<=1e-8
        for r in selected['checks']['aggregates'])
    if args.preserve_full_cruise:
        assert selected['checks'].get('screen_time_nonincrease')
    if args.required_energy_reduction:
        assert selected['checks']['required_energy_reduction']==args.required_energy_reduction
        assert selected['checks']['energy_scope']==args.energy_scope
        assert all(r['passes'] for r in selected['checks']['energy_totals'])
    assert direct_parent['summary']['successes']==block_anchor['summary']['successes']==64
    model=source/selected['model'];metadata=torch.load(model,map_location='cpu',weights_only=False)
    parent_model=parent/previous['chosen']['model']
    assert metadata['parent_sha256']==sha(parent_model)
    stage=register(args.stage,
        [__file__,model,source/'results.json',source/'receipt.json',parent_model,
         parent/'results.json',parent/'receipt.json',BASE/'learn_canonical_credit_step.py',
         BASE/'complete_motion_metrics.py',BASE/'train_pilot.py',BASE/'learning_preparation_v4/paths.json'],
        dict(new_parameter_updates=0,additional_training_paths=35,evaluation_actor_calls=5000,
             evaluation_physics_ticks=150000,wall_s=120,original_screen_reused=True,
             cumulative_common_reduction=.1,physical_shape_slack_unchanged=1e-6),
        ('This energy extension additionally retains the source energy threshold and scope, '
         'nonincreasing screen and FULL64 time, and nondecreasing actual FULL64>=99mm/s cruise. '
         'Zero shape totals must remain zero within the unchanged1e-8 bound; strict reduction '
         'applies to positive totals, matching the existing documented block rule. '
         if args.preserve_full_cruise else '')+
        ('This cruise extension selects exactly the already measured positive small parameter '
         'step after larger steps failed. It is a shared learned-parameter model, without '
         'online action intervention. Require actual FULL64 cruise>=99mm/s gain of0.05s and '
         'complete time gain of0.01s against its exact direct parent, in addition to the '
         'existing cumulative-block and per-corner checks. Earlier failed large-step '
         'criteria remain unchanged. ' if args.require_cruise_gain else '')+
        'Reuse the prior monotone-block learning rule and exact29-route screen. '
        'intermediate updates must strictly improve every positive shape total '
        'and preserve each corner, while a cumulative learning block must still '
        'exceed the original10% common-reduction gate against its unchanged '
        'block anchor. Complete only the35 missing training paths for this '
        'already selected candidate, reusing exact complete parent and block '
        'anchor traces. Retain it only if full64 succeeds and both direct-parent '
        'monotonicity and cumulative block checks pass. No relaxed physical '
        'limit, shape comparison slack or final acceptance; no replacement of '
        'the preserved strong model, development or sealed access. This is '
        'an explicit training-algorithm revision, not a retroactive pass of '
        'the earlier per-step experiment or independent validation.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    if args.from_composition:
        selected=dict(selected,model=f'candidate_{args.step:g}.pt')
        save_torch(stage/selected['model'],dict(metadata,diagnostic_only=False,
            source_validated_parameter_model_sha256=sha(model),research_candidate=True))
    else:
        shutil.copy2(model,stage/selected['model']);assert sha(stage/selected['model'])==sha(model)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    screen_ids=[r['original_path'] for r in selected['results']]
    remaining=[p for p in range(64) if p not in screen_ids];assert len(remaining)==35
    rs=evaluate(policy,[np.asarray(paths[p]) for p in remaining],stage,'remaining',None,costs,
        start+120,checkpoint_environment_factory(metadata['architecture']))
    assert costs['evaluation_actor_calls']<=5000 and costs['evaluation_physics_ticks']<=150000
    rs=[dict(r,original_path=p) for r,p in zip(rs,remaining)]
    ts=[stage/'remaining'/f'{j}.npz' for j in range(len(remaining))]
    remainder_summary=review(rs,ts)
    remainder_metrics=metric_rows(rs,ts,remainder_summary,names)
    pairs=sorted(zip(selected['results']+rs,selected['traces']+[str(t.relative_to(BASE)) for t in ts]),
                 key=lambda p:p[0]['original_path'])
    results=[r for r,t in pairs];traces=[t for r,t in pairs]
    summary=review(results,[BASE/t for t in traces])
    metrics=sorted(selected['metrics']+remainder_metrics,key=lambda r:r['path'])
    complete=dict(summary=summary,metrics=metrics,results=results,traces=traces)
    monotone=compare(complete,direct_parent,names,required_reduction=0.,require_cruise_gain=args.require_cruise_gain,
        preserve_actual_cruise=args.preserve_full_cruise,
        required_energy_reduction=args.required_energy_reduction,energy_scope=args.energy_scope)
    strictly_improves=all(r['value']<r['baseline'] if r['baseline']>1e-8 else r['value']<=1e-8
        for r in monotone.get('aggregates',[]))
    cumulative=compare(complete,block_anchor,names,required_reduction=.1)
    eligible=bool(monotone['eligible'] and strictly_improves and cumulative['eligible'])
    complete['checks']=dict(eligible=eligible,direct_parent_monotonicity=monotone,
        each_total_strictly_improved=strictly_improves,cumulative_block=cumulative)
    selected=dict(selected,intermediate_step_monotone=True,original_per_step_gate_passed=False)
    write_json(stage/'results.json',dict(candidates=[selected],chosen=selected if eligible else None,
        full_parent=direct_parent,block_anchor=block_anchor,full_candidate=complete,metric_names=names,
        costs=costs,working_model_pointer_changed=False,goal_complete=False))
    if eligible:
        write_json(stage/'training_selection.json',dict(model=selected['model'],training=summary,
            results=results,traces=traces,checks=complete['checks'],research_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_monotone_intermediate_training_step',
        full_training_eligible=eligible,costs=costs,wall_s=time.perf_counter()-start,
        original_per_step_experiment_unchanged=True,goal_complete=False))
    print('MONOTONE_BLOCK',complete['checks'],flush=True)


if __name__=='__main__':main()

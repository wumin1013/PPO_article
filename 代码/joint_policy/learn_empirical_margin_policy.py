"""Bounded actual learned-policy update after complete composition validation."""
import argparse
import time
from datetime import datetime,timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from geometry_homotopy_policy import GeometryHomotopyPolicy
from learn_single_decision_credit import SCREEN, review, adoption_checks
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate, save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='empirical_curvature_margin_feasible_proposal_20260917')
    parser.add_argument('--probe',default='empirical_curvature_margin_composition_probe_20260917')
    parser.add_argument('--stage',default='empirical_margin_policy_learning_20260917')
    args=parser.parse_args()
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    source=BASE/args.source
    probe=BASE/args.probe
    optimization=read(source/'optimizer.json')
    per_corner=optimization.get('each_corner_exit_and_reverse_nonincrease',False)
    geometry_delta=optimization.get('geometry_mix_delta',0.)
    if abs(geometry_delta)<1e-10:geometry_delta=0.
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    composition=read(probe/'comparison.json');small=read(probe/'results.json')
    for row in composition['paths']:
        assert row['complete_pair']
        for name,value in zip(composition['metric_names'],row['actual_positive_change']):
            if ('energy' in name or 'peak_curvature' in name or name=='in_edge_outward_mm' or
                (per_corner and name in ('out_edge_outward_mm','full_physical_reverse_deg'))):assert value<=1e-6
    assert all(small['candidates'][1]['summary'][name]<small['baseline'][name]
        for name in ('entry_sum_mm','exit_sum_mm','full_reverse_sum_deg'))
    screen=SCREEN+[38,39,42,43,44,45,46,47]
    stage=register(args.stage,
        [__file__,model,parent/'training_selection.json',source/'direction.npz',source/'optimizer.json',
         source/'receipt.json',probe/'receipt.json',probe/'comparison.json',
         BASE/'probe_direct_constraint_composition.py',BASE/'audit_xy_curvature_resolution.py',
         BASE/'kernel_local_feedback_policy.py',BASE/'learn_single_decision_credit.py',
         BASE/'geometry_homotopy_policy.py',BASE/'continuous_tangent_policy.py',
         BASE/'train_pilot.py',BASE/'joint_runtime_v5.py',BASE/'learning_preparation_v4/paths.json'],
        dict(steps=[.5,1.],screen_paths=screen,evaluation_actor_calls=15000,
             evaluation_physics_ticks=350000,wall_s=240,parameter_updates=1),
        ('Require each individual exit and reverse to be nonworse in screening and full training, in addition to aggregate gains. ' if per_corner else '')+
        ('The same learned update also changes a single global representation blend; save its actual value and architecture. Partial mixing does not eliminate point-tangent discontinuities. ' if geometry_delta else '')+
        'The registered small complete-function check succeeds and preserves every entry and all '
        'four actual raw-XY/physical smoothness quantities, while reducing aggregate '
        'entry/exit/reverse. Now apply the one DATA-LEARNED shared-feedback residual '
        'direction at two predeclared steps, both within the empirical norm radius. '
        'This is constrained counterfactual policy improvement, NOT PPO or deployed '
        'probe schedules. Screen29 training routes including long routes and protected '
        'paths16/31. Retain existing actual adoption checks, add>=10% reverse reduction '
        'and each raw-XY/physical energy/peak nonincrease. Choose at most one candidate '
        'on training data, complete remaining35 routes, then at most one12-route '
        'development evaluation only if full training checks pass. Compare the result '
        'with the existing margin working candidate before changing any working-model '
        'pointer. No final shape/cruise/generalization claim, no sealed access, and '
        'no continuation of a failed direction beyond these two trials.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    with np.load(source/'direction.npz') as z:direction=z['direction'].copy();names=z['metric_names'].tolist()
    saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=GeometryHomotopyPolicy(256) if geometry_delta else KernelLocalFeedbackPolicy(256)
    geometry=read(BASE/'learning_preparation_v4/paths.json')
    training=[np.asarray(p) for p in geometry['training']]
    status_file=BASE/'当前执行状态.json';status=read(status_file)
    status['running']=status['active_experiment']=stage.name
    status['current'].update(state='empirical_margin_policy_learning_running',
        training_running=True,physical_diagnostic_running=True,diagnostic_stage=stage.name)
    write_json(status_file,status)
    def enrich(rs,traces):
        value=review(rs,traces);value['constraint_metrics']=metric_rows(rs,traces,value,names)
        return value
    def compare(value,old):
        checks=adoption_checks(value,old)
        if not value['comparable_complete_pool']:return checks
        baseline={r['path']:r['values'] for r in old['constraint_metrics']}
        for j,name in enumerate(names):
            if 'energy' in name or 'peak_curvature' in name or (per_corner and name in ('out_edge_outward_mm','full_physical_reverse_deg')):
                checks['each_'+name+'_nonworse']=all(r['values'][j]<=baseline[r['path']][j]+1e-6 for r in value['constraint_metrics'])
        checks['reverse_sum_reduced_ten_percent']=value['full_reverse_sum_deg']<=.9*old['full_reverse_sum_deg']
        return checks
    baseline=enrich([selection['results'][i] for i in screen],[BASE/selection['traces'][i] for i in screen])
    def run(ids,tag,paths=training):
        rs=evaluate(policy,[paths[i] for i in ids],stage,tag,None,costs,started+240,environment_factory)
        assert costs['evaluation_actor_calls']<=15000 and costs['evaluation_physics_ticks']<=350000
        return [dict(r,original_path=i) for r,i in zip(rs,ids)],[stage/tag/f'{j}.npz' for j in range(len(ids))]
    candidates=[];cache={}
    for step in (.5,1.):
        missing=policy.load_state_dict(saved['policy'],strict=not bool(geometry_delta))
        if geometry_delta:assert missing.missing_keys==['geometry_mix'] and not missing.unexpected_keys
        with torch.no_grad():
            policy.residual_coefficients.add_(torch.tensor(step*direction,dtype=torch.float32))
            if geometry_delta:policy.geometry_mix.fill_(step*geometry_delta)
        name=f'candidate_{step:g}.pt'
        save_torch(stage/name,dict(policy=policy.state_dict(),architecture='geometry_homotopy_kernel' if geometry_delta else 'kernel_local_feedback',
            centers=256,parent_sha256=sha(model),step=step,
            geometry_mix=step*geometry_delta,
            learning_method='constrained_empirical_credit_with_observed_smoothness_margin'))
        rs,traces=run(screen,f'screen_{step:g}');value=enrich(rs,traces)
        value.update(model=name,step=step,checks=compare(value,baseline))
        value['eligible']=all(value['checks'].values());candidates.append(value);cache[name]=(rs,traces)
        write_json(stage/'screen_results.json',dict(baseline=baseline,candidates=candidates,costs=costs))
        print('EMPIRICAL_MARGIN_LEARNING',step,{k:value[k] for k in ('successes','entry_sum_mm',
            'exit_sum_mm','full_reverse_sum_deg','full_time_s','eligible')},value['checks'],flush=True)
    chosen=None;full=None;development=None;dev_details=None
    eligible=[r for r in candidates if r['eligible']]
    if eligible:
        chosen=max(eligible,key=lambda r:r['independent_zero_return'])
        write_json(stage/'screen_selection.json',dict(selected=chosen,training_only=True))
        policy.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        rs,traces=run([i for i in range(64) if i not in screen],'remaining')
        old_rs,old_traces=cache[chosen['model']]
        pairs=sorted(zip(old_rs+rs,old_traces+traces),key=lambda pair:pair[0]['original_path'])
        rs,traces=map(list,zip(*pairs));full=enrich(rs,traces)
        old=enrich(selection['results'],[BASE/t for t in selection['traces']])
        full['checks']=compare(full,old)
        write_json(stage/'training_selection.json',dict(model=chosen['model'],training=full,
            results=rs,traces=[str(t.relative_to(BASE)) for t in traces]))
        if all(full['checks'].values()):
            dr,dt=run(list(range(12)),'development',[np.asarray(p) for p in geometry['validation']])
            development=enrich(dr,dt)
            dev_details=dict(results=dr,traces=[str(t.relative_to(BASE)) for t in dt])
    write_json(stage/'results.json',dict(candidates=candidates,chosen=chosen,full_training=full,
        development=development,development_details=dev_details,costs=costs,goal_complete=False,
        working_model_pointer_changed=False))
    receipt(stage,dict(status='completed_empirical_margin_policy_learning',costs=costs,
        screen_eligible=len(eligible),full_training_eligible=bool(full and all(full['checks'].values())),
        wall_s=time.perf_counter()-started,goal_complete=False))
    status=read(status_file);status['running']=status['active_experiment']=None
    status['current'].update(state='empirical_margin_policy_learning_complete',
        training_running=False,physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['updated_utc']=datetime.now(timezone.utc).isoformat()
    status['next_action']='Review actual learned XY and compare all metrics with the retained working candidate before any model promotion.'
    write_json(status_file,status)


if __name__=='__main__':main()

"""Matched function-update repair after measured preview/steering interference."""
import time
from datetime import datetime,timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from learn_single_decision_credit import review,adoption_checks
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def checks(value,baseline):
    result=adoption_checks(value,baseline)
    result['reverse_sum_reduced_ten_percent']=bool(value['comparable_complete_pool'] and
        value['full_reverse_sum_deg']<=.9*baseline['full_reverse_sum_deg'])
    return result


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    control=BASE/'long_pool_reverse_shape_credit_learning_20260917'
    probe=BASE/'long_pool_residual_function_credit_probe_20260917'
    diagnostic=BASE/'function_constraint_interference_audit_20260917'
    selection=read(parent/'training_selection.json');source=read(probe/'results.json')
    screen=[r['original_path'] for r in source['candidates'][0]['results']]
    stage=register('preview_preserving_function_update_20260917',
        [__file__,parent/selection['model'],parent/'training_selection.json',control/'direction.npz',
         control/'results.json',control/'receipt.json',probe/'results.json',
         diagnostic/'results.json',diagnostic/'receipt.json',diagnostic/'failed_command_segment_bounds.json',
         BASE/'learn_single_decision_credit.py',BASE/'kernel_local_feedback_policy.py'],
        dict(parameter_steps=[1.,1.5],screen_paths=screen,evaluation_actor_calls=12000,
             evaluation_physics_ticks=300000,wall_s=300,incremental_head_mask=[1,1,0]),
        'Single-factor intervention on the FAILED learned parameter-update composition: '
        'set only its preview coefficient increment to zero, keeping the same parent, '
        'learned steering/feed increments,two scales and29-route screen. The parent preview '
        'policy remains learned and state-dependent; this is one coordinate-block learning '
        'step, not fixed lookahead or a deployable hand rule. The measured path40 factorial '
        'showed preview+steering failure whereas steering+feed completes; path8 needs all '
        'three updated heads for failure. Check whether dropping that increment preserves '
        'the learned shape gains across the entire screen. Require both edge sums and full '
        'reversal each>=10% better, every entry nonworse, max exit nonworse, improved complete '
        'return and unchanged full physical checks. Select on training only, then missing35 '
        'training and at most12 development. Reject if gains disappear or failure moves. '
        'No credit-accuracy or final-shape claim from the ablation; old strong models remain.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(parent/selection['model'],map_location='cpu',weights_only=False)
    with np.load(control/'direction.npz') as z:direction=z['direction'].copy()
    direction[2]=0.
    p=KernelLocalFeedbackPolicy(centers=256)
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(r) for r in paths['training']]
    def run(indices,tag,geometry=training):
        rs=evaluate(p,[geometry[i] for i in indices],stage,tag,None,costs,started+300,environment_factory)
        assert costs['evaluation_actor_calls']<=12000 and costs['evaluation_physics_ticks']<=300000
        return [dict(r,original_path=i) for r,i in zip(rs,indices)],[stage/tag/f'{i}.npz' for i in range(len(indices))]
    baseline=source['baseline'];candidates=[];cache={}
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='preview_preserving_update_running',training_running=True,physical_diagnostic_running=True)
    status['running']=stage.name;status['active_experiment']=stage.name;write_json(status_path,status)
    for step in (1.,1.5):
        p.load_state_dict(saved['policy'])
        with torch.no_grad():p.residual_coefficients.add_(torch.tensor(step*direction,dtype=torch.float32))
        name=f'candidate_{step:g}.pt'
        save_torch(stage/name,dict(policy=p.state_dict(),architecture='kernel_local_feedback',centers=256,
            parent_sha256=sha(parent/selection['model']),step=step,incremental_head_mask=[1,1,0],
            learning_method='preview_preserving_complete_function_credit'))
        rs,traces=run(screen,f'screen_{step:g}');cache[name]=(rs,traces)
        r=dict(review(rs,traces),model=name,step=step);r['checks']=checks(r,baseline);r['eligible']=all(r['checks'].values())
        candidates.append(r);write_json(stage/'screen_results.json',dict(candidates=candidates,baseline=baseline,costs=costs))
        print('PREVIEW_PRESERVING',step,{k:r[k] for k in ('successes','full_time_s','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','eligible')},r['checks'],flush=True)
    eligible=[r for r in candidates if r['eligible']];chosen=None;full=None;dev=None
    if eligible:
        chosen=max(eligible,key=lambda r:r['independent_zero_return'])
        write_json(stage/'screen_selection.json',dict(selected=chosen,training_only=True))
        p.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        rs,traces=run([i for i in range(64) if i not in screen],'remaining')
        old_rs,old_traces=cache[chosen['model']]
        pairs=sorted(zip(old_rs+rs,old_traces+traces),key=lambda row:row[0]['original_path'])
        rs,traces=map(list,zip(*pairs));full=review(rs,traces);full['checks']=checks(full,selection['training'])
        write_json(stage/'training_selection.json',dict(model=chosen['model'],training=full,results=rs,traces=[str(t.relative_to(BASE)) for t in traces]))
        if all(full['checks'].values()):
            rs,traces=run(list(range(12)),'development',[np.asarray(r) for r in paths['validation']]);dev=review(rs,traces)
    write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,chosen=chosen,full_training=full,development=dev,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_preview_preserving_function_update',costs=costs,
        full_training_eligible=bool(full and all(full['checks'].values())),wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='preview_preserving_update_complete',training_running=False,physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)


if __name__=='__main__':main()

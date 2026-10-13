"""Bounded closed-loop test after the saved-data locality diagnostic."""
import time
from datetime import datetime, timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from learn_constrained_entry_credit import SCREEN, complete_summary, eligible
from train_pilot import evaluate, save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main(source_name='local_basis_saved_data_fit_20260916',
         stage_name='local_basis_closed_loop_feedback_20260916', purpose_extra=''):
    torch.set_num_threads(1)
    source=BASE/source_name
    parent=BASE/'exit_local_credit_parameter_learning_20260916'
    pool=parent/'training_1'
    fits=[r for r in read(source/'results.json')['kernels'] if r['offline_eligible']]
    assert len(fits)==1
    stage=register(stage_name,
        [__file__,BASE/'kernel_local_feedback_policy.py',BASE/'learn_constrained_entry_credit.py',
         source/'receipt.json',source/'results.json',source/fits[0]['model'],
         pool/'summary.json',parent/'candidate_1.pt',
         BASE/'learning_preparation_v4/paths.json',
         BASE/'matched_entry_interaction_feedback_20260916/training_results.json'],
        dict(training_evaluation_decisions=10000,development_decisions=1500,
             physics_ticks=250000,wall_s=240,steps=[.5,1.],screen_paths=SCREEN,
             selected_offline_width=fits[0]['width_multiplier']),
        purpose_extra+'Evaluate only the Gaussian hidden-state basis passing all offline fit gates. '
        'Use two predeclared half/full steps and the same21-path screen and full-extent '
        'eligibility as the preceding constrained-direction study: all paths complete, '
        'entry<=0.05mm, median exit at most80% parent, no increased maximum exit or reversal '
        'sum, lower complete time+full-shape cost. If eligible, select one step on training '
        'screen only, run43 missing paths, and at most one12-path development evaluation '
        'after full64 eligibility. This promotion is not final acceptance. Previous linear '
        'fit results are reused; labels and291 parameter count match, but Gaussian basis '
        'and its ridge geometry differ. Offline MSE does not establish physical success. '
        'Stop this fit family if neither step qualifies; no test-set access or deployed '
        'probe timing, corner lookup, optimized trajectory or angle-based controller.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    original=read(pool/'summary.json')['results'];traces=[pool/f'{i}.npz' for i in range(64)]
    baseline=complete_summary(original,traces)
    screen_base=complete_summary([dict(original[i],original_path=i) for i in SCREEN],[traces[i] for i in SCREEN])
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(p) for p in paths['training']]
    saved=torch.load(source/fits[0]['model'],map_location='cpu',weights_only=False)
    policy=KernelLocalFeedbackPolicy();records=[];saved_runs={}
    for step in (.5,1.):
        policy.load_state_dict(saved['policy'])
        with torch.no_grad():policy.residual_coefficients.mul_(step)
        model=f'candidate_{step:g}.pt'
        save_torch(stage/model,dict(saved,policy=policy.state_dict(),step=step))
        tag=f'screen_{step:g}'
        result=evaluate(policy,[training[i] for i in SCREEN],stage,tag,None,costs,started+240,environment_factory)
        result=[dict(r,original_path=i) for r,i in zip(result,SCREEN)]
        trace=[stage/tag/f'{i}.npz' for i in range(len(SCREEN))]
        row=dict(complete_summary(result,trace),model=model,step=step)
        row['screen_eligible']=eligible(row,screen_base)
        saved_runs[model]=(result,trace);records.append(row)
        write_json(stage/'screen_results.json',dict(baseline=screen_base,candidates=records,costs=costs))
        print('LOCAL_BASIS_SCREEN',step,{k:row[k] for k in ('successes','median_exit_mm',
             'max_exit_mm','max_entry_mm','full_reverse_sum_deg','screen_eligible')},flush=True)
    valid=[r for r in records if r['screen_eligible']]
    chosen=min(valid,key=lambda r:r['full_time_plus_full_shape_s']) if valid else None
    write_json(stage/'screen_selection.json',dict(selected=chosen,training_only=True))
    selected=None;development=None
    if chosen:
        policy.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        missing=[i for i in range(64) if i not in SCREEN]
        extra=evaluate(policy,[training[i] for i in missing],stage,'remaining_training',None,costs,started+240,environment_factory)
        extra=[dict(r,original_path=i) for r,i in zip(extra,missing)]
        old,oldtrace=saved_runs[chosen['model']]
        extra_trace=[stage/'remaining_training'/f'{i}.npz' for i in range(len(missing))]
        combined=sorted(zip(old+extra,oldtrace+extra_trace),key=lambda item:item[0]['original_path'])
        result,trace=map(list,zip(*combined));full=complete_summary(result,trace)
        full['eligible']=eligible(full,baseline)
        write_json(stage/'full_training.json',dict(summary=full,baseline=baseline,results=result,
            traces=[p.relative_to(BASE).as_posix() for p in trace]))
        if full['eligible']:selected=dict(model=chosen['model'],step=chosen['step'],summary=full)
    assert costs['evaluation_actor_calls']<=10000
    write_json(stage/'selection.json',dict(selected=selected,training_only=True,goal_complete=False))
    if selected:
        before=costs['evaluation_actor_calls']
        result=evaluate(policy,[np.asarray(p) for p in paths['validation']],stage,'development',None,costs,started+240,environment_factory)
        development=complete_summary(result,[stage/'development'/f'{i}.npz' for i in range(12)])
        assert costs['evaluation_actor_calls']-before<=1500
    assert costs['evaluation_physics_ticks']<=250000
    write_json(stage/'results.json',dict(selected=selected,development=development,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_local_state_basis_closed_loop_test',costs=costs,
        selected_model=selected['model'] if selected else None,wall_s=time.perf_counter()-started))
    status=read(BASE/'当前执行状态.json')
    for finished in (source,stage):
        status['completed'].append(dict(stage=finished.name,receipt_sha256=sha(finished/'receipt.json')))
    status['updated_utc']=datetime.now(timezone.utc).isoformat()
    status['current']=dict(state='review_local_basis_closed_loop',training_running=False,
        retained_model='exit_local_credit_parameter_learning_20260916/candidate_1.pt',executor='unchanged_original')
    write_json(BASE/'当前执行状态.json',status)


if __name__=='__main__':main()
